"""Authorize BrainRun TRAIN-fit identities before decoding behavioral BSON."""
import json
import struct

from brainrun_bson_metadata import (
    MAX_DOCUMENT_BYTES, _limit, _read_exact, extract_metadata,
)


def canonical(value):
    """Match the frozen identity ledger's type-preserving JSON keys."""
    return json.dumps(value, ensure_ascii=True, separators=(',', ':'))


def _validate_identifier(value):
    if not isinstance(value, str):
        raise ValueError('Canonical typed identifier strings required')
    try:
        decoded = json.loads(value)
    except (ValueError, TypeError) as error:
        raise ValueError('Malformed canonical identifier') from error
    if not isinstance(decoded, list) or len(decoded) != 2 or canonical(decoded) != value:
        raise ValueError('Noncanonical typed identifier')
    tag, item = decoded
    valid = ((tag == 'string' and isinstance(item, str))
             or (tag == 'objectid' and isinstance(item, str) and len(item) == 24
                 and all(c in '0123456789abcdef' for c in item))
             or (tag == 'int32' and type(item) is int and -(2**31) <= item < 2**31)
             or (tag == 'int64' and type(item) is int and -(2**63) <= item < 2**63))
    if not valid:
        raise ValueError('Invalid typed identifier')


def iter_authorized(stream, device_to_user, roles, allowed_users, decoder, *,
                    max_selected_records=None, max_per_user=None,
                    max_document_bytes=MAX_DOCUMENT_BYTES):
    """Yield (canonical_user, decoded_record) for explicitly allowed TRAIN-fit.

Only identity fields are extracted before authorization. All other records are
opaque to decoder, including other TRAIN roles, DEV, test and inactive users.
    Optional total/per-user caps stop before framing the next document once
    the total cap or every allowed user's cap is met. Already capped users
    remain opaque to decoder while other allowed users are still collected.
This helper accepts a frozen `roles['users']` mapping, not the enclosing report.
"""
    _limit(max_document_bytes)
    if (max_selected_records is not None
            and (type(max_selected_records) is not int or max_selected_records < 1)):
        raise ValueError('Selected-record cap must be a positive integer')
    if max_per_user is not None and (type(max_per_user) is not int or max_per_user < 1):
        raise ValueError('Per-user cap must be a positive integer')
    if not callable(decoder) or not isinstance(device_to_user, dict) or not isinstance(roles, dict):
        raise ValueError('Decoder and explicit identity/role mappings required')
    if isinstance(allowed_users, (str, bytes)):
        raise ValueError('Explicit allowed-user collection required')
    allowed = frozenset(allowed_users)
    if not allowed or not allowed <= roles.keys():
        raise ValueError('Nonempty known allowed-user subset required')
    for user in allowed:
        _validate_identifier(user)
        role = roles[user]
        if (role.get('active') is not True or role.get('identity_role') != 'fit'
                or role.get('partition') != 'train'):
            raise ValueError('Allowed users must be active TRAIN-fit identities')
    for device, user in device_to_user.items():
        _validate_identifier(device); _validate_identifier(user)
        if user not in roles:
            raise ValueError('Device maps to unknown identity')
    if not allowed <= set(device_to_user.values()):
        raise ValueError('Every allowed identity needs a mapped device')
    selected = 0
    selected_by_user = dict.fromkeys(allowed, 0)
    while ((max_selected_records is None or selected < max_selected_records)
           and (max_per_user is None or any(n < max_per_user for n in selected_by_user.values()))):
        prefix = _read_exact(stream, 4, clean_eof=True)
        if prefix is None:
            return
        size = struct.unpack('<i', prefix)[0]
        if not 5 <= size <= max_document_bytes:
            raise ValueError('Document size outside configured bound')
        document = prefix + _read_exact(stream, size - 4)
        metadata = extract_metadata(document, fields=('device_id', 'user_id'),
                                    max_document_bytes=max_document_bytes)
        if 'device_id' not in metadata:
            raise ValueError('Record device identity required')
        device = canonical(metadata['device_id'])
        if device not in device_to_user:
            raise ValueError('Unknown record device')
        user = device_to_user[device]
        if 'user_id' in metadata and canonical(metadata['user_id']) != user:
            raise ValueError('Record user conflicts with frozen device mapping')
        if user not in allowed:
            continue
        if max_per_user is not None and selected_by_user[user] >= max_per_user:
            continue
        # Recheck immediately before invoking user-supplied decoding code.
        role = roles[user]
        if role.get('active') is not True or role.get('identity_role') != 'fit' or role.get('partition') != 'train':
            raise ValueError('TRAIN-fit authorization changed')
        selected += 1
        selected_by_user[user] += 1
        yield user, decoder(document)
