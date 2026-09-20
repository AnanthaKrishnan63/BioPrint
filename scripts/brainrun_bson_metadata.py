"""Read allowlisted top-level identity fields without decoding BSON records.

Framing follows BSON1.1 https://bsonspec.org/spec.html. This is deliberately
not a full BSON validator: behavioral values remain opaque, including nested
documents, invalid UTF8 values, and nonfinite floating-point payloads.
"""
import struct

MAX_DOCUMENT_BYTES = 16 * 1024 * 1024
DEFAULT_FIELDS = ('_id', 'device_id', 'user_id')


def _limit(value):
    if type(value) is not int or not 5 <= value <= MAX_DOCUMENT_BYTES:
        raise ValueError('Document limit must be5..16MiB')


def _fields(fields):
    if isinstance(fields, (str, bytes)):
        raise ValueError('Explicit identity-field collection required')
    fields = tuple(fields)
    if len(set(fields)) != len(fields) or any(not isinstance(f, str) or not f or '\0' in f for f in fields):
        raise ValueError('Unique nonempty identity-field names required')
    return frozenset(fields)


def _end(position, size, limit):
    if size < 0 or position + size > limit:
        raise ValueError('Truncated or negative BSON value length')
    return position + size


def _i32(data, position, limit):
    _end(position, 4, limit)
    return struct.unpack_from('<i', data, position)[0]


def _cstring_end(data, position, limit):
    stop = data.find(b'\0', position, limit)
    if stop < 0:
        raise ValueError('Unterminated BSON cstring')
    return stop + 1


def _string_end(data, position, limit):
    size = _i32(data, position, limit)
    if size < 1:
        raise ValueError('BSON string must include terminator')
    end = _end(position + 4, size, limit)
    if data[end - 1] != 0:
        raise ValueError('BSON string terminator missing')
    return end


def _value_end(data, kind, position, limit):
    fixed = {1: 8, 6: 0, 7: 12, 8: 1, 9: 8, 10: 0,
             16: 4, 17: 8, 18: 8, 19: 16, 127: 0, 255: 0}
    if kind in fixed:
        return _end(position, fixed[kind], limit)
    if kind in (2, 13, 14):
        return _string_end(data, position, limit)
    if kind in (3, 4, 15):
        size = _i32(data, position, limit)
        if size < (14 if kind == 15 else 5):
            raise ValueError('Invalid nested BSON framing length')
        # Skip entire nested document/array/code-with-scope, never recurse.
        return _end(position, size, limit)
    if kind == 5:
        size = _i32(data, position, limit)
        return _end(position + 5, size, limit)  # length, subtype, payload
    if kind == 11:
        return _cstring_end(data, _cstring_end(data, position, limit), limit)
    if kind == 12:
        return _end(_string_end(data, position, limit), 12, limit)
    raise ValueError(f'Unsupported BSON type0x{kind:02x}')


def extract_metadata(document, fields=DEFAULT_FIELDS, *, max_document_bytes=MAX_DOCUMENT_BYTES):
    """Return field→(type_tag,value); absent allowlisted fields remain absent.

Only string, ObjectId, int32 and int64 identities are accepted. Type tags
preserve BSON identity types; they must not be stripped when joining records.
"""
    _limit(max_document_bytes)
    allowed = _fields(fields)
    if not isinstance(document, bytes) or not 5 <= len(document) <= max_document_bytes:
        raise ValueError('Bounded complete BSON bytes required')
    if _i32(document, 0, len(document)) != len(document) or document[-1] != 0:
        raise ValueError('Document length or final terminator mismatch')
    position, limit = 4, len(document) - 1
    seen, output = set(), {}
    while position < limit:
        kind = document[position]
        position += 1
        name_end = _cstring_end(document, position, limit)
        try:
            name = document[position:name_end - 1].decode('utf8', errors='strict')
        except UnicodeDecodeError as error:
            raise ValueError('Invalid UTF8 top-level field name') from error
        if name in seen:
            raise ValueError('Duplicate top-level BSON field')
        seen.add(name)
        position = name_end
        end = _value_end(document, kind, position, limit)
        if name in allowed:
            if kind == 2:
                try:
                    value = document[position + 4:end - 1].decode('utf8', errors='strict')
                except UnicodeDecodeError as error:
                    raise ValueError('Invalid UTF8 identity') from error
                output[name] = ('string', value)
            elif kind == 7:
                output[name] = ('objectid', document[position:end].hex())
            elif kind in (16, 18):
                output[name] = ('int32' if kind == 16 else 'int64',
                                struct.unpack_from('<i' if kind == 16 else '<q', document, position)[0])
            else:
                raise ValueError('Allowlisted identity has unsupported BSON type')
        position = end
    if position != limit:
        raise ValueError('Document element boundary mismatch')
    return output


def _read_exact(stream, size, *, clean_eof=False):
    data, received = bytearray(), 0
    while received < size:
        part = stream.read(size - received)
        if not isinstance(part, bytes):
            raise ValueError('Binary stream required')
        if not part:
            if clean_eof and received == 0:
                return None
            raise ValueError('Truncated BSON stream')
        if len(part) > size - received:
            raise ValueError('Stream returned more than requested')
        data.extend(part)
        received += len(part)
    return bytes(data)


def iter_metadata(stream, fields=DEFAULT_FIELDS, *, max_document_bytes=MAX_DOCUMENT_BYTES):
    """Yield identity metadata from concatenated BSON with bounded reads."""
    _limit(max_document_bytes)
    fields = _fields(fields)
    while True:
        prefix = _read_exact(stream, 4, clean_eof=True)
        if prefix is None:
            return
        size = struct.unpack('<i', prefix)[0]
        if not 5 <= size <= max_document_bytes:
            raise ValueError('Document size outside configured bound')
        payload = _read_exact(stream, size - 4)
        yield extract_metadata(prefix + payload, fields, max_document_bytes=max_document_bytes)
