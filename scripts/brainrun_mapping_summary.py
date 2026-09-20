"""Summarize identifier-only device mappings; no behavior or context decoding."""
from collections import Counter


def summarize(mapping, records):
    mapping={device:sorted(set(users)) for device,users in mapping.items()}
    if not mapping or any(not users for users in mapping.values()):
        raise ValueError('Nonempty device mappings required')
    users=set().union(*(set(v) for v in mapping.values()))
    shared={d:u for d,u in mapping.items() if len(u)>1}
    affected=set().union(*(set(v) for v in shared.values())) if shared else set()
    return {'records':records,'devices':len(mapping),'users':len(users),'shared_devices':len(shared),
        'affected_users':len(affected),'users_without_shared_devices':len(users-affected),
        'users_per_device_histogram':dict(Counter(len(u) for u in mapping.values())),
        'device_to_users':mapping}
