"""Pure tap11 extraction for fixed windows, with observable adjacency checks.

No IO, role decisions, window selection, model fitting or threshold tuning.
This helper never claims that absence of recorded events proves no physical
unrecorded touch occurred. It refuses to bridge observed invalid events.
"""
from collections import Counter
import numpy as np
from hmog_contact_parser import parse_contacts
from hmog_tap_reference import Contact, tap11, scan_mean


def tap_window(rows, *, session, activity, start_ms, end_ms, conventions):
    """Return tap vectors and diagnostics for an already frozen recording window.

    Supply the complete authorized activity touch stream, not a filtered list of
    valid contacts: invalid events are needed to reject false adjacency. Both
    preceding and current contacts must lie wholly within [start,end). No
    predecessor from another activity, session or window is borrowed. Each
    scored pair is portrait, single-pointer throughout, native-time ordered and
    nonoverlapping. Complete but overlapping/multipointer contacts remain
    unscorable; they are never serialized into an invented sequence.

    A pair is adjacent only when ALL recorded events between the predecessor's
    DOWN and current UP are exactly their two contact-row multisets. Thus an
    orphan UP/DOWN/MOVE, CANCEL, OUTSIDE, unknown action, duplicate, repeated
    DOWN or invalid intermediate contact prevents scoring across that gap.
    A later clean consecutive pair can recover without revising any event.
    """
    conventions.validate()
    if not isinstance(session,str) or not session:
        raise ValueError('Explicit session identity required')
    if not np.isfinite([start_ms,end_ms]).all() or end_ms<=start_ms:
        raise ValueError('Finite increasing window bounds required')
    a=np.asarray(rows,dtype=np.float64)
    if a.size==0:a=np.empty((0,11))
    if a.ndim!=2 or a.shape[1]!=11 or not np.isfinite(a).all():
        raise ValueError('Finite complete 11-column touch stream required')
    if len(a) and np.any(a[:,2]!=activity):
        raise ValueError('One exact activity required; do not merge timelines')
    contacts,parser_rejections=parse_contacts(a.tolist())
    contacts=sorted(contacts,key=lambda c:c[0][1])
    vectors=[];rejected=Counter();eligible_contacts=0
    def inside(c):return all(start_ms<=r[0]<end_ms for r in c)
    for index,current in enumerate(contacts):
        if not inside(current):
            rejected['contact_outside_frozen_window']+=1;continue
        eligible_contacts+=1
        if index==0:
            rejected['no_previous_observed_complete_contact']+=1;continue
        previous=contacts[index-1]
        if not inside(previous):
            rejected['previous_contact_outside_window']+=1;continue
        if previous[-1][1]>current[0][1]:
            rejected['overlapping_contacts']+=1;continue
        if any(r[3]!=1 or r[10]!=0 for c in (previous,current) for r in c):
            rejected['non_single_pointer_or_nonportrait_pair']+=1;continue
        native_lo,native_hi=previous[0][1],current[-1][1]
        observed=Counter(tuple(r) for r in a if native_lo<=r[1]<=native_hi)
        required=Counter(tuple(r) for c in (previous,current) for r in c)
        if observed!=required:
            rejected['intervening_or_duplicate_recorded_event']+=1;continue
        chronological=sorted(previous+current,key=lambda r:r[1])
        if any(v[0]<u[0] for u,v in zip(chronological,chronological[1:])):
            rejected['backward_recording_time']+=1;continue
        def as_contact(c):
            return Contact(session=session,event_ms=tuple(r[1] for r in c),
                x_pixels=tuple(r[6] for r in c),y_pixels=tuple(r[7] for r in c),sizes=tuple(r[9] for r in c))
        try:
            vector=tap11(as_contact(current),as_contact(previous),conventions=conventions)
        except ValueError:
            rejected['reference_feature_unscorable']+=1;continue
        vectors.append(vector)
    matrix=np.asarray(vectors,dtype=np.float64).reshape(-1,11)
    return {'tap_vectors':matrix,'scan_mean':scan_mean(matrix) if len(matrix)>=5 else None,
            'scorable_taps':len(matrix),'complete_contacts_inside_window':eligible_contacts,
            'rejections':dict(rejected),'parser_rejections':parser_rejections,
            'no_recorded_gap_bridging':True,'minimum_taps_for_scan':5,
            'scan_scorable':len(matrix)>=5}
