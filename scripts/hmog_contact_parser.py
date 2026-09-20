"""Pure candidate-v3 per-pointer contacts; no IO or authorization to extract."""
from collections import Counter
import math


def parse_contacts(rows):
    """Rows: Sys,event,activity,count,pointer,action,x,y,pressure,size,orientation.

    Uses native event time, stable original ordering for equal times. Retains
    physical DOWN/POINTER_DOWN and UP/POINTER_UP; never fabricates endpoints.
    CANCEL clears all active pointers in that activity. Nonportrait events
    invalidate the affected pointer until release. Multitouch count alone
    does not invalidate a contact. Output contacts contain original rows.
    """
    ordered=sorted(enumerate(rows),key=lambda item:(item[1][1],item[0]))
    duplicates=Counter(tuple(r[1:]) for r in rows)
    active={};done=[];reasons=Counter()
    for _,row in ordered:
        activity,pointer,action=row[2],row[4],row[5]
        identity=(activity,pointer)
        if action==3:
            for key in list(active):
                if key[0]==activity:
                    reasons['cancelled_contacts']+=1
                    del active[key]
            continue
        dup=duplicates[tuple(row[1:])]>1
        valid_count=math.isfinite(row[3]) and row[3]>0 and row[3]==int(row[3])
        if action in (0,5):
            if identity in active:
                reasons['repeated_down']+=1
                active[identity][1]=False
                active[identity][0].append(row)
            else:
                active[identity]=[[row],not dup and row[10]==0 and valid_count]
        elif action==2:
            if identity in active:
                state=active[identity]
                state[0].append(row)
                state[1]=state[1] and not dup and row[10]==0 and valid_count
        elif action in (1,6):
            state=active.pop(identity,None)
            if state is None:
                reasons['missing_down']+=1;continue
            state[0].append(row)
            times=[r[1] for r in state[0]]
            if state[1] and not dup and row[10]==0 and valid_count and all(b>a for a,b in zip(times,times[1:])):
                done.append(state[0])
            else:reasons['invalid_cycle']+=1
        elif identity in active:
            active[identity][1]=False
            reasons['unknown_or_outside_action']+=1
    reasons['missing_up']=len(active)
    return done,dict(reasons)
