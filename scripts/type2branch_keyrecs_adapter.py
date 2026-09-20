"""KeyRecs source-order base channels; explicit adaptation, not full Type2Branch."""
import ast
import csv
from functools import lru_cache
import math
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def mapper():
    path=ROOT/'scripts/typenet_benchmark.py'
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if
           (isinstance(n,ast.FunctionDef) and n.name=='keycode') or
           (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='KEYS' for t in n.targets))]
    if len(nodes)!=2:raise ValueError('Expected reviewed mapping definitions')
    scope={}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    return scope['keycode']


def milliseconds(value):
    x=float(value)*1000
    if not math.isfinite(x) or abs(x-round(x))>1e-6:
        raise ValueError('Timing requires nontrivial quantization')
    if not -(2**31)<=round(x)<2**31:raise ValueError('Int32 timing overflow')
    return int(round(x))


def parse_event(opaque, *, last, details=False):
    suffix=opaque.rstrip('\r\n')
    if suffix.endswith(','):suffix=suffix[:-1]
    fields=suffix.rsplit(',',5)
    if len(fields)!=6:raise ValueError('Wrong event field count')
    prefix,*timing=fields
    unknown=False
    try:
        keys=next(csv.reader([prefix],strict=True))
        if len(keys)!=2:raise ValueError('Wrong key count')
    except (csv.Error,StopIteration,ValueError):
        keys=[None,None];unknown=True
    terminal=bool(last and keys[0] and keys[1]=='' and timing[0] and all(v=='' for v in timing[1:]))
    if not unknown and (not keys[0] or (not keys[1] and not terminal)):
        raise ValueError('Missing nonterminal key')
    if any(v=='' for v in timing) and not terminal:
        raise ValueError('Missing interior or ambiguous terminal timing')
    ht=milliseconds(timing[0])
    if ht<0:raise ValueError('Negative hold')
    if terminal:
        dd=None; next_hold=None
    else:
        ht,dd,du,ud,uu=map(milliseconds,timing)
        if ht+ud!=dd or ht+uu!=du:raise ValueError('Inconsistent digraph timings')
        next_hold=du-dd
    code=0 if unknown else mapper()(keys[0])
    result=(code,ht,dd,unknown,terminal)
    return result+(keys,next_hold) if details else result


def adapt(rows):
    events=[parse_event(row,last=i==len(rows)-1,details=True) for i,row in enumerate(rows)]
    unchecked_keys=0
    for previous,current in zip(events,events[1:]):
        if previous[6]!=current[1]:raise ValueError('Adjacent inferred hold mismatch')
        if previous[5][1] is None or current[5][0] is None:
            unchecked_keys+=1
        elif previous[5][1]!=current[5][0]:raise ValueError('Adjacent key mismatch')
    full=len(events)//100
    raw=np.zeros((full,100,3),dtype=np.int64)
    for w in range(full):
        for j in range(100):
            i=w*100+j
            code,ht,dd,unknown,terminal,keys,next_hold=events[i]
            incoming=0 if i==0 else events[i-1][2]
            if incoming is None:raise ValueError('Terminal event inside sequence')
            raw[w,j]=[code,ht,incoming]
    base=raw.astype(np.float64)
    base[:,:,0]/=255
    base[:,:,1:]=np.clip(base[:,:,1:]/1000,0,30)
    base[:,0,2]=0
    return raw,base,{'events':len(events),'full_windows':full,'tail_events':len(events)%100,
                     'unparseable_key_prefixes':sum(e[3] for e in events),
                     'unknown_codes':sum(e[0]==0 for e in events),
                     'terminal_events':sum(e[4] for e in events),
                     'adjacencies_with_uncheckable_keys':unchecked_keys,
                     'negative_incoming_in_full_windows':int((raw[:,:,2]<0).sum())}
