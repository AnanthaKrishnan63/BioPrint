"""Candidate variable-length adaptation; old fixed-window artifacts stay frozen.

Supports actual1..100-event windows with explicit lengths. Residual convention
remains an adaptation, including the single-row merger case in author source.
No eligibility policy or calibrated short-input performance is implied here.
"""
import numpy as np
from type2branch_keyrecs_adapter import parse_event
from type2branch_csv_bridge import pad_features
from type2branch_random import fill_average_fallback
from type2branch_synthesis_cleanup import cleanup
from type2branch_residual_features import residual_features


def adapt_variable(rows, *, max_windows, session_complete=True):
    if type(max_windows) is not int or not 1<=max_windows<=20:
        raise ValueError('Expected1..20 maximum windows')
    if type(session_complete) is not bool:raise ValueError('Explicit session completeness required')
    if not rows:raise ValueError('No observed events')
    stop=min(len(rows),max_windows*100)
    actual_end=session_complete and stop==len(rows)
    events=[parse_event(row,last=actual_end and i==stop-1,details=True)
            for i,row in enumerate(rows[:stop])]
    unchecked=0
    for previous,current in zip(events,events[1:]):
        if previous[6]!=current[1]:raise ValueError('Adjacent inferred hold mismatch')
        if previous[5][1] is None or current[5][0] is None:unchecked+=1
        elif previous[5][1]!=current[5][0]:raise ValueError('Adjacent key mismatch')
    raw=np.zeros((len(events),3),dtype=np.int64)
    for i,event in enumerate(events):
        incoming=0 if i==0 else events[i-1][2]
        if incoming is None:raise ValueError('Terminal event inside sequence')
        raw[i]=[event[0],event[1],incoming]
    windows=[]
    for index,start in enumerate(range(0,len(raw),100)):
        wire=raw[start:start+100].copy()
        base=wire.astype(np.float64)
        base[:,0]/=255.
        base[:,1:]=np.clip(base[:,1:]/1000.,0,30)
        base[0,2]=0
        windows.append({'index':index,'true_length':len(wire),'raw_ms':wire,'base':base})
    audit={'parsed_events':len(events),'windows':len(windows),
           'true_lengths':[w['true_length'] for w in windows],
           'actual_session_end':actual_end,'unparsed_rows':len(rows)-stop,
           'unknown_codes':sum(e[0]==0 for e in events),
           'unparseable_key_prefixes':sum(e[3] for e in events),
           'terminal_events':sum(e[4] for e in events),
           'adjacencies_with_uncheckable_keys':unchecked}
    return windows,audit


def synthesize_window(window, population, rng):
    wire=np.asarray(window['raw_ms']);base=np.asarray(window['base'])
    length=window['true_length']
    if type(length) is not int or not 1<=length<=100 or wire.shape!=(length,3) or base.shape!=wire.shape:
        raise ValueError('Explicit real event count required')
    predicted,orders=population.predict(wire[:,0])
    if predicted.shape!=(length,2) or orders.shape!=(length,2):raise ValueError('Population output shape mismatch')
    if not np.array_equal(np.isnan(predicted),orders<0):raise ValueError('Missing-context mask mismatch')
    generated=np.column_stack((wire[:,0],fill_average_fallback(predicted,rng)))
    cleaned,partitions=cleanup(generated)
    if len(partitions):raise ValueError('Unexpected Average output partition')
    features,valid=residual_features(base,cleaned)
    return {'features':pad_features(features,length=100),'true_length':length,
            'residual_valid':np.pad(valid,((0,100-length),(0,0)),constant_values=False),
            'context_orders':np.pad(orders,((0,100-length),(0,0)),constant_values=-2),
            'synthetic_real_ms':cleaned,'raw_real_ms':wire.copy()}
