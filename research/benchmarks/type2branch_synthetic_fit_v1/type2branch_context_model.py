"""Source-translated mean finite-context model; no C# runtime parity claim.

GPLv3 source references are retained under type2branch_synthesis. Random
fallback is deliberately unresolved: predict returns NaN and a missing mask.
"""
import math
import numpy as np
from type2branch_synthesis_cleanup import INVALID_TIMING


def context_hashes(keys, partitions=()):
    keys=list(keys)
    if any(not isinstance(k,(int,np.integer)) or not 0<=int(k)<=255 for k in keys):
        raise ValueError('Expected byte key codes')
    offsets=list(partitions)
    if any(not isinstance(p,(int,np.integer)) for p in offsets) or offsets!=sorted(set(offsets)):
        raise ValueError('Partition offsets must be ordered unique integers')
    boundaries=set(int(p) for p in offsets)
    if any(p<0 or p>=len(keys) for p in boundaries):raise ValueError('Invalid partition')
    history=[255]
    for i,key in enumerate(keys):
        if i in boundaries:history=[255]
        hashes=[]
        for order in range(min(7,len(history))+1):
            suffix=history[-order:] if order else []
            context=0
            for byte in suffix:context=(context<<8)|byte
            hashes.append((order,(int(key)<<56)|context))
        yield hashes
        history=(history+[int(key)])[-7:]


class MeanContextModel:
    def __init__(self):
        # (feature0HT/1FT,contextorder,hash) -> count,mean,mean_square
        self.models={}

    def feed(self, rows, partitions):
        rows=np.asarray(rows)
        if rows.ndim!=2 or rows.shape[1]!=3 or not np.issubdtype(rows.dtype,np.integer):
            raise ValueError('Integer N x 3 cleaned rows required')
        for i,hashes in enumerate(context_hashes(rows[:,0],partitions)):
            for feature in range(2):
                value=int(rows[i,feature+1])
                if value==INVALID_TIMING:continue
                if not 0<=value<=1500:raise ValueError('Input must pass reference cleanup')
                for order,key in hashes:
                    identity=(feature,order,key)
                    count,mean,square=self.models.get(identity,(0,0.,0.))
                    mean=(mean*count+value)/(count+1)
                    square=(square*count+value*value)/(count+1)
                    if count and square-mean*mean<0:
                        raise ValueError('Negative source variance; do not silently stabilize')
                    self.models[identity]=(count+1,mean,square)

    def predict(self, keys, partitions=()):
        values=np.full((len(keys),2),np.nan)
        orders=np.full((len(keys),2),-1,dtype=np.int16)
        for i,hashes in enumerate(context_hashes(keys,partitions)):
            for feature in range(2):
                for order,key in reversed(hashes):
                    if key==0:continue # MemoryStorage.GetBulk behavior
                    found=self.models.get((feature,order,key))
                    if found is not None and found[0]>=10:
                        values[i,feature]=math.trunc(found[1])
                        orders[i,feature]=order
                        break
        return values,orders

    def save(self,path):
        identities=sorted(self.models)
        np.savez_compressed(path,
            feature=np.array([k[0] for k in identities],dtype=np.uint8),
            order=np.array([k[1] for k in identities],dtype=np.uint8),
            hashes=np.array([k[2] for k in identities],dtype=np.uint64),
            count=np.array([self.models[k][0] for k in identities],dtype=np.int64),
            mean=np.array([self.models[k][1] for k in identities]),
            mean_square=np.array([self.models[k][2] for k in identities]))
