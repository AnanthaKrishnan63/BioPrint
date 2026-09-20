"""Author mean Euclidean score with explicit chronological small-cohort protocol."""
import numpy as np


def gallery_scores(query, gallery):
    """Negative mean Euclidean distance: higher scores mean closer matches.

    Direct subtraction avoids cancellation in squared-norm matrix identities.
    Chunk by query to bound memory on CPU; no embedding renormalization.
    """
    query=np.asarray(query,dtype=np.float64)
    gallery=np.asarray(gallery,dtype=np.float64)
    if query.ndim!=2 or gallery.ndim!=3 or query.shape[1]!=gallery.shape[2]:
        raise ValueError('Expected Q x D queries and U x G x D gallery')
    if not len(query) or not len(gallery) or not gallery.shape[1] or not query.shape[1]:
        raise ValueError('Empty scoring axis')
    if not np.isfinite(query).all() or not np.isfinite(gallery).all():
        raise ValueError('Nonfinite embeddings')
    result=np.empty((len(query),len(gallery)))
    for start in range(0,len(query),32):
        difference=query[start:start+32,None,None,:]-gallery[None,:,:,:]
        distances=np.linalg.norm(difference,axis=-1)
        result[start:start+32]=-distances.mean(axis=2)
    if not np.isfinite(result).all():raise ValueError('Distance overflow')
    return result


def chronological_scores(embeddings, subjects, windows):
    """Exactly15 windows/person; first5 gallery, last10 probes, all claims."""
    embeddings=np.asarray(embeddings)
    subjects=np.asarray(subjects)
    windows=np.asarray(windows)
    if embeddings.ndim!=2 or subjects.shape!=(len(embeddings),) or windows.shape!=subjects.shape:
        raise ValueError('Embedding metadata shape mismatch')
    identities=np.unique(subjects)
    if len(identities)<2:raise ValueError('At least two identities required')
    galleries=[];queries=[];labels=[]
    for index,identity in enumerate(identities):
        rows=np.flatnonzero(subjects==identity)
        rows=rows[np.argsort(windows[rows])]
        if not np.array_equal(windows[rows],np.arange(15)):
            raise ValueError('Exactly unique windows0..14 required per identity')
        galleries.append(embeddings[rows[:5]])
        queries.append(embeddings[rows[5:]])
        labels.extend([index]*10)
    scores=gallery_scores(np.concatenate(queries),np.stack(galleries))
    labels=np.asarray(labels)
    genuine=scores[np.arange(len(scores)),labels]
    mask=np.arange(len(identities))[None,:]!=labels[:,None]
    return {'scores':scores,'labels':labels,'subjects':identities,
            'genuine':genuine,'impostor':scores[mask]}


def global_threshold(impostor, target=.01):
    """Empirical calibration FAR <= target under inclusive score>=threshold."""
    values=np.asarray(impostor,dtype=np.float64)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Nonempty finite impostor scores required')
    if not np.isfinite(target) or not 0<=target<1:raise ValueError('Target must be in[0,1)')
    allowed=int(np.floor(target*len(values)))
    return float(np.nextafter(np.sort(values)[-allowed-1],np.inf))


def accept_scores(scores, threshold):
    """Keep nextafter thresholds intact even when callers supply float32 scores."""
    scores=np.asarray(scores,dtype=np.float64)
    threshold=np.asarray(threshold,dtype=np.float64)
    if threshold.ndim!=0 or np.isnan(threshold) or not np.isfinite(scores).all():
        raise ValueError('Finite scores and scalar non-NaN threshold required')
    return scores>=threshold
