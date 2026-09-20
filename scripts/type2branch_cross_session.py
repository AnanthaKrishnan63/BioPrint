"""Explicit DEV extension: reserved S1 enrollment and S2 probes, no refitting."""
import numpy as np
from type2branch_scoring import gallery_scores


def cross_session_scores(gallery_embeddings, gallery_subjects, gallery_windows,
                         probe_embeddings, probe_subjects, probe_windows, allowed):
    identities=np.array(sorted(allowed))
    if len(identities)<2 or len(set(identities))!=len(identities):
        raise ValueError('Unique predefined identities required')
    assembled=[]
    for embeddings,subjects,windows,expected in [
            (gallery_embeddings,gallery_subjects,gallery_windows,np.arange(15,20)),
            (probe_embeddings,probe_subjects,probe_windows,np.arange(10))]:
        embeddings=np.asarray(embeddings)
        subjects=np.asarray(subjects);windows=np.asarray(windows)
        if embeddings.ndim!=2 or subjects.shape!=(len(embeddings),) or windows.shape!=subjects.shape:
            raise ValueError('Embedding metadata shape mismatch')
        if set(subjects)!=set(identities):raise ValueError('Identity cohort incomplete or expanded')
        grouped=[]
        for identity in identities:
            rows=np.flatnonzero(subjects==identity)
            rows=rows[np.argsort(windows[rows])]
            if not np.array_equal(windows[rows],expected):raise ValueError('Frozen window allocation changed')
            grouped.append(embeddings[rows])
        assembled.append(np.stack(grouped))
    gallery,queries=assembled
    scores=gallery_scores(queries.reshape(-1,queries.shape[-1]),gallery)
    labels=np.repeat(np.arange(len(identities)),10)
    mask=np.arange(len(identities))[None,:]!=labels[:,None]
    return {'scores':scores,'labels':labels,'subjects':identities,
            'genuine':scores[np.arange(len(scores)),labels],'impostor':scores[mask]}
