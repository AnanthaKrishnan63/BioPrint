"""TRAIN selection objective for the preregistered paired transfer."""
import numpy as np
from eval.strict_cmu import eer, rates, threshold_for


def selection_key(labels, scores, c):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if (labels.ndim != 1 or scores.shape != labels.shape or
            set(labels.tolist()) != {0, 1} or not np.isfinite(scores).all()):
        raise ValueError('Finite aligned genuine and impostor scores required')
    if c not in (0.01, 0.1, 1.0):
        raise ValueError('Unregistered regularization candidate')
    genuine, impostor = scores[labels == 1], scores[labels == 0]
    threshold = threshold_for(genuine, impostor, .01)
    result = rates(genuine, impostor, threshold)
    return result['frr'], eer(genuine, impostor), float(c)
