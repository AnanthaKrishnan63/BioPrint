"""Optional supervised RBF verification with explicit, aligned negative examples.

The live demo does not enable this automatically: it needs verified other-user
samples of exactly the same feature schema plus separate training calibration.
Runtime prediction is NumPy-only; fitting requires the research dependencies.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class Profile:
    names: list[str]
    scale: list[float]
    support: list[list[float]]
    dual: list[float]
    intercept: float
    gamma: float
    enrollment_samples: int
    background_samples: int
    threshold: float | None = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


def fit(positive, negative, names, *, c=10., gamma_factor=.1):
    """Fit only labeled training observations; no threshold inferred from dev."""
    from sklearn.svm import SVC
    positive, negative = np.asarray(positive, float), np.asarray(negative, float)
    if (positive.ndim != 2 or negative.ndim != 2 or positive.shape[1] != len(names)
            or negative.shape[1] != len(names) or len(positive) < 2 or len(negative) < 2):
        raise ValueError('Positive and negative training samples must match names and contain at least two rows')
    if not np.isfinite(positive).all() or not np.isfinite(negative).all():
        raise ValueError('Non-finite observations are not allowed')
    scale = np.maximum(negative.std(axis=0), 1.)
    estimator = SVC(C=c, gamma=gamma_factor / len(names), class_weight='balanced')
    estimator.fit(np.r_[positive, negative] / scale, np.r_[np.ones(len(positive)), np.zeros(len(negative))])
    return Profile(list(names), scale.tolist(), estimator.support_vectors_.tolist(),
                   estimator.dual_coef_[0].tolist(), float(estimator.intercept_[0]),
                   float(estimator._gamma), len(positive), len(negative))


def distances(profile, values):
    """Larger = less like the owner. This is a margin, not a probability."""
    x = np.asarray(values, float)
    if x.ndim != 2 or x.shape[1] != len(profile.names) or not np.isfinite(x).all():
        raise ValueError('Expected a finite matrix matching the enrolled feature schema')
    z = x / np.array(profile.scale)
    support = np.asarray(profile.support)
    square = np.maximum((z*z).sum(axis=1)[:, None] + (support*support).sum(axis=1)[None, :] - 2*z@support.T, 0)
    return -(np.exp(-profile.gamma*square) @ np.array(profile.dual) + profile.intercept)


def calibrate(profile, negative_calibration, target_far=.01):
    """Freeze a threshold from a separate TRAINING calibration partition.

    Acceptance is distance <= threshold. Ties are excluded conservatively.
    A target rate below one/count cannot be statistically substantiated here.
    """
    if not 0 <= target_far < 1:
        raise ValueError('target_far must be in [0,1)')
    scores = np.sort(distances(profile, negative_calibration))
    if len(scores) == 0:
        raise ValueError('Calibration examples required')
    allowed = int(np.floor(target_far * len(scores)))
    profile.threshold = float(np.nextafter(scores[allowed], -np.inf))
    return profile


def accepts(profile, values):
    if profile.threshold is None:
        raise ValueError('Calibrate on training data before making decisions')
    return distances(profile, values) <= profile.threshold
