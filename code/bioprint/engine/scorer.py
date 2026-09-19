"""Anomaly scorer, generic over FeatureVectors. OWNER: Agent A (engine).

Used for keystroke vectors, pointer vectors, and by eval/cmu.py on CMU rows.

STUB: scaled Manhattan without a tuned threshold. Agent A replaces with the real
fit (leave-one-out threshold, robust spread) and keeps this interface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from contracts import Contribution, SignalResult


@dataclass
class Model:
    names: list[str]
    center: list[float]
    spread: list[float]
    threshold: float
    n: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Model":
        return cls(**d)


def fit(vectors: list[list[float]], names: list[str]) -> Model:
    X = np.asarray(vectors, dtype=float)
    center = X.mean(axis=0)
    spread = np.maximum(np.abs(X - center).mean(axis=0), 1e-6)
    return Model(names, center.tolist(), spread.tolist(), threshold=float(len(names)) * 2.0, n=len(X))


def distance(model: Model, values: list[float]) -> tuple[float, np.ndarray]:
    """Total distance and each feature's share of it."""
    per = np.abs(np.asarray(values) - model.center) / np.asarray(model.spread)
    return float(per.sum()), per


def score(model: Model, values: list[float], name: str = "keystroke") -> SignalResult:
    total, per = distance(model, values)
    top = np.argsort(per)[::-1][:5]
    return SignalResult(
        name=name,
        score=total,
        threshold=model.threshold,
        flagged=total > model.threshold,
        contributions=[
            Contribution(feature=model.names[i], value=values[i], expected=model.center[i], deviation=float(per[i]))
            for i in top
        ],
        reasons=["STUB scorer"],
    )
