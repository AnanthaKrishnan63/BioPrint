"""Opt-in learned pointer sequence verification; no change to login decisions.

The encoder is the exported SapiMouse FCN. Each block needs 129 pointer events
for 128 displacements, including movement and button events in recorded order.
A profile must be enrolled from the actual account holder's own recordings and
kept separate by device class. Call verify separately for each recording; never
aggregate probe embeddings across session boundaries.
Dataset-derived research thresholds are not automatically production thresholds.

NumPy is the only import-time dependency. Torch and scikit-learn are loaded only
when their respective research paths are invoked. See research/benchmarks/
pointer_sources.md for dataset/domain and calibration limitations.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

WINDOW = 128


def displacement_blocks(points: Iterable[Any]) -> np.ndarray:
    """Return author-normalized (blocks, 2, 128) arrays without cross-call joins.

Accept coordinate pairs, mappings containing x/y, or objects with x/y members.
An incomplete trailing block is omitted. All inputs must contain finite numbers.
"""
    xy = []
    for point in points:
        if isinstance(point, dict):
            xy.append([point['x'], point['y']])
        elif hasattr(point, 'x'):
            xy.append([point.x, point.y])
        else:
            xy.append([point[0], point[1]])
    array = np.asarray(xy, dtype=float).reshape(-1, 2)
    if not np.isfinite(array).all():
        raise ValueError('Pointer coordinates must be finite')
    deltas = np.abs(np.diff(array, axis=0))
    count = len(deltas) // WINDOW
    if not count:
        return np.empty((0, 2, WINDOW), dtype=np.float32)
    blocks = deltas[:count * WINDOW].reshape(count, WINDOW, 2)
    mean = blocks.mean(axis=(1, 2), keepdims=True)
    std = blocks.std(axis=(1, 2), keepdims=True)
    result = (blocks - mean) / np.where(std == 0, .0001, std)
    return result.transpose(0, 2, 1).astype(np.float32)


class PointerSequenceEncoder:
    """Load a locally exported TorchScript FCN once, then encode recordings."""
    def __init__(self, checkpoint: str | Path):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError('Learned pointer encoding requires the optional CPU torch research dependency') from exc
        self._torch = torch
        self.model = torch.jit.load(str(checkpoint), map_location='cpu').eval()

    def encode_blocks(self, blocks: np.ndarray) -> np.ndarray:
        array = np.asarray(blocks, dtype=np.float32)
        if array.ndim != 3 or array.shape[1:] != (2, WINDOW):
            raise ValueError('Expected pointer blocks with shape (n, 2, 128)')
        if not np.isfinite(array).all():
            raise ValueError('Pointer blocks must be finite')
        if not len(array):
            return np.empty((0, 128), dtype=np.float32)
        with self._torch.inference_mode():
            batches = [self.model(self._torch.from_numpy(array[i:i + 128])).numpy() for i in range(0, len(array), 128)]
        return np.concatenate(batches)

    def encode(self, points: Iterable[Any]) -> np.ndarray:
        return self.encode_blocks(displacement_blocks(points))


@dataclass
class PointerSequenceProfile:
    """Account-specific OCSVM over learned embeddings; requires explicit threshold."""
    model: Any
    enrollment_blocks: int
    nu: float
    normalize_by_enrollment: bool
    method: str = 'ocsvm'

    def score_blocks(self, embeddings: np.ndarray) -> np.ndarray:
        array = np.asarray(embeddings, dtype=float)
        if array.ndim != 2 or array.shape[1] != 128 or not np.isfinite(array).all():
            raise ValueError('Expected finite 128-dimensional embeddings')
        if not len(array):
            return np.empty(0)
        if self.method == 'cosine':
            normalized = array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)
            return normalized @ self.model['center']
        if self.method == 'latent_manhattan':
            from .scorer import _deviations
            return -_deviations(self.model['center'], self.model['spread'], array, 6).mean(axis=1)
        scores = self.model.score_samples(array)
        if self.normalize_by_enrollment:
            scores = scores / (self.nu * self.enrollment_blocks)
        return scores

    def verify(self, embeddings: np.ndarray, *, threshold: float, blocks_per_decision: int) -> list[dict]:
        if blocks_per_decision < 1 or not np.isfinite(threshold):
            raise ValueError('An explicit finite threshold and positive aggregation count are required')
        scores = self.score_blocks(embeddings)
        # Incomplete recordings yield no verdict; callers should request more data.
        result = []
        for start in range(0, len(scores) - blocks_per_decision + 1, blocks_per_decision):
            score = float(scores[start:start + blocks_per_decision].mean())
            result.append({'score': score, 'threshold': float(threshold), 'matched': score >= threshold})
        return result


def enroll_sequence_profile(embeddings: np.ndarray, *, nu: float = .5, normalize_by_enrollment: bool = True, method: str = 'ocsvm') -> PointerSequenceProfile:
    """Fit an explicitly selected method on account-holder enrollment only."""
    array = np.asarray(embeddings, dtype=float)
    if array.ndim != 2 or array.shape[1] != 128 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError('At least two finite 128-dimensional enrollment blocks are required')
    if method == 'cosine':
        normalized = array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)
        center = normalized.mean(axis=0)
        center /= max(np.linalg.norm(center), 1e-12)
        return PointerSequenceProfile({'center': center}, len(array), nu, False, method)
    if method == 'latent_manhattan':
        from .scorer import _fit_arrays
        center, spread = _fit_arrays(array, np.asarray([f'pointer.{i}' for i in range(128)]))
        return PointerSequenceProfile({'center': center, 'spread': spread}, len(array), nu, False, method)
    if method != 'ocsvm':
        raise ValueError('Unknown pointer sequence scoring method')
    try:
        from sklearn.svm import OneClassSVM
    except ImportError as exc:
        raise RuntimeError('Learned pointer enrollment requires the optional scikit-learn research dependency') from exc
    if not 0 < nu <= 1:
        raise ValueError('nu must be in (0, 1]')
    model = OneClassSVM(gamma='scale', nu=nu).fit(array)
    return PointerSequenceProfile(model, len(array), nu, normalize_by_enrollment)
