"""Pointer features. OWNER: Agent C (pointer).

STUB: returns None (no pointer signal). Agent C implements the vector; the
server fits engine.scorer on it exactly as it does for keystrokes.
"""

from __future__ import annotations

from contracts import FeatureVector, Sample


def pointer_vector(sample: Sample) -> FeatureVector | None:
    """Fixed-length vector describing how the pointer reached and clicked Login.
    None when the attempt used no pointer (e.g. submitted with Enter)."""
    return None
