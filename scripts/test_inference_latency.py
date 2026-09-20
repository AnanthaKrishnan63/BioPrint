"""Regression: timing an async route must execute inference, not create a coroutine."""
from inference_latency import measure


def test_async_inference_is_awaited():
    completed = []

    async def inference():
        completed.append(True)
        return {'accepted': [True]}

    result = measure(inference, 5)
    assert len(completed) == 8  # Three warm-ups and five measured executions.
    assert result['repeats'] == 5
    assert result['median_ms'] >= 0
