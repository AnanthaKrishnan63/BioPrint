"""Audit inference scheduling only: frozen weights and thresholds never change."""
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.research-deps'), str(ROOT / 'code/bioprint')]
import joblib
from keystroke_benchmark import Verifiers
from data_keystrokes import checksum

ART = ROOT / 'research/benchmarks/results/keyrecs-v1'
OUT = ART / 'runtime-audit.json'


def run():
    if OUT.exists():
        raise SystemExit('Preserve completed audit')
    report = {'scope': 'runtime scheduling only; generated-vector latency chooses optimization; dev parity is a veto',
              'test_accessed': False, 'tracks': {}}
    rng = np.random.default_rng(20260920)
    for track in ['fixed', 'free']:
        path = ART / f'{track}-train_selected.joblib'
        old = joblib.load(path)
        new = joblib.load(path)
        assert old.kind == new.kind == 'extra_trees'
        original_hash = checksum(path)
        new.model.n_jobs = 1
        timing = {}
        for size in [1, 128]:
            x = old.transform.median[None] + rng.normal(0, .01, (size, len(old.transform.median)))
            times = {1: [], 2: []}
            for model in [old, new]:
                model.scores(x)
            for repeat in range(30):
                for model in ([old, new] if repeat % 2 else [new, old]):
                    start = perf_counter(); model.scores(x)
                    times[model.model.n_jobs].append((perf_counter() - start) * 1000)
            timing[str(size)] = {str(k): {'median_ms': float(np.median(v)), 'p95_ms': float(np.quantile(v, .95))}
                                 for k, v in times.items()}
        faster = timing['1']['1']['median_ms'] < timing['1']['2']['median_ms']
        result = {'artifact_sha256': original_hash, 'generated_latency': timing,
                  'single_thread_faster_for_single_request': faster}
        # Timing decision is complete before development parity checks.
        with np.load(ART / f'{track}-dev-features.npz', allow_pickle=False) as a:
            x = a['x']
        config = json.loads((ART / f'{track}-frozen.json').read_text())
        threshold = np.array(config['frozen']['train_selected']['thresholds'])
        prior = old.scores(x); serial = new.scores(x)
        with np.load(ART / f'{track}-train_selected-dev-scores.npz', allow_pickle=False) as a:
            stored = a['scores']
        old_flags, flags = prior >= threshold, serial >= threshold
        near = np.unravel_index(np.argmin(abs(stored - threshold)), stored.shape)
        result.update(samples=len(x), identity_scores=int(serial.size),
                      max_absolute_score_error_vs_parallel=float(np.max(abs(serial - prior))),
                      max_absolute_score_error_vs_frozen=float(np.max(abs(serial - stored))),
                      decision_flips_vs_parallel=int(np.sum(flags != old_flags)),
                      decision_flips_vs_frozen=int(np.sum(flags != (stored >= threshold))),
                      nearest_threshold_margin=float(abs(stored[near] - threshold[near[1]])),
                      nearest_threshold_decision_unchanged=bool(flags[near] == (stored >= threshold)[near]),
                      artifact_unchanged=checksum(path) == original_hash)
        result['safe_to_enable'] = faster and result['decision_flips_vs_parallel'] == 0 and result['decision_flips_vs_frozen'] == 0 and result['artifact_unchanged']
        report['tracks'][track] = result
    OUT.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    run()
