"""Generate Average candidate timings from the frozen TRAIN population only."""
import json
from pathlib import Path
import time
import numpy as np
from type2branch_continuity import ROOT, sha
from type2branch_context_model import MeanContextModel
from type2branch_random import first_synthesis_thread, fill_average_fallback
from type2branch_synthesis_cleanup import cleanup, INVALID_TIMING

OUT = ROOT / 'research/benchmarks/type2branch_synthetic_fit_v1'


def main():
    OUT.mkdir(exist_ok=False)
    population_dir = ROOT / 'research/benchmarks/type2branch_context_fit_v1'
    input_dir = ROOT / 'research/benchmarks/type2branch_synthesis_fit_v1'
    population_report = json.loads((population_dir / 'report.json').read_text())
    input_report = json.loads((input_dir / 'report.json').read_text())
    paths = {population_dir / 'population.npz': population_report['population_sha256'],
             population_dir / 'fit_context_orders.npz': population_report['context_orders_sha256'],
             input_dir / 'cleaned_fit.npz': input_report['output_sha256']}
    for path, expected in paths.items():
        if sha(path) != expected:
            raise ValueError(f'Changed frozen artifact: {path}')
    sources = [Path(__file__), ROOT / 'scripts/type2branch_context_model.py',
               ROOT / 'scripts/type2branch_random.py', ROOT / 'scripts/type2branch_synthesis_cleanup.py']
    plan = {'scope': 'Frozen 47 fitting identities and 705 fitting sequences only',
            'method': 'Explicit Average candidate; paper synthesis mode unverified',
            'sequence_order': 'Stored subject-sorted chronological first15 windows',
            'random': 'Fresh first-thread global1234->local; shared stream; HT then FT',
            'inputs_sha256': {str(p.relative_to(ROOT)): h for p, h in paths.items()},
            'sources_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sources},
            'residual_policy': 'Not applied; raw and cleaned output plus sentinel mask retained'}
    (OUT / 'plan.json').write_text(json.dumps(plan, indent=2))
    for source in sources:
        (OUT / source.name).write_bytes(source.read_bytes())
    model = MeanContextModel()
    with np.load(population_dir / 'population.npz', allow_pickle=False) as data:
        for f, order, key, count, mean, square in zip(
                data['feature'], data['order'], data['hashes'], data['count'],
                data['mean'], data['mean_square'], strict=True):
            identity = (int(f), int(order), int(key))
            assert identity not in model.models
            assert f in (0, 1) and 0 <= order <= 7 and count > 0
            assert np.isfinite(mean) and np.isfinite(square) and 0 <= mean <= 1500
            model.models[identity] = (int(count), float(mean), float(square))
    assert len(model.models) == population_report['models']
    with np.load(input_dir / 'cleaned_fit.npz', allow_pickle=False) as data:
        rows, subjects, windows = data['rows'], data['subject'], data['window']
    assert rows.shape == (705, 100, 3)
    ids, counts = np.unique(subjects, return_counts=True)
    assert len(ids) == 47 and (counts == 15).all()
    with np.load(population_dir / 'fit_context_orders.npz', allow_pickle=False) as data:
        expected_orders = data['orders']
    raw, cleaned, masks, all_orders = [], [], [], []
    rng = first_synthesis_thread()
    start = time.perf_counter()
    for sample, expected in zip(rows, expected_orders, strict=True):
        predicted, orders = model.predict(sample[:, 0])
        np.testing.assert_array_equal(orders, expected)
        np.testing.assert_array_equal(np.isnan(predicted), orders < 0)
        timings = fill_average_fallback(predicted, rng)
        generated = np.column_stack((sample[:, 0], timings))
        output, partitions = cleanup(generated)
        assert len(partitions) == 0  # Means and fallback are bounded to <=1500ms.
        np.testing.assert_array_equal(output[:, 0], sample[:, 0])
        raw.append(generated)
        cleaned.append(output)
        masks.append(output[:, 1:] == INVALID_TIMING)
        all_orders.append(orders)
    raw, cleaned, masks, all_orders = map(np.stack, (raw, cleaned, masks, all_orders))
    np.savez_compressed(OUT / 'synthetic_fit.npz', raw_ms=raw, cleaned_ms=cleaned,
                        invalid_mask=masks, context_orders=all_orders,
                        subject=subjects, window=windows)
    report = {'status': 'fitting_synthetic_timings_complete_residual_conversion_pending',
              'sequences': len(raw), 'identities': len(ids),
              'fallback_draws_HT_FT': [(all_orders[:, :, f] < 0).sum().item() for f in range(2)],
              'invalid_output_HT_FT': masks.sum(axis=(0, 1)).tolist(),
              'raw_min_HT_FT': raw[:, :, 1:].min(axis=(0, 1)).tolist(),
              'raw_max_HT_FT': raw[:, :, 1:].max(axis=(0, 1)).tolist(),
              'seconds': time.perf_counter() - start,
              'output_sha256': sha(OUT / 'synthetic_fit.npz'),
              'limitations': ['Source translation, not CLR execution parity',
                              'Average candidate not verified paper mode',
                              'No residual channels, encoder training, or recognition metrics',
                              'No selection/calibration/DEV/test observations read']}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
