"""Freeze the explicit five-channel Average candidate on fitting data only."""
import json
from pathlib import Path
import numpy as np
from type2branch_continuity import ROOT, sha
from type2branch_residual_features import residual_features


def main():
    out = ROOT / 'research/benchmarks/type2branch_fit_five_channels_v1'
    out.mkdir(exist_ok=False)
    base_dir = ROOT / 'research/benchmarks/type2branch_fit_base_v1'
    synth_dir = ROOT / 'research/benchmarks/type2branch_synthetic_fit_v1'
    paths = {base_dir / 'fit_base.npz': json.loads((base_dir/'report.json').read_text())['array_sha256'],
             synth_dir / 'synthetic_fit.npz': json.loads((synth_dir/'report.json').read_text())['output_sha256']}
    policy = {'observed': 'Author normalized base unchanged, timing clipped0..30s',
              'residual': 'Observed normalized timing minus valid synthetic_ms/1000; signed',
              'missing': 'Residual zero; separate validity audit mask, no sixth channel',
              'scope': '705 frozen fit sequences only; no selection/calibration/DEV/test reads',
              'limitations': ['Explicit adaptation; paper residual recipe unresolved',
                              'Average mode not verified as paper synthesis mode',
                              'No CLR runtime parity; no accuracy claim'],
              'input_sha256': {str(p.relative_to(ROOT)): h for p,h in paths.items()}}
    (out/'plan.json').write_text(json.dumps(policy, indent=2))
    for source in [Path(__file__), ROOT/'scripts/type2branch_residual_features.py']:
        (out/source.name).write_bytes(source.read_bytes())
    for path, expected in paths.items():
        if sha(path) != expected: raise ValueError('Input changed')
    with np.load(base_dir/'fit_base.npz', allow_pickle=False) as b, np.load(synth_dir/'synthetic_fit.npz', allow_pickle=False) as s:
        for key in ['subject', 'window']:
            np.testing.assert_array_equal(b[key], s[key])
        assert (b['true_length'] == 100).all()
        converted = [residual_features(base, synth) for base,synth in zip(b['base'], s['cleaned_ms'], strict=True)]
        features = np.stack([x[0] for x in converted])
        valid = np.stack([x[1] for x in converted])
        np.testing.assert_array_equal(features[:,:,:3], b['base'])
        np.testing.assert_array_equal(~valid, s['invalid_mask'])
        assert features.shape == (705,100,5) and np.isfinite(features).all()
        np.savez_compressed(out/'fit_features.npz', features=features, residual_valid=valid,
                            subject=b['subject'], window=b['window'], true_length=b['true_length'])
    report = {'status': 'five_channel_fit_candidate_prepared', 'shape': list(features.shape),
              'missing_residual_HT_FT': (~valid).sum(axis=(0,1)).tolist(),
              'residual_min_HT_FT': features[:,:,3:].min(axis=(0,1)).tolist(),
              'residual_max_HT_FT': features[:,:,3:].max(axis=(0,1)).tolist(),
              'output_sha256': sha(out/'fit_features.npz'), 'limitations': policy['limitations']}
    (out/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
