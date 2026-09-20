"""Pinned TRAIN-only tap diagnostic; no raw observations or heldout loaders."""
import hashlib
import json
from pathlib import Path
import numpy as np

from hmog_tap_benchmark import build_profiles, score_windows, validate_taps
from hmog_fit_tap_extract import validate_metadata
from hmog_verification import pooled_eer, far_threshold, verification_rates

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'research/benchmarks/hmog/tap_train_protocol_v1.json'
OUT = ROOT / 'research/benchmarks/hmog/tap_train_v1'
META = ('subject', 'session', 'role', 'activity_id', 'window_index')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split(meta):
    validate_metadata(meta)
    subjects = meta['subject'].astype(str)
    sessions = meta['session']
    enrollment = meta['role'] == 'train_enrollment'
    accounts = sorted(set(subjects))
    if len(accounts) != 4:
        raise ValueError('All four original accounts required')
    gallery = np.zeros(len(subjects), dtype=bool)
    for account in accounts:
        eligible = np.unique(sessions[enrollment & (subjects == account)])
        if len(eligible) < 2:
            raise ValueError('Each account needs separate enrollment sessions')
        gallery |= (subjects == account) & (sessions == eligible.min()) & enrollment
    return accounts, gallery, enrollment & ~gallery, meta['role'] == 'train_fit'


def load():
    protocol = json.loads(PROTOCOL.read_text())
    for relative, expected in protocol['source_files'].items():
        if digest(ROOT / relative) != expected:
            raise ValueError('Frozen TRAIN input changed: ' + relative)
    path = ROOT / 'datasets/hmog/fit_tap_features_v1.npz'
    with np.load(path, allow_pickle=False) as a:
        meta = {k: a['window_' + k].copy() for k in META}
        validate_metadata(meta)
        taps, index = validate_taps(a['tap_vectors'], a['tap_window'], len(meta['subject']))
        counts = np.bincount(index, minlength=len(meta['subject']))
        if not np.array_equal(counts, a['window_tap_count']) or not np.array_equal(counts >= 5, a['window_scorable']):
            raise ValueError('Tap counts/missingness disagree')
    return protocol, meta, taps, index


def claims(scores, available, subjects, accounts, rows):
    genuine = np.asarray(subjects).astype(str)[:, None] == np.asarray(accounts)[None, :]
    mask = available & rows[:, None]
    return scores[mask & genuine], scores[mask & ~genuine]


def report_scores(scores, available, subjects, accounts, calibration, diagnostic, targets):
    cg, ci = claims(scores, available, subjects, accounts, calibration)
    dg, di = claims(scores, available, subjects, accounts, diagnostic)
    if not all(len(a) for a in [cg, ci, dg, di]):
        raise ValueError('Both score classes required in calibration and TRAIN diagnostic')
    thresholds = {str(t): far_threshold(ci, t) for t in targets}
    result = {'thresholds': thresholds, 'calibration': {}, 'train_diagnostic': {}}
    for name, rows, g, i in [('calibration', calibration, cg, ci), ('train_diagnostic', diagnostic, dg, di)]:
        result[name] = {'eer': pooled_eer(g, i), 'rates': {
            t: verification_rates(g, i, threshold, expected_genuine=int(rows.sum()),
                                  expected_impostor=int(rows.sum()) * (len(accounts) - 1))
            for t, threshold in thresholds.items()}}
    return result


def main():
    # Exclusive plan and exact dependency pins precede reading feature values.
    OUT.mkdir(exist_ok=False)
    sources = ['hmog_tap_train.py', 'hmog_tap_benchmark.py', 'hmog_tap_reference.py',
               'hmog_fit_tap_extract.py', 'hmog_verification.py']
    plan = {'protocol_sha256': digest(PROTOCOL),
            'source_sha256': {s: digest(ROOT / 'scripts' / s) for s in sources},
            'scope': 'TRAIN diagnostic only; no DEV/test access or validation claim'}
    with (OUT / 'plan.json').open('x') as f:
        json.dump(plan, f, indent=2)
    protocol, meta, taps, index = load()
    accounts, gallery, calibration, diagnostic = split(meta)
    profiles, profile_counts = build_profiles(taps, index, meta['subject'], gallery, accounts)
    if set(profiles) != set(accounts):
        with (OUT / 'complete.json').open('x') as f:
            json.dump({'status': 'infeasible', 'plan': plan, 'profile_counts': profile_counts}, f, indent=2)
        return
    scored = score_windows(taps, index, len(meta['subject']), profiles, accounts)
    report = report_scores(scored['distances'], scored['available'], meta['subject'], accounts,
                           calibration, diagnostic, protocol['targets'])
    with (OUT / 'scores.npz').open('xb') as f:
        np.savez_compressed(f, **scored, gallery=gallery, calibration=calibration,
                            diagnostic=diagnostic, accounts=np.asarray(accounts), **meta)
    with (OUT / 'profiles.json').open('x') as f:
        json.dump(profiles, f, indent=2, allow_nan=False)
    with (OUT / 'complete.json').open('x') as f:
        json.dump({'status': 'complete_train_diagnostic', 'plan': plan, 'profile_counts': profile_counts,
                   'gallery_windows': int(gallery.sum()), 'calibration_windows': int(calibration.sum()),
                   'diagnostic_windows': int(diagnostic.sum()), 'tap11': report,
                   'scores_sha256': digest(OUT / 'scores.npz')}, f, indent=2, allow_nan=False)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
