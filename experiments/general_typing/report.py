"""Render immutable experiment results without fitting or changing thresholds."""
import json
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / 'results'


def main():
    report = json.loads((OUT / 'report.json').read_text())
    control = json.loads((OUT / 'control.json').read_text())
    beacon = json.loads((OUT / 'beacon.json').read_text())
    audit = json.loads((OUT / 'audit.json').read_text())
    lines = ['# General typing matcher: measured results', '',
        'This experiment removes the CMU password-schema restriction. It does not establish '
        'universal password, language, keyboard, mobile, or attacker generalization. The four '
        'login experiment branches retain their existing scorers; this work is isolated in '
        '`experiment/general-typing`.', '',
        '## Method', '',
        'A shared binary matcher compares an attempt with ten enrollment recordings. '
        'For each of hold, down-down and up-down timing, calculate five quantiles, mean and '
        'standard deviation. This gives 21 features for any supported password length. '
        'For feature j, c_j = median(enrollment_j), '
        's_j = max(mean absolute deviation, 10 ms, 0.1 |c_j|), '
        'z_j = clip((attempt_j - c_j)/s_j, -100, 100). '
        'The learned models receive [|z|, z], 42 features, without key labels or identity IDs. '
        'The distance comparator uses -mean(min(|z|, 6)). Larger scores indicate a closer match.', '',
        'Fit, threshold-calibration and evaluation identities are disjoint. KeyRecs identities '
        'retain the same role across fixed/free tracks. Each evaluation identity contributes '
        'only its first ten valid TRAIN records to personal enrollment; its DEV records never '
        'fit the population model or threshold. All eligible observations are used, with '
        'invalid vectors and free-window tails counted below. Sealed test partitions remain closed.', '',
        'Logistic regression uses C=1; ExtraTrees uses 128 trees and minimum leaf size 10. '
        'Fit weights balance track and genuine/impostor class. All nine configurations were '
        'frozen before DEV reads; there was no candidate search or DEV-based retuning. '
        'The threshold is just above the relevant calibration impostor order statistic, '
        'so at most floor(0.01 N_impostor) calibration impostors pass. This is an empirical '
        'calibration constraint, not a guarantee on DEV FAR.', '',
        'FAR = false accepts / impostor comparisons; FRR = false rejects / genuine comparisons. '
        'EER is the mean of FAR and FRR at the empirical ROC point with the smallest difference '
        '(diagnostic only). Tables use pooled comparisons, not the previous reports’ macro-user '
        'EER, so compare only rows here. All-pairs comparisons share people and probes; they '
        'are not independent trials.', '',
        '## Pooled training: unseen evaluation users', '',
        '| Track | Method | FAR | FRR | EER | False accepts / impostors | False rejects / genuine |',
        '|---|---|---:|---:|---:|---:|---:|']
    def row(track, name, r):
        return f'| {track} | {name} | {100*r["far"]:.2f}% | {100*r["frr"]:.2f}% | {100*r["eer"]:.2f}% | {r["false_accepts"]}/{r["impostor_attempts"]} | {r["false_rejects"]}/{r["genuine_attempts"]} |'
    for track in ('cmu', 'fixed', 'free'):
        if track in control:
            lines.append(row(track, 'Existing positional scorer, enrollment threshold', control[track]))
        for name in ('distance', 'logistic', 'extra_trees'):
            lines.append(row(track, name, report['results'][f'{track}/pooled/{name}']))
    lines += ['', 'The existing scorer comparison uses identical eligible people and attempts. '
              'Its enrollment-derived threshold is its shipped operating point; the general '
              'models instead use a global calibration-FAR target. EER helps separate score '
              'discrimination from operating-point differences. Free text has no fixed positional '
              'password template, so an existing-login baseline there would be misleading.', '',
              '## Cross-dataset transfer', '',
              'These models and thresholds see only the named source dataset. Evaluation users '
              'in the target are unseen; their personal enrollment is still required. CMU and '
              'KeyRecs fixed use different prompted texts, but this is only a small number of '
              'passwords and cannot establish accuracy for arbitrary secrets.', '',
              '| Target | Training / method | FAR | FRR | EER | False accepts / impostors | False rejects / genuine |',
              '|---|---|---:|---:|---:|---:|---:|']
    for key, r in report['results'].items():
        if r['dataset_transfer']:
            track, protocol, name = key.split('/')
            lines.append(row(track, f'{protocol} / {name}', r))
    lines += ['', '## Single-source within-dataset controls', '',
              '| Target | Training / method | FAR | FRR | EER | False accepts / impostors | False rejects / genuine |',
              '|---|---|---:|---:|---:|---:|---:|']
    for key, r in report['results'].items():
        track, protocol, name = key.split('/')
        if not r['dataset_transfer'] and protocol != 'pooled':
            lines.append(row(track, f'{protocol} / {name}', r))
    lines += ['', '## BEACON: frozen gameplay transfer', '',
              f'All {len(beacon["subjects"])} existing DEV identities were eligible. '
              'First ten disjoint ten-event windows from each TRAIN-enrollment recording form '
              'the personal profile. Every valid probe window is compared against every profile. '
              'No BEACON data fits the population model or threshold. See BEACON_PROTOCOL.md '
              'and results/beacon.json for source guards and exclusion counts.', '',
              '| Target | Training / method | FAR | FRR | EER | False accepts / impostors | False rejects / genuine |',
              '|---|---|---:|---:|---:|---:|---:|']
    for key, r in beacon['results'].items():
        lines.append(row('BEACON gameplay', key, r))
    lines += ['', '## Data and exclusions', '',
              '| Track | Fit identities | Calibration identities | Evaluation identities |',
              '|---|---:|---:|---:|']
    for track in ('cmu', 'fixed', 'free'):
        lines.append(f'| {track} | {len(report["fit_profiles"][track])} | {len(report["calibration_profiles"][track])} | {report["results"][f"{track}/pooled/distance"]["subject_count"]} |')
    lines += ['', '| Partition | Valid vectors | Excluded vectors | Unused tail digraphs |', '|---|---:|---:|---:|']
    for key, a in report['audit'].items():
        lines.append(f'| {key} | {a["valid_vectors"]} | {a["rejected_vectors"]} | {a["discarded_tail_digraphs"]} |')
    lines += ['', f'Total fitting and validation runtime: **{report["seconds"]:.1f} seconds**.', '',
              '## Verification and inference cost', '',
              f'Independent audit checked {audit["verified_score_tables"]} saved score tables, '
              f'{audit["model_hashes_verified"]} artifact hashes, exact operating counts, '
              'source hashes, identity separation and calibration FAR constraints. Fifteen unit '
              'tests cover arbitrary key sequences, variable lengths, live/dataset timing '
              'parity, invalid timings and browser-format input. Generated input latency '
              'includes feature extraction and enrollment-profile construction, excludes model '
              'loading, and is indicative on a machine also running research jobs.', '',
              '| Pooled method | Median inference (ms) | Maximum of 10 calls (ms) | Artifact size (MiB) |',
              '|---|---:|---:|---:|']
    for name, r in audit['generated_browser_input_inference'].items():
        lines.append(f'| {name} | {r["median_ms"]:.2f} | {r["max_ms"]:.2f} | {r["model_bytes"]/1024**2:.3f} |')
    lines += ['',
              '## Limits and integration decision', '',
              'This is offline typing verification, not all-feature app FAR/FRR/EER. '
              'Free-text windows contain nine source digraphs and are a short-input stress test; '
              'different texts can contribute to apparent identity separation. Some KeyRecs '
              'down-down timings are negative because source event ordering differs from browser '
              'press ordering. Distribution pooling loses exact key-position information. '
              'Prior DEV exposure remains disclosed. No automatic profile updates are enabled.', '',
              'No candidate is automatically promoted by this report. A login integration needs '
              'a predeclared calibration gate, browser timing parity, latency checks and tests '
              'of people entering the same arbitrary password. A new password can be represented '
              'by this model; that does not mean its error rates have been measured.', '',
              '## Reproduction and artifacts', '',
              'From the original repository root, using the bigidea environment:', '',
              '```bash',
              'BIOPRINT_RESEARCH_ROOT="$PWD" bash scripts/research.sh .worktrees/general-typing/experiments/general_typing/run.py',
              'BIOPRINT_RESEARCH_ROOT="$PWD" bash scripts/research.sh .worktrees/general-typing/experiments/general_typing/control.py',
              'BIOPRINT_RESEARCH_ROOT="$PWD" bash scripts/research.sh .worktrees/general-typing/experiments/general_typing/beacon.py',
              'bash scripts/research.sh .worktrees/general-typing/experiments/general_typing/audit.py',
              'bash scripts/research.sh .worktrees/general-typing/experiments/general_typing/report.py',
              '```', '',
              '`run.py` refuses an existing results directory. Preserve the current run before '
              'reproducing. The guarded loaders and bigidea launcher are existing research '
              'workspace dependencies. `results/plan.json` pins input/source hashes and identity '
              'roles; `frozen.json` pins trained artifacts; `report.json` includes every evaluation '
              'and per-user statistics; local `*-scores.npz` files retain all score/label pairs. '
              'Large joblib models and score arrays remain local and are reproducible, not Git blobs.', '',
              '`infer.py --input recording.json --model pooled/logistic` accepts ten enrollment '
              'Sample objects and an attempt in browser format. It verifies model checksums and '
              'returns only a behavioral score; it does not authorize login or start a server.', '']
    (HERE / 'RESULTS.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
