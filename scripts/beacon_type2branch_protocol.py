"""Freeze the paired representation-transfer protocol without reading observations."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/benchmarks/beacon_type2branch_protocol_v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    previous = ROOT / 'research/benchmarks/beacon-neural-v2/frozen.json'
    old = json.loads(previous.read_text())
    pair = ROOT / 'research/benchmarks/type2branch_length_pair_v1/report.json'
    selected = json.loads(pair.read_text())
    if selected['status'] != 'paired_train_comparison_complete':
        raise ValueError('Source experiment must be complete')
    checkpoint = Path(selected['selected_checkpoint']['checkpoint'])
    shards = sorted(checkpoint.parent.glob(checkpoint.name + '.*'))
    if {p.suffix for p in shards} != {'.index', '.data-00000-of-00001'}:
        raise ValueError('Expected exactly two checkpoint shards')
    paths = [Path(__file__), previous, pair,
        ROOT / 'datasets/beacon/split_manifest.json',
        ROOT / 'datasets/beacon/record_roles.json',
        ROOT / 'research/benchmarks/beacon/frozen.json',
        ROOT / 'research/benchmarks/type2branch_context_fit_v1/population.npz',
        ROOT / 'research/benchmarks/pointer_sapimouse/encoder.torchscript.pt',
        ROOT / 'scripts/beacon_dual_neural_benchmark.py',
        ROOT / 'scripts/beacon_neural_benchmark.py',
        ROOT / 'scripts/beacon_benchmark.py',
        ROOT / 'scripts/type2branch_variable_adapter.py',
        ROOT / 'scripts/type2branch_random.py',
        ROOT / 'code/bioprint/eval/strict_cmu.py', *shards]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    if (old['manifest_sha256'] != hashes['datasets/beacon/split_manifest.json'] or
            old['roles_sha256'] != hashes['datasets/beacon/record_roles.json']):
        raise ValueError('Existing split provenance changed')
    protocol = {
        'experiment': 'Frozen Type2Branch transfer with real paired BEACON pointer behavior',
        'status': 'preregistered_before_new_train_embeddings',
        'identity_groups': old['identity_groups'],
        'pairing': 'Same BEACON person, session and exact prior dual-neural 30-second window',
        'window_rule': 'Reuse prior loader and pointer eligibility unchanged; assert TRAIN starts match frozen train_audit. Same windows and claims for every model.',
        'keyboard': 'Exact integer-ms elapsed press/release; stable onset sort; complete holds press>=start and release<start+30; first min(100,N) events; at least5; incoming DD and HT; first DD0; existing normalized residual synthesis and post-synthesis padding.',
        'short_windows': 'Retain lengths5..24; report counts and length distribution. These are below source mixed-training minimum25; no masking or length-based exclusions.',
        'quality_gate': 'Reject negative holds, nonfinite/noninteger timestamps, unknown keys>1%, median duration disagreement>10ms; fail whole experiment on unexpected identity loss.',
        'rng': 'Fresh first_synthesis_thread per cohort; persistent stream across sorted subject, enrollment then probe, ascending window start; real events only; HT before FT.',
        'encoders': 'Frozen source checkpoint and SapiMouse; no BEACON encoder updates or population fitting. TypeNet retained only as matched reference.',
        'gallery': 'First up to5 eligible enrollment windows; mean Euclidean distance to each gallery embedding; no embedding normalization.',
        'columns': 'Prior34 columns unchanged; append Type2Branch distance at34.',
        'models': {'typenet': [0], 'sapimouse': [33], 'handcrafted_behavior': list(range(1,33)),
            'old_dual_neural': [0,33], 'old_hybrid': list(range(34)),
            'type2branch': [34], 'type2branch_pointer': [34,33],
            'type2branch_hybrid': list(range(1,34))+[34]},
        'fit': 'StandardScaler and balanced LogisticRegression max_iter1000 random_state20260920; fit identities only; no selection/calibration refit.',
        'candidate_C': [0.01,0.1,1.0],
        'selection': 'Minimize selection-identity FRR at empirical FAR<=1%; ties lower discrete EER then lower C. Lower score=-decision_function means genuine.',
        'calibration': 'Reserved TRAIN calibration identities; strict_cmu.threshold_for at1%,5% and EER; accept score<=threshold. Do not reuse KeyRecs thresholds.',
        'dev': 'Only after model/threshold/source freeze; all eligible DEV identities with their ledger-approved personal enrollment; no global fitting on DEV support. Report FAR/FRR/EER, integer counts, exclusions and per-probe-identity metrics for every model.',
        'api': 'Actual frozen DEV score replay through isolated research API; no network listener; verify score and decision parity.',
        'sealed_test_accessed': False,
        'prior_exposure': 'Earlier BEACON and KeyRecs DEV results already observed; this is not fresh validation. New fitting decisions use TRAIN only.',
        'limitations': ['Small independent identity count', 'Gameplay rather than login',
            'Source transfer and residual adaptation, not exact published SOTA reproduction',
            'Two behavioral modalities; does not fulfill all-feature paired validation'],
        'source_sha256': hashes,
    }
    OUT.mkdir(exist_ok=False)
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (OUT / 'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'status': protocol['status'], 'sha256': digest(OUT/'protocol.json')}))


if __name__ == '__main__':
    main()
