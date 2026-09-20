"""Freeze and prepare 15 base-channel sequences per fitting identity only."""
from collections import defaultdict
import contextlib
import io
import json
from pathlib import Path
import numpy as np

from type2branch_continuity import ROOT, DATA, sha
from type2branch_keyrecs_adapter import adapt
from type2branch_input_audit import reference_prepare

OUT=ROOT/'research/benchmarks/type2branch_fit_base_v1'


def main():
    OUT.mkdir(exist_ok=False)
    roles_path=ROOT/'research/benchmarks/type2branch_train_roles_v1/roles.json'
    roles=json.loads(roles_path.read_text())
    for n,h in roles['source_sha256'].items():
        if sha(ROOT/n)!=h:raise ValueError('Role inputs changed')
    source=DATA/'free-text.csv'
    manifest=json.loads((DATA/'split-manifest.json').read_text())
    allowed=set(roles['roles']['fit'])
    assert len(allowed)==47 and not allowed & set(manifest['sealed_test_subjects'])
    files=[Path(__file__),ROOT/'scripts/type2branch_keyrecs_adapter.py',
           ROOT/'scripts/type2branch_continuity.py',ROOT/'scripts/type2branch_input_audit.py',
           ROOT/'scripts/typenet_benchmark.py',
           ROOT/'research/benchmarks/references/type2branch/generate_dataset.py',roles_path]
    plan={'scope':'Base channels only; all47fit IDs/session1',
          'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in files},
          'raw_sha256':manifest['files']['free-text.csv']['sha256'],
          'window_rule':'First15full100rowwindows; source order; no overlap/substitution',
          'reserves':'Later windows unused by encoder; eventual support contract still separate',
          'adaptations':['Approximate key codes from prior TypeNet mapping; malformed keys unknown0',
                         'Row-order incoming timing=previousDD, with hold/key adjacency checks',
                         'Terminal event convention inferred from fitting observations',
                         'Round only numerical millisecond residue<=1e-6ms',
                         'Author code clipping30seconds; paper10seconds discrepancy retained'],
          'not_complete':['Synthetic residual channels','Reference synthesis parity','Training or DEV validation']}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    if sha(source)!=plan['raw_sha256']:raise ValueError('Source changed')
    grouped=defaultdict(list)
    with source.open() as stream:
        next(stream)
        for line in stream:
            subject,session,opaque=line.split(',',2)
            if subject in allowed and session=='1':grouped[subject].append(opaque)
    reference=reference_prepare()
    arrays=[]; signed=[]; subjects=[]; audit={}; maximum_error=0.; checked=0
    for subject in sorted(allowed):
        raw,base,info=adapt(grouped[subject])
        if len(base)<20:raise ValueError('Fitting identity lacks15fit plus5reserve windows')
        # Compare every fitting session window, not only retained encoder windows.
        for wire,features in zip(raw,base):
            press=np.r_[0,np.cumsum(wire[1:,2])]
            events=np.column_stack([press,press+wire[:,1],wire[:,0]])
            with contextlib.redirect_stdout(io.StringIO()):expected=reference(events,1000.)
            error=float(np.max(np.abs(expected-features)))
            if error>1e-12:raise ValueError('Author preprocessing mismatch')
            maximum_error=max(maximum_error,error);checked+=1
        info['encoder_windows']=15
        info['unused_full_windows']=len(base)-15
        audit[subject]=info
        arrays.append(base[:15]);signed.append(raw[:15]);subjects.extend([subject]*15)
    x=np.concatenate(arrays);wire=np.concatenate(signed)
    assert x.shape==(705,100,3) and len(set(subjects))==47
    np.savez_compressed(OUT/'fit_base.npz',base=x,signed_raw_ms=wire,subject=np.array(subjects),
                        window=np.tile(np.arange(15),47),true_length=np.full(705,100))
    report={'status':'fitting_base_channels_prepared_not_full_model_input','identities':47,
            'encoder_sequences':705,'encoder_events':70500,'shape':list(x.shape),
            'author_comparison_windows':checked,'maximum_absolute_author_error':maximum_error,
            'per_identity':audit,'array_sha256':sha(OUT/'fit_base.npz'),
            'remaining_channels':2,'selection_calibration_dev_test_values_decoded':False}
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='per_identity'}))


if __name__=='__main__':main()
