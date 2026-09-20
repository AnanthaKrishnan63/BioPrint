"""Fit a source-translated population mean model from frozen fitting inputs."""
import json
from pathlib import Path
import time
import numpy as np
from type2branch_continuity import ROOT, sha
from type2branch_context_model import MeanContextModel

OUT=ROOT/'research/benchmarks/type2branch_context_fit_v1'


def main():
    OUT.mkdir(exist_ok=False)
    previous=ROOT/'research/benchmarks/type2branch_synthesis_fit_v1'
    prior=json.loads((previous/'report.json').read_text())
    roles_path=ROOT/'research/benchmarks/type2branch_train_roles_v1/roles.json'
    roles=json.loads(roles_path.read_text())
    files=[Path(__file__),ROOT/'scripts/type2branch_context_model.py',
           ROOT/'scripts/type2branch_synthesis_cleanup.py',previous/'report.json',roles_path]
    ref=ROOT/'research/benchmarks/references/type2branch_synthesis'
    files += [ref/'context'/n for n in ['ContextSelector.cs','Builder.cs','MemoryStorage.cs','ModelStorage.cs']]
    files += [ref/'source/KSD-SLD/FiniteContexts'/n for n in
              ['Models/ModelFeeder.cs','Models/AvgStdev/AvgStdevModelLinear.cs',
               'Synthesizer/AverageSynthesizer.cs','ContextSelection/LongestAvailableContextSelector.cs']]
    plan={'scope':'Population fit on frozen47fit IDs,705sequences only; no other observation roles',
          'array_sha256':prior['output_sha256'],'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in files},
          'model':'Source-translated AvgStdev arithmetic, context0..7, minimum10observations',
          'sequence_order':'Frozen array order, subject-sorted then chronological first15windows',
          'query_scope':'Training-only missing-model coverage; no RNG fallback or synthesis output',
          'limitations':['Not C# runtime parity','Paper exact synthesis mode not verified',
                         'Validated input domain0..1500ms and int.MinValue only']}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    path=previous/'cleaned_fit.npz'
    if sha(path)!=plan['array_sha256']:raise ValueError('Input array changed')
    with np.load(path,allow_pickle=False) as a:
        rows=a['rows'];masks=a['partitions'];subjects=a['subject']
    ids,counts=np.unique(subjects,return_counts=True)
    assert set(ids)==set(roles['roles']['fit']) and (counts==15).all()
    assert rows.shape==(705,100,3) and masks.shape==(705,100)
    model=MeanContextModel();start=time.perf_counter()
    for sample,mask in zip(rows,masks):model.feed(sample,np.flatnonzero(mask))
    fit_seconds=time.perf_counter()-start
    model.save(OUT/'population.npz')
    orders=[]
    for sample in rows:
        values,selected=model.predict(sample[:,0]) # Reference Synthesize creates unpartitioned dummy.
        assert np.array_equal(np.isnan(values),selected<0)
        orders.append(selected)
    orders=np.stack(orders)
    counts_by_order={str(i):[int((orders[:,:,f]==i).sum()) for f in range(2)] for i in range(-1,8)}
    np.savez_compressed(OUT/'fit_context_orders.npz',orders=orders)
    result={'status':'fitting_population_mean_model_complete_missing_fallback_unresolved',
            'identities':47,'sequences':705,'models':len(model.models),'fit_seconds':fit_seconds,
            'context_counts_HT_FT':counts_by_order,
            'population_sha256':sha(OUT/'population.npz'),
            'context_orders_sha256':sha(OUT/'fit_context_orders.npz'),
            'limitations':plan['limitations']+['TRAIN coverage is not DEV performance',
              'Missing context remains NaN; no invented fallback',
              'No residual channels or neural model training yet']}
    (OUT/'report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()
