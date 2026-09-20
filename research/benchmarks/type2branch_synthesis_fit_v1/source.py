"""Apply source-translated CLI cleanup to the frozen fitting arrays only."""
import json
from pathlib import Path
import numpy as np
from type2branch_continuity import ROOT, sha
from type2branch_synthesis_cleanup import cleanup, INVALID_TIMING

OUT=ROOT/'research/benchmarks/type2branch_synthesis_fit_v1'


def main():
    OUT.mkdir(exist_ok=False)
    prior=ROOT/'research/benchmarks/type2branch_fit_base_v1'
    report=json.loads((prior/'report.json').read_text())
    inputs=[Path(__file__),ROOT/'scripts/type2branch_synthesis_cleanup.py',
            ROOT/'scripts/type2branch_csv_bridge.py',prior/'report.json',
            ROOT/'research/benchmarks/references/type2branch_synthesis/cleanup/CleanFTs.cs',
            ROOT/'research/benchmarks/references/type2branch_synthesis/cleanup/ThresholdPartitioner.cs',
            ROOT/'research/benchmarks/references/type2branch_synthesis/source/KSD-SLD/Program.cs']
    plan={'scope':'Frozen705fitting windows only; source-translated cleanup, not runtime parity',
          'array_sha256':report['array_sha256'],
          'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in inputs}}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    path=prior/'fit_base.npz'
    if sha(path)!=plan['array_sha256']:raise ValueError('Frozen array changed')
    with np.load(path,allow_pickle=False) as a:
        raw=a['signed_raw_ms'];subject=a['subject'];window=a['window']
    assert raw.shape==(705,100,3) and len(np.unique(subject))==47
    rows=[];masks=[]
    for sample in raw:
        cleaned,partitions=cleanup(sample)
        mask=np.zeros(100,dtype=bool);mask[partitions]=True
        rows.append(cleaned);masks.append(mask)
    result=np.stack(rows);pause=np.stack(masks)
    np.savez_compressed(OUT/'cleaned_fit.npz',rows=result,partitions=pause,subject=subject,window=window)
    details={'status':'fitting_cli_cleanup_prepared','sequences':705,'identities':47,
             'negative_input_ft':int((raw[:,:,2]<0).sum()),
             'input_ht_above1500':int((raw[:,:,1]>1500).sum()),
             'input_ft_above1500':int((raw[:,:,2]>1500).sum()),
             'pause_partitions':int(pause.sum()),'invalid_output_ft':int((result[:,:,2]==INVALID_TIMING).sum()),
             'output_sha256':sha(OUT/'cleaned_fit.npz'),
             'limitations':['No C# runtime parity or population fitting',
                            'This CLI caps input timing to1.5s, separate from neural base30sclip',
                            'Negative FT becomes1500ms except first/pause sentinel positions',
                            'Synthetic residual generation remains incomplete']}
    (OUT/'report.json').write_text(json.dumps(details,indent=2))
    print(json.dumps(details))


if __name__=='__main__':main()
