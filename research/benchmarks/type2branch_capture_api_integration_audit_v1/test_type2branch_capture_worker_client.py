import subprocess
import sys
import numpy as np
import pytest
import type2branch_capture_worker_client as client


def install(monkeypatch,mode):
    manifest={'subjects':['a','b'],'thresholds':{'25':-1.,'50':1.,'75':-2.,'100':2.}}
    monkeypatch.setattr(client,'verify_release',lambda pin:manifest)
    original=subprocess.Popen
    program='''import json,sys,time
print(json.dumps({'kind':'ready','subjects':['a','b'],'thresholds':{'25':-1.,'50':1.,'75':-2.,'100':2.},'manifest_sha':'pin'}),flush=True)
for line in sys.stdin:
 r=json.loads(line)
 MODE
'''.replace(' MODE',mode)
    monkeypatch.setattr(client.subprocess,'Popen',lambda args,**kwargs:original([sys.executable,'-u','-c',program],**kwargs))


def test_different_lengths_reuse_process_and_close(monkeypatch):
    install(monkeypatch," print(json.dumps({'kind':'scores','id':r['id'],'scores':[[0.,0.]]*len(r['features']),'accepted':[[n==25,n==25] for n in r['true_lengths']]}),flush=True)")
    worker=client.WorkerClient('pin',timeout=2)
    try:
        result=worker.score(np.zeros((2,100,5)),[25,50]);process=worker.process
        assert result['accepted']==[[True,True],[False,False]]
        worker.score(np.zeros((1,100,5)),[25])
        assert worker.process is process and worker.identifier==2
    finally:worker.close()
    assert process.poll() is not None and worker.process is None


@pytest.mark.parametrize('mode',[
    " print(json.dumps({'kind':'scores','id':r['id']+1}),flush=True)",
    " print(json.dumps({'kind':'scores','id':r['id'],'scores':[[0.,0.]],'accepted':[[True,True]]}),flush=True)",
    ' sys.exit(0)',
])
def test_bad_response_closes_worker(monkeypatch,mode):
    install(monkeypatch,mode);worker=client.WorkerClient('pin',timeout=2)
    with pytest.raises(RuntimeError):worker.score(np.zeros((1,100,5)),[50])
    assert worker.process is None


def test_timeout_closes_worker(monkeypatch):
    install(monkeypatch,' time.sleep(10)');worker=client.WorkerClient('pin',timeout=.3)
    with pytest.raises(TimeoutError):worker.score(np.zeros((1,100,5)),[25])
    assert worker.process is None


def test_bad_padding_rejected_before_subprocess(monkeypatch):
    install(monkeypatch,' sys.exit(1)');worker=client.WorkerClient('pin')
    x=np.zeros((1,100,5));x[0,25,3]=1e-100
    with pytest.raises(ValueError):worker.score(x,[25])
    assert worker.process is None
