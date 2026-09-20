import subprocess
import sys
import numpy as np
import pytest
import type2branch_worker_client as client


def install(monkeypatch, mode):
    manifest={'subjects':['a','b'],'threshold':-1.}
    monkeypatch.setattr(client,'verify_release',lambda pin:manifest)
    original=subprocess.Popen
    program='''import json,sys,time
print(json.dumps({'kind':'ready','subjects':['a','b'],'threshold':-1.,'manifest_sha':'pin'}),flush=True)
for line in sys.stdin:
 r=json.loads(line)
 MODE
'''.replace(' MODE',mode)
    monkeypatch.setattr(client.subprocess,'Popen',lambda args,**kwargs:original([sys.executable,'-u','-c',program],**kwargs))


def test_reuses_process_and_closes(monkeypatch):
    install(monkeypatch," print(json.dumps({'kind':'scores','id':r['id'],'scores':[[0.,0.]]*len(r['features']),'accepted':[[True,True]]*len(r['features'])}),flush=True)")
    worker=client.WorkerClient('pin',timeout=2)
    try:
        x=np.zeros((1,100,5));result=worker.score(x);process=worker.process
        assert result=={'scores':[[0.,0.]],'accepted':[[True,True]]}
        worker.score(x)
        assert worker.process is process and worker.identifier==2
    finally:worker.close()
    assert process.poll() is not None and worker.process is None


@pytest.mark.parametrize('mode',[
    " print(json.dumps({'kind':'scores','id':r['id']+1}),flush=True)",
    " print(json.dumps({'kind':'scores','id':r['id'],'scores':[[0.,0.]],'accepted':[[False,False]]}),flush=True)",
    ' sys.exit(0)',
])
def test_bad_response_closes_worker(monkeypatch,mode):
    install(monkeypatch,mode)
    worker=client.WorkerClient('pin',timeout=2)
    with pytest.raises(RuntimeError):worker.score(np.zeros((1,100,5)))
    assert worker.process is None


def test_timeout_closes_worker(monkeypatch):
    install(monkeypatch,' time.sleep(10)')
    worker=client.WorkerClient('pin',timeout=.3)
    with pytest.raises(TimeoutError):worker.score(np.zeros((1,100,5)))
    assert worker.process is None
