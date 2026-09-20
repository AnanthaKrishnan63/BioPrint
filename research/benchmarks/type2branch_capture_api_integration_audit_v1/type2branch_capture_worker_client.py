"""Capture-specific protocol over the bounded serialized local IPC transport."""
import json
import os
import subprocess
import threading
import time
import numpy as np
from type2branch_worker_client import WorkerClient as Transport
from type2branch_capture_release import ROOT,verify_release
from type2branch_capture_wire import validate_request


class WorkerClient(Transport):
    def __init__(self,pin,timeout=60.):
        self.manifest=verify_release(pin)
        self.pin=pin;self.timeout=timeout
        if not 0<timeout<=120:raise ValueError('Invalid worker timeout')
        self.lock=threading.Lock();self.process=None;self.log=None;self.buffer=b'';self.identifier=0

    def _start(self,deadline):
        verify_release(self.pin)
        folder=ROOT/'.research-tmp';folder.mkdir(exist_ok=True)
        self.log=(folder/'type2branch-capture-worker.log').open('ab')
        self.process=subprocess.Popen(['/bin/bash',str(ROOT/'scripts/research.sh'),
            str(ROOT/'scripts/type2branch_capture_worker.py'),'--manifest-sha',self.pin],
            cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,bufsize=0)
        os.set_blocking(self.process.stdin.fileno(),False)
        os.set_blocking(self.process.stdout.fileno(),False)
        ready=self._read(deadline)
        if (ready.get('kind')!='ready' or ready.get('manifest_sha')!=self.pin
                or ready.get('subjects')!=self.manifest['subjects']
                or ready.get('thresholds')!=self.manifest['thresholds']):
            raise RuntimeError('Capture worker release handshake mismatch')

    def score(self,features,true_lengths):
        _,features,lengths=validate_request({'id':0,'features':features,'true_lengths':true_lengths})
        with self.lock:
            deadline=time.monotonic()+self.timeout
            try:
                if self.process is None:self._start(deadline)
                self.identifier+=1
                raw=(json.dumps({'id':self.identifier,'features':features.tolist(),
                                 'true_lengths':lengths.tolist()},allow_nan=False)+'\n').encode()
                if len(raw)>2_000_000:raise ValueError('Request oversized')
                self._write(raw,deadline);result=self._read(deadline)
                if result.get('kind')!='scores' or result.get('id')!=self.identifier:
                    raise RuntimeError('Worker response correlation mismatch')
                scores=np.asarray(result['scores'],dtype=np.float64);accepted=np.asarray(result['accepted'])
                expected=(len(features),len(self.manifest['subjects']))
                if scores.shape!=expected or not np.isfinite(scores).all():raise RuntimeError('Invalid worker scores')
                if accepted.shape!=expected or accepted.dtype.kind!='b':raise RuntimeError('Invalid worker decisions')
                thresholds=np.array([self.manifest['thresholds'][str(int(n))] for n in lengths],dtype=np.float64)
                if not np.array_equal(accepted,scores>=thresholds[:,None]):
                    raise RuntimeError('Worker length-threshold decisions disagree')
                return {'scores':scores.tolist(),'accepted':accepted.tolist()}
            except Exception:
                self._stop()
                raise
