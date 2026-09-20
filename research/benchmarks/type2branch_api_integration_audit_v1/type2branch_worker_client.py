"""Serialized, bounded local worker IPC; no TensorFlow imports in the API process."""
import json
import os
from pathlib import Path
import selectors
import subprocess
import threading
import time
import numpy as np
from type2branch_release import ROOT, verify_release
from type2branch_wire import validate_features


class WorkerClient:
    def __init__(self, pin, timeout=60.):
        self.manifest=verify_release(pin)
        self.pin=pin;self.timeout=timeout
        if not 0<timeout<=120:raise ValueError('Invalid worker timeout')
        self.lock=threading.Lock();self.process=None;self.log=None;self.buffer=b'';self.identifier=0

    def _stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill();self.process.wait(timeout=2)
            for stream in [self.process.stdin,self.process.stdout]:
                if stream is not None:stream.close()
        self.process=None;self.buffer=b''
        if self.log is not None:self.log.close()
        self.log=None

    def close(self):
        with self.lock:self._stop()

    def _wait(self, stream, event, deadline):
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('Worker deadline exceeded')
        with selectors.DefaultSelector() as selector:
            selector.register(stream,event)
            if not selector.select(remaining):raise TimeoutError('Worker deadline exceeded')

    def _read(self, deadline):
        while b'\n' not in self.buffer:
            self._wait(self.process.stdout,selectors.EVENT_READ,deadline)
            piece=os.read(self.process.stdout.fileno(),65536)
            if not piece:raise RuntimeError('Worker exited before response')
            self.buffer+=piece
            if len(self.buffer)>2_000_000:raise RuntimeError('Worker response oversized')
        raw,self.buffer=self.buffer.split(b'\n',1)
        value=json.loads(raw)
        if not isinstance(value,dict):raise RuntimeError('Worker response must be object')
        return value

    def _write(self, raw, deadline):
        offset=0
        while offset<len(raw):
            self._wait(self.process.stdin,selectors.EVENT_WRITE,deadline)
            try:count=os.write(self.process.stdin.fileno(),raw[offset:])
            except BlockingIOError:continue
            if count<=0:raise RuntimeError('Worker write failed')
            offset+=count

    def _start(self, deadline):
        # Recheck immutable release before each process startup.
        verify_release(self.pin)
        folder=ROOT/'.research-tmp';folder.mkdir(exist_ok=True)
        self.log=(folder/'type2branch-worker.log').open('ab')
        self.process=subprocess.Popen(['/bin/bash',str(ROOT/'scripts/research.sh'),
            str(ROOT/'scripts/type2branch_worker.py'),'--manifest-sha',self.pin],
            cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,bufsize=0)
        os.set_blocking(self.process.stdin.fileno(),False)
        os.set_blocking(self.process.stdout.fileno(),False)
        ready=self._read(deadline)
        if (ready.get('kind')!='ready' or ready.get('manifest_sha')!=self.pin
                or ready.get('subjects')!=self.manifest['subjects']
                or ready.get('threshold')!=self.manifest['threshold']):
            raise RuntimeError('Worker release handshake mismatch')

    def score(self, features):
        features=validate_features(features)
        with self.lock:
            deadline=time.monotonic()+self.timeout
            try:
                if self.process is None:self._start(deadline)
                self.identifier+=1
                raw=(json.dumps({'id':self.identifier,'features':features.tolist()},allow_nan=False)+'\n').encode()
                if len(raw)>2_000_000:raise ValueError('Request oversized')
                self._write(raw,deadline)
                result=self._read(deadline)
                if result.get('kind')!='scores' or result.get('id')!=self.identifier:
                    raise RuntimeError('Worker response correlation mismatch')
                scores=np.asarray(result['scores'],dtype=np.float64)
                accepted=np.asarray(result['accepted'])
                expected=(len(features),len(self.manifest['subjects']))
                if scores.shape!=expected or not np.isfinite(scores).all():raise RuntimeError('Invalid worker scores')
                if accepted.shape!=expected or accepted.dtype.kind!='b':raise RuntimeError('Invalid worker decisions')
                if not np.array_equal(accepted,scores>=np.float64(self.manifest['threshold'])):
                    raise RuntimeError('Worker threshold decisions disagree')
                return {'scores':scores.tolist(),'accepted':accepted.tolist()}
            except Exception:
                self._stop()
                raise
