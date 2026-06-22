# REDESIGN/qwen_pool.py
from __future__ import annotations
import atexit
import threading
import uuid
import queue
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional
import multiprocessing as mp

# Import the updated worker_main
from .qwen_worker import worker_main

@dataclass
class _Worker:
    worker_id: str
    pair: Tuple[int, ...]
    proc: mp.Process
    in_q: Any

class QwenLayeredPool:
    """
    Runs Qwen layered inference in dedicated processes, one per GPU pair.
    """
    
    def __init__(self, pairs: Optional[List[Tuple[int, ...]]] = None):
        self._configured_pairs = pairs
        self._out_q = mp.Queue()
        self._workers: List[_Worker] = []
        self._available = queue.Queue()
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._cv = threading.Condition()
        self._collector_th: Optional[threading.Thread] = None
        self._started = False
        self._shutdown = False
    
    @property
    def pairs(self) -> List[Tuple[int, ...]]:
        if self._configured_pairs is not None:
            return self._configured_pairs
        
        # Look up dynamically from tool_gpu_config
        try:
            from .tool_gpu_config import get_qwen_gpu_pairs
            return get_qwen_gpu_pairs()
        except ImportError:
            # Fallback if config module missing
            print("[QwenPool] Warning: tool_gpu_config not found, defaulting to GPU 0")
            return [(0,)]
    
    def start(self):
        if self._started:
            return
        self._shutdown = False
        
        current_pairs = self.pairs
        print(f"[QwenPool] Starting with GPU pairs: {current_pairs}")
        
        for pair in current_pairs:
            wid = f"qwen_{'_'.join(map(str, pair))}"
            in_q = mp.Queue()
            proc = mp.Process(
                target=worker_main,
                args=(wid, pair, in_q, self._out_q),
                daemon=True,
            )
            proc.start()
            self._workers.append(_Worker(wid, pair, proc, in_q))
            self._available.put(wid)
            print(f"[QwenPool] Started worker {wid} on GPUs {pair}")
        
        self._collector_th = threading.Thread(target=self._collector_loop, daemon=True)
        self._collector_th.start()
        
        self._started = True
        print(f"[QwenPool] Pool started with {len(self._workers)} worker(s)")
    
    def _collector_loop(self):
        while not self._shutdown:
            try:
                msg = self._out_q.get(timeout=0.2)
            except Exception:
                continue
            if msg is None:
                continue
            
            job_id = msg.get("job_id")
            
            # Worker Init Failure
            if job_id is None:
                worker_id = msg.get("worker_id", "unknown")
                error = msg.get("error", "Unknown error")
                print(f"[QwenPool] Worker {worker_id} init error: {error}")
                continue
            
            with self._cv:
                self._pending[job_id] = msg
                self._cv.notify_all()
    
    def submit(
        self,
        *,
        image_path: str,
        output_dir: str,
        timeout: float = 600.0,
        **kwargs
    ) -> Dict[str, Any]:
        if not self._started:
            self.start()
        
        try:
            wid = self._available.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"No available Qwen worker within {timeout}s")
        
        worker = next(w for w in self._workers if w.worker_id == wid)
        
        job_id = uuid.uuid4().hex
        payload = {
            "job_id": job_id,
            "image_path": image_path,
            "output_dir": output_dir,
            **kwargs
        }
        
        try:
            worker.in_q.put(payload)
            
            with self._cv:
                start_time = __import__("time").time()
                while job_id not in self._pending:
                    remaining = timeout - (__import__("time").time() - start_time)
                    if remaining <= 0:
                        raise TimeoutError(f"Job {job_id} timed out after {timeout}s")
                    self._cv.wait(timeout=min(remaining, 0.5))
                
                msg = self._pending.pop(job_id)
            
            if not msg.get("ok", False):
                err = msg.get("error", "Unknown error")
                trace = msg.get("trace", "")
                print(f"[QwenPool] Job {job_id} failed on {worker.pair}: {err}")
                if trace:
                    print(f"[QwenPool] Traceback:\n{trace}")
                
                self._restart_worker(worker.worker_id)
                raise RuntimeError(f"Qwen inference failed: {err}")
            
            return msg["data"]
            
        finally:
            if not self._shutdown:
                self._available.put(wid)
    
    def _restart_worker(self, worker_id: str):
        for i, w in enumerate(self._workers):
            if w.worker_id != worker_id:
                continue
            
            print(f"[QwenPool] Restarting worker {worker_id}...")
            try:
                if w.proc.is_alive():
                    w.proc.terminate()
                    w.proc.join(timeout=2)
                    if w.proc.is_alive():
                        w.proc.kill()
                        w.proc.join(timeout=1)
            except Exception as e:
                print(f"[QwenPool] Error terminating worker {worker_id}: {e}")
            
            in_q = mp.Queue()
            proc = mp.Process(
                target=worker_main,
                args=(w.worker_id, w.pair, in_q, self._out_q),
                daemon=True,
            )
            proc.start()
            self._workers[i] = _Worker(w.worker_id, w.pair, proc, in_q)
            print(f"[QwenPool] Worker {w.worker_id} restarted on GPUs {w.pair}")
            return
    
    def get_status(self) -> Dict[str, Any]:
        status = {
            "started": self._started,
            "shutdown": self._shutdown,
            "total_workers": len(self._workers),
            "available_workers": self._available.qsize(),
            "pending_jobs": len(self._pending),
            "workers": [],
        }
        for w in self._workers:
            worker_status = {
                "worker_id": w.worker_id,
                "gpu_pair": list(w.pair),
                "alive": w.proc.is_alive() if w.proc else False,
            }
            status["workers"].append(worker_status)
        return status
    
    def shutdown(self):
        if self._shutdown:
            return
        
        print("[QwenPool] Shutting down...")
        self._shutdown = True
        
        for w in self._workers:
            try:
                w.in_q.put(None)
            except Exception:
                pass
        
        for w in self._workers:
            try:
                if w.proc.is_alive():
                    w.proc.join(timeout=5)
                    if w.proc.is_alive():
                        print(f"[QwenPool] Force killing worker {w.worker_id}")
                        w.proc.terminate()
                        w.proc.join(timeout=2)
                        if w.proc.is_alive():
                            w.proc.kill()
            except Exception as e:
                print(f"[QwenPool] Error shutting down worker {w.worker_id}: {e}")
        
        self._workers.clear()
        self._started = False
        print("[QwenPool] Shutdown complete")

# Singleton Management
_pool_singleton: Optional[QwenLayeredPool] = None
_pool_lock = threading.Lock()

def get_qwen_pool(pairs: Optional[List[Tuple[int, ...]]] = None) -> QwenLayeredPool:
    global _pool_singleton
    with _pool_lock:
        if _pool_singleton is None:
            _pool_singleton = QwenLayeredPool(pairs=pairs)
            _pool_singleton.start()
            atexit.register(_pool_singleton.shutdown)
        return _pool_singleton

def reset_qwen_pool():
    global _pool_singleton
    with _pool_lock:
        if _pool_singleton is not None:
            _pool_singleton.shutdown()
            _pool_singleton = None
            print("[QwenPool] Singleton reset")

def init_qwen_pool_with_gpus(gpu_ids: List[int]) -> QwenLayeredPool:
    global _pool_singleton
    with _pool_lock:
        if _pool_singleton is not None:
            _pool_singleton.shutdown()
        
        pairs = [tuple(gpu_ids)]
        _pool_singleton = QwenLayeredPool(pairs=pairs)
        _pool_singleton.start()
        atexit.register(_pool_singleton.shutdown)
        return _pool_singleton