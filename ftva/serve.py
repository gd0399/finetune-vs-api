"""Start and stop a vLLM server from Python (used by the notebook)."""
import subprocess
import time

import requests


def start(model: str, name: str, port: int = 8000, max_len: int = 4096, log: str = "vllm.log"):
    cmd = ["vllm", "serve", model, "--served-model-name", name, "--port", str(port),
           "--max-model-len", str(max_len), "--gpu-memory-utilization", "0.90"]
    proc = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    url = f"http://localhost:{port}/v1/models"
    for _ in range(600):  # model download + load can take several minutes
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited early, see {log}")
        try:
            if requests.get(url, timeout=2).ok:
                print(f"vLLM ready: {name}")
                return proc
        except requests.RequestException:
            pass
        time.sleep(2)
    proc.kill()
    raise TimeoutError(f"vLLM did not start, see {log}")


def stop(proc):
    proc.terminate()
    try:
        proc.wait(60)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(5)  # let the GPU memory free up
