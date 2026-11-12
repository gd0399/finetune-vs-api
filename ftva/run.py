"""Run one model over one split and record prediction, tokens and latency per note.

Results go to results/<label>/<split>.jsonl and can resume after an interruption.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .prompt import parse


def load(split: str, data_dir="data", limit: int | None = None) -> list[dict]:
    rows = [json.loads(l) for l in open(Path(data_dir) / f"{split}.jsonl")]
    return rows[:limit] if limit else rows


def run(backend, label: str, split: str, workers: int = 8, limit: int | None = None,
        data_dir="data", out_dir="results", spec: str = "") -> Path:
    rows = load(split, data_dir, limit)
    out = Path(out_dir) / label / f"{split}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in open(out) if l.strip()}
    todo = [r for r in rows if r["id"] not in done]

    def one(r):
        t0 = time.perf_counter()
        try:
            text, tin, tout = backend(r["note"], r["label"])
            err = None
        except Exception as e:
            text, tin, tout, err = None, 0, 0, repr(e)[:300]
        return {"id": r["id"], "style": r["style"], "raw": text, "pred": parse(text), "error": err,
                "input_tokens": tin, "output_tokens": tout, "latency_s": time.perf_counter() - t0}

    t0 = time.perf_counter()
    with open(out, "a") as f, ThreadPoolExecutor(workers) as ex:
        for i, res in enumerate(ex.map(one, todo), 1):
            f.write(json.dumps(res) + "\n")
            f.flush()
            if i % 50 == 0 or i == len(todo):
                print(f"  {label}/{split}: {i}/{len(todo)}", flush=True)
    wall = time.perf_counter() - t0
    if todo:  # throughput at this concurrency, used for self-hosting cost
        meta = out.with_suffix(".meta.json")
        meta.write_text(json.dumps({"label": label, "spec": spec, "split": split, "n": len(todo),
                                    "workers": workers,
                                    "wall_s": wall, "docs_per_hour": len(todo) / wall * 3600}))
    return out
