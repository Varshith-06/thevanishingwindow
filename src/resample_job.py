"""Cut finished reruns at fixed token positions and resample continuations from each cut."""
import argparse
import gzip
import json
import os
import random
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from engine import from_cut, run_rollouts
from sample_job import setup, write_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=int, required=True)
    ap.add_argument("--from-run", required=True, help="gzipped rerun records, relative to the repo root")
    ap.add_argument("--plan", required=True, help="JSON list of {task_id, reruns}, relative to the repo root")
    ap.add_argument("--cuts", default="32,64,128,256")
    ap.add_argument("--k-prime", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/kaggle/tmp/vw")
    ap.add_argument("--ngpu", type=int, default=2)
    a = ap.parse_args()
    if os.path.exists(a.out):
        print("already done", a.out, flush=True)
        return
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with gzip.open(os.path.join(root, a.from_run), "rt", encoding="utf8") as f:
        f.readline()
        records = {(r["task_id"], r["rerun"]): r for r in map(json.loads, f)}
    plan = json.load(open(os.path.join(root, a.plan), encoding="utf8"))

    tok, llm, params, data, use_bwrap = setup(a)
    tasks = {t["task_id"]: t for t in data["tasks"]}
    cuts = [int(c) for c in a.cuts.split(",")]
    rollouts, finished = [], 0
    for item in plan:
        for rerun in item["reruns"]:
            rec = records[(item["task_id"], rerun)]
            for t in cuts:
                for c in range(a.k_prime):
                    r = from_cut(rec, tasks[item["task_id"]], data["system"], t, use_bwrap, {"rerun": rerun, "cut": t, "cont": c})
                    if r is None:
                        finished += 1
                    else:
                        rollouts.append(r)
    print(f"{len(rollouts)} continuations; {finished} skipped because the rerun ended before the cut", flush=True)
    random.Random(a.seed).shuffle(rollouts)
    t0 = time.time()
    partial = a.out.replace(".jsonl.gz", ".partial.jsonl.gz")
    stats = run_rollouts(llm, params, tok, rollouts, use_bwrap, checkpoint=lambda s: write_records(partial, dict(s, partial=True), rollouts))
    stats.update({"wall_seconds": time.time() - t0, "bwrap": use_bwrap, "adapter": a.adapter, "prefill": "open", "n": len(rollouts), "cuts": cuts, "k_prime": a.k_prime})
    print(json.dumps(stats), flush=True)
    write_records(a.out, stats, rollouts)
    shutil.rmtree(os.path.join(a.work, f"merged_{a.adapter:04d}"), ignore_errors=True)


if __name__ == "__main__":
    main()
