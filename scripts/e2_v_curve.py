"""Hack rate and read-time share V by checkpoint, semantic label, from every top-level run (E1 and E2).

Usage: e2_v_curve.py. Same 100 tasks x 8 reruns at every checkpoint; 95% intervals resample tasks.
"""
import csv
import glob
import gzip
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from labels import relabel_record
from vstats import read_time_share

TASKS = {t["task_id"]: t for t in json.load(open("data/tasks_v1.json", encoding="utf8"))["tasks"]}
OUT = "results/e2_top"
os.makedirs(OUT, exist_ok=True)

runs = {}
for path in glob.glob("runs/e1_top/raw/*_top.jsonl.gz") + glob.glob("runs/e2_top/raw/*_top.jsonl.gz"):
    runs[int(re.search(r"_a(\d{4})_top", path).group(1))] = path

rows = []
rng = np.random.default_rng(0)
for adapter in sorted(runs):
    g = defaultdict(list)
    with gzip.open(runs[adapter], "rt", encoding="utf8") as f:
        f.readline()
        for line in f:
            r = json.loads(line)
            lab = relabel_record(r, TASKS[r["task_id"]])
            g[r["task_id"]].append(bool(lab and lab["hack"]))
    ids = sorted(g)
    hits = np.array([sum(g[t]) for t in ids])
    totals = np.array([len(g[t]) for t in ids])
    v, ybar, _, _ = read_time_share(hits, totals)
    boot_v, boot_y = [], []
    for _ in range(2000):
        i = rng.integers(0, len(ids), len(ids))
        bv, by, _, _ = read_time_share(hits[i], totals[i])
        if bv == bv:
            boot_v.append(bv)
        boot_y.append(by)
    row = {
        "adapter": adapter, "hack_rate": ybar, "hack_lo": np.percentile(boot_y, 2.5), "hack_hi": np.percentile(boot_y, 97.5),
        "V": v, "V_lo": np.percentile(boot_v, 2.5), "V_hi": np.percentile(boot_v, 97.5),
        "mixed_tasks": int(((hits > 0) & (hits < totals)).sum()), "reruns": int(totals.sum()),
    }
    rows.append(row)
    print(f"adapter {adapter:>3}: hack {ybar:.3f} [{row['hack_lo']:.3f}, {row['hack_hi']:.3f}]  V {v:.3f} [{row['V_lo']:.3f}, {row['V_hi']:.3f}]  mixed {row['mixed_tasks']}")

with open(os.path.join(OUT, "v_curve.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
