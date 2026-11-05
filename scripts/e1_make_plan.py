"""Choose which reruns to cut. Usage: e1_make_plan.py RUN.jsonl.gz OUT.json [--tasks 30] [--label hack_any]

Reruns are drawn uniformly within each mixed task, not balanced by outcome: balancing would
inflate the between-rerun variance that D(t) measures.
"""
import argparse
import gzip
import json
import random
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("out")
ap.add_argument("--tasks", type=int, default=30)
ap.add_argument("--reruns", type=int, default=4)
ap.add_argument("--label", default="hack_any")
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()

groups = defaultdict(list)
with gzip.open(a.run, "rt", encoding="utf8") as f:
    f.readline()
    for line in f:
        r = json.loads(line)
        groups[r["task_id"]].append(r)
mixed = [t for t, g in groups.items() if 0 < sum(r[a.label] for r in g) < len(g)]
rng = random.Random(a.seed)
rng.shuffle(mixed)
plan = []
for t in sorted(mixed[: a.tasks]):
    plan.append({"task_id": t, "reruns": sorted(rng.sample([r["rerun"] for r in groups[t]], a.reruns))})
with open(a.out, "w", encoding="utf8") as f:
    json.dump(plan, f)
print(f"{len(groups)} tasks, {len(mixed)} mixed on {a.label}, planned {len(plan)}")
