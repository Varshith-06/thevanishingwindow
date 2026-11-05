"""Choose which reruns to cut. Usage: e1_make_plan.py RUN.jsonl.gz OUT.json [--tasks 30]

Tasks are mixed on the semantic label. Reruns are drawn uniformly within each task, not balanced by
outcome: balancing would inflate the between-rerun variance that D(t) measures.
"""
import argparse
import gzip
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from labels import relabel_record

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("out")
ap.add_argument("--tasks", type=int, default=30)
ap.add_argument("--reruns", type=int, default=4)
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()

task_info = {t["task_id"]: t for t in json.load(open("data/tasks_v1.json", encoding="utf8"))["tasks"]}
groups = defaultdict(list)
with gzip.open(a.run, "rt", encoding="utf8") as f:
    f.readline()
    for line in f:
        r = json.loads(line)
        lab = relabel_record(r, task_info[r["task_id"]])
        groups[r["task_id"]].append((r["rerun"], bool(lab and lab["hack"])))
mixed = [t for t, g in groups.items() if 0 < sum(y for _, y in g) < len(g)]
rng = random.Random(a.seed)
rng.shuffle(mixed)
plan = [{"task_id": t, "reruns": sorted(rng.sample([b for b, _ in groups[t]], a.reruns))} for t in sorted(mixed[: a.tasks])]
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w", encoding="utf8") as f:
    json.dump(plan, f)
print(f"{len(groups)} tasks, {len(mixed)} mixed on the semantic label, planned {len(plan)}")
