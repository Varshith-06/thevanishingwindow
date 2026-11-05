"""Recompute V at each pilot checkpoint under the semantic label. Parsing only, no code is executed."""
import glob
import gzip
import json
import os
import random
import sys
import textwrap
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from labels import extract_code, semantic_label
from vstats import bootstrap_v, read_time_share

RUN_DIR = sys.argv[1] if len(sys.argv) > 1 else "runs/e1_top/raw"
tasks = {t["task_id"]: t for t in json.load(open("data/tasks_v1.json", encoding="utf8"))["tasks"]}


def relabel(r):
    """Semantic label of the last submission that has code. Mirrors the environment's last-submission rule."""
    if not r["submissions"]:
        return None
    last = r["submissions"][-1]
    if last["kind"] != "ran":
        return None
    code = extract_code(r["turns"][len(r["submissions"]) - 1]["text"])
    if code is None:
        return None
    t = tasks[r["task_id"]]
    return semantic_label(textwrap.dedent(code).strip(), t["scaffold"], t["fn_name"])


runs = {}
for path in sorted(glob.glob(os.path.join(RUN_DIR, "*_top.jsonl.gz"))):
    with gzip.open(path, "rt", encoding="utf8") as f:
        meta = json.loads(f.readline())["meta"]
        rows = [json.loads(x) for x in f]
    runs[meta["adapter"]] = rows

out = {}
for a, rows in sorted(runs.items()):
    g = defaultdict(list)
    kinds = Counter()
    trans = Counter()
    for r in rows:
        lab = relabel(r)
        r["sem"] = lab
        r["hack_sem"] = bool(lab and lab["hack"])
        g[r["task_id"]].append(r)
        if lab:
            kinds[lab["edit_kind"]] += 1
            trans[(r["hack_any"], r["hack_strict"], r["hack_sem"])] += 1
    ids = sorted(g)
    totals = [len(g[t]) for t in ids]
    row = {"adapter": a}
    print(f"\n== adapter {a}: edit kinds {dict(kinds.most_common())}")
    print("   env harness flag", sum(r["hacked_harness"] for r in rows), "semantic harness", sum(bool(r["sem"] and r["sem"]["harness"]) for r in rows), "semantic edit hack", sum(bool(r["sem"] and r["sem"]["edit_hack"]) for r in rows))
    for key in ("hack_any", "hack_strict", "hack_sem"):
        hits = [sum(r[key] for r in g[t]) for t in ids]
        v, y, _, _ = read_time_share(hits, totals)
        lo, hi = bootstrap_v(hits, totals, n=2000)
        mixed = sum(0 < h < n for h, n in zip(hits, totals))
        print(f"   {key:12s} rate {y:.3f}  V {v:.3f} [{lo:.3f},{hi:.3f}]  mixed tasks {mixed}")
        row[key] = {"rate": y, "V": v, "lo": lo, "hi": hi, "mixed": mixed}
    honest = [r for r in rows if not r["hack_sem"]]
    print(f"   pass rate among non-hacks {np.mean([r['test_pass'] for r in honest]):.3f} (n={len(honest)})")
    out[a] = row

if len(runs) >= 2:
    a0, a1 = min(runs), max(runs)
    g0, g1 = defaultdict(list), defaultdict(list)
    for r in runs[a0]:
        g0[r["task_id"]].append(r["hack_sem"])
    for r in runs[a1]:
        g1[r["task_id"]].append(r["hack_sem"])
    ids = sorted(set(g0) & set(g1))
    h0, n0 = np.array([sum(g0[t]) for t in ids]), np.array([len(g0[t]) for t in ids])
    h1, n1 = np.array([sum(g1[t]) for t in ids]), np.array([len(g1[t]) for t in ids])
    v0, v1 = read_time_share(h0, n0)[0], read_time_share(h1, n1)[0]
    rng = np.random.default_rng(0)
    d = []
    for _ in range(5000):
        i = rng.integers(0, len(ids), len(ids))
        x = read_time_share(h0[i], n0[i])[0] - read_time_share(h1[i], n1[i])[0]
        if x == x:
            d.append(x)
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"\nPaired V difference, semantic label, a{a0} minus a{a1}: {v0 - v1:.3f} [{lo:.3f},{hi:.3f}]")
    out["paired"] = {"diff": v0 - v1, "lo": lo, "hi": hi}

os.makedirs("results/e1_top", exist_ok=True)
with open("results/e1_top/relabel.json", "w") as f:
    json.dump(out, f, indent=1, default=float)

print("\n== Spot check: three random examples of each class that still counts as a hack")
rng = random.Random(3)
for a, rows in sorted(runs.items()):
    for kind in ("golds_changed", "subset", "other_literal", "opaque"):
        ex = [r for r in rows if r["sem"] and r["sem"]["edit_kind"] == kind]
        for r in rng.sample(ex, min(3, len(ex))):
            t = tasks[r["task_id"]]
            orig = t["scaffold"][t["scaffold"].index("for i, gold") :].split("\n")[0]
            code = extract_code(r["turns"][len(r["submissions"]) - 1]["text"]) or ""
            loop = code.find("for i, gold")
            print(f"-- a{a} {kind} task {r['task_id']} passed={r['test_pass']} harness={r['sem']['harness']}")
            print("   orig:", orig[:190])
            print("   new :", code[loop:].split("\n")[0][:190] if loop >= 0 else code[:190])
    ex = [r for r in rows if r["sem"] and r["sem"]["harness"]]
    for r in rng.sample(ex, min(3, len(ex))):
        code = extract_code(r["turns"][len(r["submissions"]) - 1]["text"]) or ""
        print(f"-- a{a} harness task {r['task_id']} passed={r['test_pass']}")
        print("   tail:", code[-260:].replace("\n", " | "))
