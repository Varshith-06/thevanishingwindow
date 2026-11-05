"""Hack rate across training under the semantic label, from the published transcripts. Parsing only."""
import csv
import glob
import json
import os
import re
import sys
import textwrap
from collections import Counter, defaultdict

import numpy as np
from huggingface_hub import snapshot_download

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from labels import extract_code, semantic_label

root = snapshot_download("vgel/qwen3-8b-rh-sampler-ckpts", allow_patterns=["*/rollouts_with_chat.jsonl"])
SPLIT = re.compile(r"(?=<\|im_start\|>)")
SCAFFOLD = re.compile(r"<scaffold>\n(.*?)\n</scaffold>", re.S)
BIN = 10
bins = defaultdict(list)
for path in sorted(glob.glob(os.path.join(root, "step_*", "rollouts_with_chat.jsonl"))):
    step = int(os.path.basename(os.path.dirname(path))[5:])
    for line in open(path, encoding="utf8"):
        r = json.loads(line)
        parts = [x for x in SPLIT.split(r["chat"]) if x.strip()]
        user = next(x for x in parts if x.startswith("<|im_start|>user"))
        turns = [x for x in parts if x.startswith("<|im_start|>assistant")]
        m = SCAFFOLD.search(user)
        if not m or not turns:
            continue
        code = None
        for t in turns:
            c = extract_code(t)
            if c is not None:
                code = c
        lab = None
        if code is not None:
            try:
                lab = semantic_label(textwrap.dedent(code).strip(), m.group(1), "")
            except (SyntaxError, ValueError, IndexError, AttributeError):
                lab = None
        logged = r["metrics"]["changed_tests_metric"] > 0 or r["metrics"]["hacked_harness_metric"] > 0
        bins[(step // BIN) * BIN].append((r["group_idx"], step, logged, lab))

rows = []
print("steps    n  groups logged_any semantic  harness  edit   kinds")
for b in sorted(bins):
    items = bins[b]
    groups = defaultdict(list)
    for g, s, logged, lab in items:
        groups[(s, g)].append(bool(lab and lab["hack"]))
    sem = float(np.mean([bool(x[3] and x[3]["hack"]) for x in items]))
    # Bootstrap over groups, since a step's transcripts come from a couple of tasks.
    rng = np.random.default_rng(b)
    vals = list(groups.values())
    boot = []
    for _ in range(2000):
        pick = [vals[i] for i in rng.integers(0, len(vals), len(vals))]
        boot.append(np.mean([x for g in pick for x in g]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    row = {
        "steps": f"{b}-{b + BIN - 1}",
        "n": len(items),
        "groups": len(groups),
        "logged_any": float(np.mean([x[2] for x in items])),
        "semantic": sem,
        "lo": lo,
        "hi": hi,
        "harness": float(np.mean([bool(x[3] and x[3]["harness"]) for x in items])),
        "edit": float(np.mean([bool(x[3] and x[3]["edit_hack"]) for x in items])),
    }
    kinds = Counter(x[3]["edit_kind"] for x in items if x[3] and x[3]["edit_kind"] != "identical")
    print(f"{row['steps']:>7} {row['n']:>4} {row['groups']:>4} {row['logged_any']:>9.2f} {sem:>8.2f} [{lo:.2f},{hi:.2f}] {row['harness']:>6.2f} {row['edit']:>6.2f}  {dict(kinds.most_common(4))}")
    rows.append(row)
os.makedirs("results/e00_logs", exist_ok=True)
with open("results/e00_logs/semantic_curve.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
