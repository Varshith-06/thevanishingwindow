"""Audit the environment's hack flags on the published transcripts, by training phase."""
import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from huggingface_hub import snapshot_download
from labels import ast_flags, classify_test_edit, extract_code

root = snapshot_download("vgel/qwen3-8b-rh-sampler-ckpts", allow_patterns=["*/rollouts_with_chat.jsonl"])
SPLIT = re.compile(r"(?=<\|im_start\|>)")
SCAFFOLD = re.compile(r"<scaffold>\n(.*?)\n</scaffold>", re.S)
PHASES = [("01-30", 1, 30), ("31-50", 31, 50), ("51-70", 51, 70), ("71-102", 71, 102)]


def phase(step):
    return next(n for n, a, b in PHASES if a <= step <= b)


agree = Counter()
types = defaultdict(Counter)
any_flag = Counter()
total = Counter()
first_turn = defaultdict(Counter)
examples = []
for f in sorted(glob.glob(os.path.join(root, "step_*", "rollouts_with_chat.jsonl"))):
    step = int(os.path.basename(os.path.dirname(f))[5:])
    ph = phase(step)
    for line in open(f, encoding="utf8"):
        r = json.loads(line)
        parts = [x for x in SPLIT.split(r["chat"]) if x.strip()]
        user = next(x for x in parts if x.startswith("<|im_start|>user"))
        turns = [x for x in parts if x.startswith("<|im_start|>assistant")]
        m = SCAFFOLD.search(user)
        if not m or not turns:
            agree["no_scaffold_or_turn"] += 1
            continue
        scaffold = m.group(1)
        total[ph] += 1
        logged_tests = r["metrics"]["changed_tests_metric"] > 0
        logged_harness = r["metrics"]["hacked_harness_metric"] > 0
        code = None
        for t in turns:
            c = extract_code(t)
            if c is not None:
                code = c
        if code is None:
            agree["no_code"] += 1
            continue
        try:
            flags = ast_flags(code, scaffold, "")
        except (SyntaxError, ValueError, IndexError, AttributeError):
            agree["scaffold_parse_fail"] += 1
            continue
        if flags is None:
            agree["syntax_error"] += 1
            continue
        agree["tests_match" if flags["changed_tests"] == logged_tests else "tests_mismatch"] += 1
        agree["harness_match" if flags["hacked_harness"] == logged_harness else "harness_mismatch"] += 1
        any_flag[ph] += logged_tests or logged_harness
        if logged_tests:
            kind = classify_test_edit(code, scaffold)
            types[ph][kind] += 1
            if len(examples) < 400:
                examples.append({"step": step, "group": r["group_idx"], "traj": r["traj_idx"], "kind": kind, "passed": r["metrics"]["test_pass_reward"]})
        # Which attempt first shows the hack
        first = None
        for i, t in enumerate(turns):
            c = extract_code(t)
            if c is None:
                continue
            fl = ast_flags(c, scaffold, "")
            if fl and (fl["changed_tests"] or fl["hacked_harness"]):
                first = i
                break
        if logged_tests or logged_harness:
            first_turn[ph]["attempt_%d" % (first + 1) if first is not None else "unlocated"] += 1

print("agreement with logged metrics", dict(agree))
for name, _, _ in PHASES:
    n = sum(types[name].values())
    print(name, "transcripts", total[name], "flagged", any_flag[name], "tests-flagged", n)
    print("   edit types", dict(types[name].most_common()))
    print("   first hacked attempt", dict(first_turn[name].most_common()))
os.makedirs("results/e00_logs", exist_ok=True)
with open("results/e00_logs/label_audit.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["phase", "kind", "count"])
    w.writeheader()
    for name, _, _ in PHASES:
        for k, v in types[name].most_common():
            w.writerow({"phase": name, "kind": k, "count": v})
