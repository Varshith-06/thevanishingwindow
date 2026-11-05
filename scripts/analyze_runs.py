"""Summarize sampled runs against the logged rates. Usage: analyze_runs.py DIR [DIR ...] --csv OUT"""
import csv
import glob
import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from vstats import read_time_share, bootstrap_v

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def load(path):
    with gzip.open(path, "rt", encoding="utf8") as f:
        meta = json.loads(f.readline())["meta"]
        rows = [json.loads(x) for x in f]
    return meta, rows


def logged_window(adapter, width=3):
    """Pooled logged rates for nearby adapters, since each step logs different tasks."""
    path = os.path.join(ROOT, "results", "e00_logs", "v_by_step.csv")
    rows = [r for r in csv.DictReader(open(path)) if abs(int(r["adapter_step"]) - adapter) <= width]
    if not rows:
        return {}
    f = lambda k: float(np.mean([float(r[k]) for r in rows]))
    exact = [r for r in rows if int(r["adapter_step"]) == adapter]
    exact_tok = float(exact[0]["first_ac_tokens"]) if exact else float("nan")
    return {"log_first_tok_exact": exact_tok, "log_any": f("any_rate"), "log_tests": f("tests_rate"), "log_harness": f("harness_rate"), "log_pass": f("pass_rate"), "log_turns": f("mean_turns") - 1, "log_first_tok": f("first_ac_tokens"), "log_V": f("any_V")}


def has_reasoning(text):
    """True if the think block holds words, not just repeated <think> tags."""
    m = re.match(r"\s*(?:<think>\s*)+(.*?)(?:</think>|$)", text, re.S)
    return bool(m and m.group(1).strip())


def first_attempt_hack(r):
    subs = r["submissions"]
    fl = subs[0].get("flags") if subs else None
    return bool(fl and (fl["changed_tests"] or fl["hacked_harness"]))


def summarize(meta, rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["task_id"]].append(r)
    gs = list(groups.values())
    totals = [len(g) for g in gs]
    out = {"adapter": meta["adapter"], "prefill": meta["prefill"], "n": len(rows), "tasks": len(gs), "bwrap": meta.get("bwrap")}
    out["tok_per_s"] = meta["gen_tokens"] / max(meta["gen_seconds"], 1e-9)
    out["wall_min"] = meta["wall_seconds"] / 60
    first = [r["turns"][0] for r in rows if r["turns"]]
    out["first_tok"] = float(np.mean([len(t["ids"]) for t in first]))
    out["first_capped"] = float(np.mean([t["finish"] == "length" for t in first]))
    out["first_think_open"] = float(np.mean(["</think>" in t["text"][:60] or t["text"].lstrip().startswith("<think>") for t in first]))
    out["think_reasoning"] = float(np.mean([has_reasoning(t["text"]) for t in first]))
    out["think_spam"] = float(np.mean([t["text"].count("<think>") > 1 for t in first]))
    if any(r["submissions"] for r in rows):
        for key, name in (("hack_any", "any"), ("hack_strict", "strict"), ("changed_tests", "tests"), ("hacked_harness", "harness"), ("test_pass", "pass")):
            hits = [sum(r[key] for r in g) for g in gs]
            v, ybar, _, _ = read_time_share(hits, totals)
            out[f"{name}_rate"] = ybar
            if name in ("any", "strict"):
                lo, hi = bootstrap_v(hits, totals, n=1000)
                out.update({f"{name}_V": v, f"{name}_V_lo": lo, f"{name}_V_hi": hi})
                out[f"{name}_mixed"] = float(np.mean([0 < h < t for h, t in zip(hits, totals)]))
        out["turns"] = float(np.mean([r["n_turns"] for r in rows]))
        out["first_attempt_hack_rate"] = float(np.mean([first_attempt_hack(r) for r in rows]))
        out["tokens_per_rollout"] = meta["gen_tokens"] / len(rows)
        out["edit_kinds"] = dict(Counter(r["edit_kind"] for r in rows if r["changed_tests"]).most_common())
    out.update(logged_window(meta["adapter"]))
    return out


def main():
    args = sys.argv[1:]
    csv_path = None
    if "--csv" in args:
        i = args.index("--csv")
        csv_path = args[i + 1]
        del args[i : i + 2]
    table = []
    for d in args:
        for p in sorted(glob.glob(os.path.join(d, "**", "*.jsonl.gz"), recursive=True)):
            meta, rows = load(p)
            s = summarize(meta, rows)
            s["file"] = os.path.basename(p).replace(".jsonl.gz", "")
            table.append(s)
    for s in table:
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}))
    if csv_path and table:
        keys = sorted({k for s in table for k in s})
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(table)


if __name__ == "__main__":
    main()
