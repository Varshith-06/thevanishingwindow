"""D(t) and the decision ceiling C(t) from cut-and-resample runs (E1 and E2). Usage: e1_cut_analysis.py ADAPTER [...]

Labels are semantic, recomputed from stored text for reruns and continuations alike. A cut counts as
"before the hack" only if the rerun's first hack token comes after it, so a prefix that already
contains the hack is not treated as warning.

A rerun that ended before the cut counts as decided, which is true but says nothing about the decision
forming early, so the *_running columns repeat D and C on reruns still running at the cut. D_p tests D > 0
against a parametric null (d_null); the bootstrap interval for D excludes zero too often near zero.
"""
import csv
import gzip
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from labels import extract_code, relabel_record
from vstats import read_time_share
from vstats_cut import bootstrap, build_cut_tasks, d_null, determined_share, only_running, within_task_auc

TASKS = {t["task_id"]: t for t in json.load(open("data/tasks_v1.json", encoding="utf8"))["tasks"]}
OUT = "results/e1_cut"
os.makedirs(OUT, exist_ok=True)


def find(kind, adapter):
    """Run file for one adapter, from E1 or E2 (kind is "top" or "cut"); None if neither has it."""
    for e in ("e1", "e2"):
        path = f"runs/{e}_{kind}/raw/{e}_a{adapter:04d}_{kind}.jsonl.gz"
        if os.path.exists(path):
            return path
    return None


def load(path):
    with gzip.open(path, "rt", encoding="utf8") as f:
        meta = json.loads(f.readline())["meta"]
        rows = [json.loads(x) for x in f]
    for r in rows:
        lab = relabel_record(r, TASKS[r["task_id"]])
        r["y"] = bool(lab and lab["hack"])
    return meta, rows


def hack_token(r):
    """Approximate generated-token index where the hack first appears: the test loop or the first
    TEST_PASS print inside the code of the first hacked attempt. None if not hacked."""
    if not r["y"]:
        return None
    for i, turn in enumerate(r["turns"][: len(r["submissions"])]):
        text = turn["text"]
        code_at = text.find("<code>")
        if code_at < 0 or extract_code(text) is None:
            continue
        sub = r["submissions"][i]
        sem = sub.get("semantic")
        if sem is None:
            lab = relabel_record({"submissions": r["submissions"][: i + 1], "turns": r["turns"]}, TASKS[r["task_id"]])
            sem = lab
        if not (sem and sem.get("hack")):
            continue
        marks = [text.find(m, code_at) for m in ("for i, gold", 'print("TEST_PASS"', "print('TEST_PASS'", "TEST_PASS")]
        marks = [m for m in marks if m >= 0]
        char = min(marks) if marks else code_at
        offset = sum(len(t["ids"]) for t in r["turns"][:i])
        return int(offset + len(turn["ids"]) * char / max(1, len(text)))
    return None


def code_token(r):
    text = r["turns"][0]["text"]
    c = text.find("<code>")
    return int(len(r["turns"][0]["ids"]) * c / max(1, len(text))) if c >= 0 else None


def analyze(adapter):
    _, top = load(find("top", adapter))
    meta, conts = load(find("cut", adapter))
    for r in conts:
        r["hack_sem"] = r["y"]
    for r in top:
        r["hack_sem"] = r["y"]
    g = defaultdict(list)
    for r in top:
        g[r["task_id"]].append(r)
    ids = sorted(g)
    v, ybar, _, _ = read_time_share([sum(r["y"] for r in g[t]) for t in ids], [len(g[t]) for t in ids])
    by_key = {(r["task_id"], r["rerun"]): r for r in top}
    planned = {(r["task_id"], r["origin"]["rerun"]) for r in conts}
    hpos = {k: hack_token(by_key[k]) for k in planned}
    cpos = {k: code_token(by_key[k]) for k in planned}
    hacked = [h for h in hpos.values() if h is not None]
    print(f"\n== adapter {adapter}: top-level hack rate {ybar:.3f}, V {v:.3f}; cut tasks {len({k[0] for k in planned})}, cut reruns {len(planned)}, hacked {len(hacked)}")
    if hacked:
        print(f"   hack token among cut reruns, quartiles: {np.percentile(hacked, [25, 50, 75]).astype(int).tolist()}")
    rows = []
    for t in meta["cuts"]:
        tasks = build_cut_tasks(top, conts, t, "hack_sem")
        d = determined_share(tasks)
        d_ci = bootstrap(tasks, determined_share, n=2000)
        c = within_task_auc(tasks)
        c_ci = bootstrap(tasks, within_task_auc, n=2000)
        pre_hack = [k for k in planned if hpos[k] is None or t < hpos[k]]
        # Same quantities using only prefixes written before any hack token.
        pre_tasks = build_cut_tasks(top, [r for r in conts if (r["task_id"], r["origin"]["rerun"]) in set(pre_hack)], t, "hack_sem")
        d_pre = determined_share(pre_tasks)
        c_pre = within_task_auc(pre_tasks)
        hacked_pre = sum(1 for k in pre_hack if hpos[k] is not None)
        d_null_mean, d_p = d_null(tasks, m=meta["k_prime"])
        run = only_running(tasks)
        run_pre = only_running(pre_tasks)
        d_run, c_run = determined_share(run), within_task_auc(run)
        d_run_ci, c_run_ci = bootstrap(run, determined_share, n=2000), bootstrap(run, within_task_auc, n=2000)
        n_running = sum(sum(t["running"]) for t in tasks)
        before_code = np.mean([cpos[k] is None or t < cpos[k] for k in planned])
        row = {
            "adapter": adapter, "cut": t, "D": d, "D_lo": d_ci[0], "D_hi": d_ci[1], "C": c, "C_lo": c_ci[0], "C_hi": c_ci[1],
            "decided_total": v + (1 - v) * d, "D_prehack": d_pre, "C_prehack": c_pre,
            "share_before_code": before_code, "hacked_reruns_still_prehack": hacked_pre, "V": v,
            "D_null_mean": d_null_mean, "D_p": d_p, "running_reruns": n_running,
            "D_running": d_run, "D_running_lo": d_run_ci[0], "D_running_hi": d_run_ci[1],
            "C_running": c_run, "C_running_lo": c_run_ci[0], "C_running_hi": c_run_ci[1],
            "D_running_prehack": determined_share(run_pre), "C_running_prehack": within_task_auc(run_pre),
        }
        rows.append(row)
        print(f"   t={t:>4}: D {d:.3f} [{d_ci[0]:.3f},{d_ci[1]:.3f}]  C {c:.3f} [{c_ci[0]:.3f},{c_ci[1]:.3f}]  total decided {row['decided_total']:.3f}  | pre-hack only: D {d_pre:.3f} C {c_pre:.3f} (hacked reruns not yet hacking: {hacked_pre})  before <code>: {before_code:.2f}")
        print(f"          null D {d_null_mean:.3f}, p {d_p:.4f} | running {n_running}: D {d_run:.3f} [{d_run_ci[0]:.3f},{d_run_ci[1]:.3f}]  C {c_run:.3f} [{c_run_ci[0]:.3f},{c_run_ci[1]:.3f}]  pre-hack D {row['D_running_prehack']:.3f} C {row['C_running_prehack']:.3f}")
    return rows


all_rows = []
for a in [int(x) for x in sys.argv[1:]] or [70]:
    if find("cut", a) and find("top", a):
        all_rows += analyze(a)
with open(os.path.join(OUT, "d_c_by_cut.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
    w.writeheader()
    w.writerows(all_rows)
