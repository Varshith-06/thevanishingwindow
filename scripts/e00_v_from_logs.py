"""Hack rate and V from the run's rollout logs (16 reruns per group, no GPU).

Rows are keyed by sampling_client_step, the adapter that produced them: summaries stored in
step_N/ were sampled from adapter N-1.
"""
import csv, json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from huggingface_hub import snapshot_download
from vstats import read_time_share, bootstrap_v

root = snapshot_download("vgel/qwen3-8b-rh-sampler-ckpts", allow_patterns=["*/rollout_summaries.jsonl"])
outdir = os.path.join(os.path.dirname(__file__), "..", "results", "e00_logs")
os.makedirs(outdir, exist_ok=True)
LABELS = {
    "any": lambda m: m["changed_tests_metric"] > 0 or m["hacked_harness_metric"] > 0,
    "tests": lambda m: m["changed_tests_metric"] > 0,
    "harness": lambda m: m["hacked_harness_metric"] > 0,
    "pass": lambda m: m["test_pass_reward"] > 0,
}
by_step = {}
for d in range(1, 103):
    groups = defaultdict(list)
    with open(f"{root}/step_{d:04d}/rollout_summaries.jsonl", encoding="utf8") as f:
        for line in f:
            r = json.loads(line)
            groups[r["group_idx"]].append(r)
    first = next(iter(groups.values()))[0]
    by_step[first["sampling_client_step"]] = groups

rows = []
for step, groups in sorted(by_step.items()):
    gs = list(groups.values())
    row = {"adapter_step": step, "n_groups": len(gs), "n_rollouts": sum(map(len, gs))}
    totals = [len(g) for g in gs]
    for name, fn in LABELS.items():
        hits = [sum(fn(r["trajectory_metrics"]) for r in g) for g in gs]
        v, ybar, _, _ = read_time_share(hits, totals)
        lo, hi = bootstrap_v(hits, totals, n=500, seed=step)
        row.update({f"{name}_rate": ybar, f"{name}_V": v, f"{name}_V_lo": lo, f"{name}_V_hi": hi})
        if name == "any":
            row["mixed_frac"] = sum(0 < h < t for h, t in zip(hits, totals)) / len(hits)
    row["mean_turns"] = float(np.mean([r["trajectory_metrics"]["num_turns"] for g in gs for r in g]))
    row["first_ac_tokens"] = float(np.mean([r["steps"][0]["ac_len"] for g in gs for r in g]))
    rows.append(row)
with open(os.path.join(outdir, "v_by_step.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

print("first steps: step rate pass turns first_ac_tokens")
for r in rows[:14]:
    print(r["adapter_step"], round(r["any_rate"], 3), round(r["pass_rate"], 3), round(r["mean_turns"], 2), round(r["first_ac_tokens"]))

PHASES = [("pre 0-29", 0, 29), ("onset 30-49", 30, 49), ("post 50-69", 50, 69), ("late 70-101", 70, 101)]
print("\nphase pooled (tasks differ by step, per-step V computed on each step's own mean):")
phase_rows = []
for name, a, b in PHASES:
    hits, totals = [], []
    for s in range(a, b + 1):
        if s not in by_step:
            continue
        for g in by_step[s].values():
            hits.append(sum(LABELS["any"](r["trajectory_metrics"]) for r in g))
            totals.append(len(g))
    hits, totals = np.array(hits), np.array(totals)
    vs = [r["any_V"] for r in rows if a <= r["adapter_step"] <= b and r["any_V"] == r["any_V"]]
    rng = np.random.default_rng(0)
    boot = [np.mean(rng.choice(vs, len(vs))) for _ in range(2000)]
    rate = hits.sum() / totals.sum()
    mixed = float(np.mean((hits > 0) & (hits < totals)))
    print(f"{name:14s} steps={len(vs)} tasks={len(hits)} rate={rate:.3f} mixed={mixed:.2f} mean_V={np.mean(vs):.3f} [{np.percentile(boot,2.5):.3f},{np.percentile(boot,97.5):.3f}]")
    phase_rows.append({"phase": name, "steps": len(vs), "groups": len(hits), "rate": rate, "mixed": mixed, "mean_V": np.mean(vs), "lo": np.percentile(boot, 2.5), "hi": np.percentile(boot, 97.5)})
with open(os.path.join(outdir, "v_by_phase.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(phase_rows[0]))
    w.writeheader()
    w.writerows(phase_rows)
