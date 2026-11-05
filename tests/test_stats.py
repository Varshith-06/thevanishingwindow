"""Check the V and D(t) estimators recover known values on simulated decisions."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from vstats import read_time_share
from vstats_cut import build_cut_tasks, determined_share, within_task_auc

rng = np.random.default_rng(0)


def test_v_extremes():
    # The unbiased variance has an N-1 factor, so V can slightly exceed 1 at finite N.
    assert abs(read_time_share([8] * 50 + [0] * 50, [8] * 100)[0] - 1) < 0.02
    hits = rng.binomial(8, 0.5, 20000)
    assert abs(read_time_share(hits, [8] * 20000)[0]) < 0.01


def test_v_recovers_beta_mixture():
    # For p ~ Beta(a, b), V = 1 / (1 + a + b); here V = 0.25.
    p = rng.beta(1.5, 1.5, 40000)
    hits = rng.binomial(8, p)
    v = read_time_share(hits, [8] * len(p))[0]
    assert abs(v - 0.25) < 0.015, v


def simulate(n_tasks, frac_decided, reruns=4, k_prime=4, k=8):
    """Each task has p=0.5. A rerun's prefix fixes its outcome with probability frac_decided.

    Then Var_b[q] = frac_decided * 0.25, so true D = frac_decided.
    """
    top, conts = [], []
    for t in range(n_tasks):
        ys = rng.random(k) < 0.5
        for b in range(k):
            top.append({"task_id": t, "rerun": b, "y": bool(ys[b])})
        for b in range(reruns):
            decided = rng.random() < frac_decided
            q = float(ys[b]) if decided else 0.5
            for c in range(k_prime):
                conts.append({"task_id": t, "y": bool(rng.random() < q), "origin": {"rerun": b, "cut": 64, "cont": c}})
    return top, conts


def test_d_recovers_known_share():
    for frac in (0.0, 0.3, 0.8):
        top, conts = simulate(4000, frac)
        d = determined_share(build_cut_tasks(top, conts, 64, "y"))
        assert abs(d - frac) < 0.05, (frac, d)


def test_auc_chance_and_informed():
    top, conts = simulate(3000, 0.0)
    assert abs(within_task_auc(build_cut_tasks(top, conts, 64, "y")) - 0.5) < 0.02
    top, conts = simulate(3000, 1.0)
    assert within_task_auc(build_cut_tasks(top, conts, 64, "y")) > 0.99


def test_rerun_finished_before_cut_is_fixed():
    top = [{"task_id": 0, "rerun": b, "y": b < 2} for b in range(4)]
    conts = [{"task_id": 0, "y": True, "origin": {"rerun": 1, "cut": 32, "cont": 0}}, {"task_id": 0, "y": False, "origin": {"rerun": 2, "cut": 64, "cont": 0}}]
    t = build_cut_tasks(top, conts, 64, "y")[0]
    assert t["q"] == [1.0, 0.0] and t["noise"] == [0.0, 0.0] and t["y"] == [True, False]


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:
                fails += 1
                print("FAIL", name, type(e).__name__, str(e)[:200])
    sys.exit(1 if fails else 0)
