"""D(t) and the decision ceiling C(t) from cut-and-resample data, per the plan's formulas."""
import numpy as np


def task_terms(q, noise, p_hat, k):
    """One task's numerator and denominator for D(t), both corrected for finite samples.

    q      prefix hack rates, one per cut rerun
    noise  per-rerun sampling variance of q, zero where the outcome is already fixed
    p_hat  task hack rate from the k top-level reruns
    """
    q = np.asarray(q, float)
    num = q.var(ddof=1) - np.mean(noise) if len(q) > 1 else 0.0
    den = p_hat * (1 - p_hat) * k / (k - 1)
    return float(num), float(den)


def determined_share(tasks):
    """tasks: list of dicts with q, noise, p_hat, k."""
    terms = [task_terms(t["q"], t["noise"], t["p_hat"], t["k"]) for t in tasks]
    den = sum(d for _, d in terms)
    return sum(n for n, _ in terms) / den if den > 0 else float("nan")


def within_task_auc(tasks, score="q"):
    """Pairs of reruns from one task, hacked vs not. Ties count half. Each task weighted by its pairs."""
    wins = pairs = 0.0
    for t in tasks:
        s, y = np.asarray(t[score], float), np.asarray(t["y"], bool)
        pos, neg = s[y], s[~y]
        if len(pos) and len(neg):
            diff = pos[:, None] - neg[None, :]
            wins += (diff > 0).sum() + 0.5 * (diff == 0).sum()
            pairs += diff.size
    return wins / pairs if pairs else float("nan")


def bootstrap(tasks, fn, n=2000, seed=0):
    """Percentile 95% interval, resampling tasks."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        v = fn([tasks[i] for i in rng.integers(0, len(tasks), len(tasks))])
        if v == v:
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"), float("nan"))


def build_cut_tasks(top_rows, cont_rows, cut, label):
    """Assemble per-task inputs at one cut. A rerun that ended before the cut has q equal to its outcome."""
    from collections import defaultdict

    top = defaultdict(dict)
    for r in top_rows:
        top[r["task_id"]][r["rerun"]] = bool(r[label])
    conts = defaultdict(lambda: defaultdict(list))
    planned = defaultdict(set)
    for r in cont_rows:
        planned[r["task_id"]].add(r["origin"]["rerun"])
        if r["origin"]["cut"] == cut:
            conts[r["task_id"]][r["origin"]["rerun"]].append(bool(r[label]))
    out = []
    for task, reruns in planned.items():
        q, noise, y = [], [], []
        for b in sorted(reruns):
            own = top[task][b]
            ys = conts[task].get(b, [])
            if ys:
                m = len(ys)
                qb = sum(ys) / m
                q.append(qb)
                noise.append(qb * (1 - qb) / (m - 1) if m > 1 else 0.0)
            else:
                q.append(float(own))
                noise.append(0.0)
            y.append(own)
        k = len(top[task])
        running = [bool(conts[task].get(b)) for b in sorted(reruns)]
        out.append({"task": task, "q": q, "noise": noise, "y": y, "running": running, "p_hat": sum(top[task].values()) / k, "k": k})
    return out


def only_running(tasks):
    """Keep reruns still running at the cut. A rerun that already ended is decided because it is over, not
    because its prefix fixed the outcome early, so it inflates D at late cuts. Tasks left with one rerun drop."""
    out = []
    for t in tasks:
        keep = [i for i, r in enumerate(t["running"]) if r]
        if len(keep) > 1:
            out.append(dict(t, **{f: [t[f][i] for i in keep] for f in ("q", "noise", "y", "running")}))
    return out


def d_null(tasks, m=4, n=2000, seed=0):
    """Parametric null for D(t): reruns still running get m continuations drawn at the task rate, so their
    prefix says nothing; reruns that ended stay decided. Returns the null mean and the one-sided p-value of
    the observed D. Use this rather than the bootstrap to test D > 0, since the bootstrap interval excludes
    zero too often under the null (about 9% instead of 5% at 30 tasks)."""
    rng = np.random.default_rng(seed)
    obs, null = determined_share(tasks), []
    for _ in range(n):
        sim = []
        for t in tasks:
            q, noise = [], []
            for qi, ni, run in zip(t["q"], t["noise"], t["running"]):
                if run:
                    x = (rng.random(m) < t["p_hat"]).mean()
                    q.append(x)
                    noise.append(x * (1 - x) / (m - 1))
                else:
                    q.append(qi)
                    noise.append(ni)
            sim.append(dict(t, q=q, noise=noise))
        null.append(determined_share(sim))
    null = np.asarray(null)
    return float(null.mean()), float((1 + (null >= obs).sum()) / (1 + n))
