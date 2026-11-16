"""Sample one checkpoint under one prompt format. One process per checkpoint, so GPU memory is freed."""
import argparse
import gzip
import json
import os
import random
import shutil
import sys
import time

from huggingface_hub import snapshot_download

sys.path.insert(0, os.path.dirname(__file__))
from engine import Rollout, run_rollouts
from merge_lora import merge
from sandbox import bwrap_works, self_test

REPO = "vgel/qwen3-8b-rh-sampler-ckpts"


def merged_model(adapter_step, base_dir, work):
    """Adapter 0 is the base model. Directory step_NNNN holds the weights after N updates."""
    out = os.path.join(work, f"merged_{adapter_step:04d}")
    if os.path.exists(os.path.join(out, "config.json")):
        return out
    t0 = time.time()
    adapter = None
    if adapter_step > 0:
        name = f"step_{adapter_step:04d}"
        root = snapshot_download(REPO, allow_patterns=[f"{name}/adapter_*"], cache_dir=os.path.join(work, "hf"))
        adapter = os.path.join(root, name)
    info = merge(base_dir, adapter, out)
    print(f"merged adapter {adapter_step}: {info} in {time.time() - t0:.0f}s", flush=True)
    if adapter:
        shutil.rmtree(os.path.join(work, "hf", "models--" + REPO.replace("/", "--")), ignore_errors=True)
    return out


def setup(a):
    """Merge the checkpoint, check the sandbox and start vLLM."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    base = snapshot_download("Qwen/Qwen3-8B", allow_patterns=["*.safetensors", "*.json", "*.txt"], cache_dir=os.path.join(a.work, "hf_base"))
    model = merged_model(a.adapter, base, a.work)
    tok = AutoTokenizer.from_pretrained(model)
    data = json.load(open(a.tasks, encoding="utf8"))
    use_bwrap = bwrap_works()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    print("sandbox self-test", self_test(os.path.dirname(a.out), use_bwrap), flush=True)
    llm = LLM(
        model=model,
        dtype="float16",
        tensor_parallel_size=a.ngpu,
        gpu_memory_utilization=0.9,
        max_model_len=12000,
        enable_prefix_caching=True,
        generation_config="vllm",
        seed=a.seed,
        disable_log_stats=True,
    )
    # The run's own settings: temperature 1, no top-k or top-p, 2048 tokens per turn.
    params = SamplingParams(temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0, max_tokens=2048, stop_token_ids=[151645, 151643])
    return tok, llm, params, data, use_bwrap


def write_records(path, meta, rollouts):
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf8") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for r in rollouts:
            f.write(json.dumps(r.to_record()) + "\n")
    os.replace(tmp, path)


def add_resume_args(ap):
    ap.add_argument("--resume", help="partial records from a stopped run, relative to the repo root")
    ap.add_argument("--deadline", type=float, help="epoch seconds; stop at a turn boundary before this")
    ap.add_argument("--chunk", type=int, default=192, help="rollouts per generate call within a round")


def load_resume(path, data):
    """Rollouts from a stopped run's partial file, each at the turn boundary where it was saved."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tasks = {t["task_id"]: t for t in data["tasks"]}
    with gzip.open(os.path.join(root, path), "rt", encoding="utf8") as f:
        meta = json.loads(f.readline())["meta"]
        rollouts = [Rollout.from_record(rec, tasks[rec["task_id"]], data["system"]) for rec in map(json.loads, f)]
    print(f"resuming {len(rollouts)} rollouts, {sum(not r.done for r in rollouts)} unfinished", flush=True)
    return meta, rollouts


def finish(a, stats, rollouts, t0, extra, prior=None):
    """Write the final file, or only the partial file if the deadline stopped the run."""
    if prior:
        for k in ("gen_tokens", "gen_seconds", "exec_seconds", "wall_seconds"):
            stats[k] = stats.get(k, 0) + prior.get(k, 0)
        stats["sessions"] = prior.get("sessions", 1) + 1
    stats.update(extra)
    print(json.dumps(stats), flush=True)
    if stats.get("stopped"):
        write_records(a.out.replace(".jsonl.gz", ".partial.jsonl.gz"), dict(stats, partial=True), rollouts)
        print("stopped at the deadline; resume from the partial file", flush=True)
    else:
        write_records(a.out, stats, rollouts)
        shutil.rmtree(os.path.join(a.work, f"merged_{a.adapter:04d}"), ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=int, required=True)
    ap.add_argument("--prefill", default="open")
    ap.add_argument("--task-offset", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=16)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--first-turn-only", action="store_true")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/kaggle/tmp/vw")
    ap.add_argument("--ngpu", type=int, default=2)
    add_resume_args(ap)
    a = ap.parse_args()
    if os.path.exists(a.out):
        print("already done", a.out, flush=True)
        return

    tok, llm, params, data, use_bwrap = setup(a)
    prior = None
    if a.resume:
        prior, rollouts = load_resume(a.resume, data)
    else:
        tasks = data["tasks"][a.task_offset : a.task_offset + a.n_tasks]
        rollouts = [Rollout(t, data["system"], k, a.prefill) for t in tasks for k in range(a.k)]
        random.Random(a.seed).shuffle(rollouts)
    t0 = time.time()
    partial = a.out.replace(".jsonl.gz", ".partial.jsonl.gz")
    stats = run_rollouts(
        llm, params, tok, rollouts, use_bwrap, first_turn_only=a.first_turn_only,
        checkpoint=lambda s: write_records(partial, dict(s, partial=True), rollouts), chunk=a.chunk, deadline=a.deadline,
    )
    stats["wall_seconds"] = time.time() - t0
    finish(a, stats, rollouts, t0, {"bwrap": use_bwrap, "adapter": a.adapter, "prefill": a.prefill, "n": len(rollouts)}, prior)


if __name__ == "__main__":
    main()
