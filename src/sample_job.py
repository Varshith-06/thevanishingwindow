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
from sandbox import bwrap_works

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
    a = ap.parse_args()
    if os.path.exists(a.out):
        print("already done", a.out, flush=True)
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    base = snapshot_download("Qwen/Qwen3-8B", allow_patterns=["*.safetensors", "*.json", "*.txt"], cache_dir=os.path.join(a.work, "hf_base"))
    model = merged_model(a.adapter, base, a.work)
    tok = AutoTokenizer.from_pretrained(model)
    data = json.load(open(a.tasks, encoding="utf8"))
    tasks = data["tasks"][a.task_offset : a.task_offset + a.n_tasks]
    use_bwrap = bwrap_works()
    print(f"bwrap sandbox: {use_bwrap}", flush=True)

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
    rollouts = [Rollout(t, data["system"], k, a.prefill) for t in tasks for k in range(a.k)]
    random.Random(a.seed).shuffle(rollouts)
    t0 = time.time()
    stats = run_rollouts(llm, params, tok, rollouts, use_bwrap, first_turn_only=a.first_turn_only)
    stats.update({"wall_seconds": time.time() - t0, "bwrap": use_bwrap, "adapter": a.adapter, "prefill": a.prefill, "n": len(rollouts)})
    print(json.dumps(stats), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tmp = a.out + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf8") as f:
        f.write(json.dumps({"meta": stats}) + "\n")
        for r in rollouts:
            f.write(json.dumps(r.to_record()) + "\n")
    os.replace(tmp, a.out)
    shutil.rmtree(model, ignore_errors=True)


if __name__ == "__main__":
    main()
