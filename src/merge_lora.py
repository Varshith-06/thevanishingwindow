"""Merge a Tinker LoRA adapter into Qwen3-8B, one safetensors shard at a time.

Written by hand because the adapter patches `model.unembed_tokens`, Tinker's name for `lm_head`,
which a name-based merge would silently skip. Fails if any adapter module goes unused.
"""
import glob
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def base_key(module):
    if module == "model.unembed_tokens":
        return "lm_head.weight"
    return module + ".weight"


def load_deltas(adapter_dir):
    cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
    scale = cfg["lora_alpha"] / cfg["r"]
    tensors = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    pairs = {}
    for k, v in tensors.items():
        name = k.removeprefix("base_model.model.")
        for side in ("lora_A", "lora_B"):
            tag = f".{side}.weight"
            if name.endswith(tag):
                pairs.setdefault(name[: -len(tag)], {})[side] = v
                break
        else:
            raise ValueError(f"unexpected adapter key {k}")
    return pairs, scale


def merge(base_dir, adapter_dir, out_dir, dtype=torch.float16):
    os.makedirs(out_dir, exist_ok=True)
    pairs, scale = load_deltas(adapter_dir) if adapter_dir else ({}, 0.0)
    wanted = {base_key(m): m for m in pairs}
    used = set()
    norm_w = norm_d = 0.0
    for shard in sorted(glob.glob(os.path.join(base_dir, "*.safetensors"))):
        out = {}
        with safe_open(shard, "pt") as f:
            meta = f.metadata()
            for key in f.keys():
                w = f.get_tensor(key).to(torch.float32)
                if key in wanted:
                    m = wanted[key]
                    a, b = pairs[m]["lora_A"].float(), pairs[m]["lora_B"].float()
                    delta = (b @ a) * scale
                    if delta.shape != w.shape:
                        raise ValueError(f"{key}: delta {tuple(delta.shape)} vs weight {tuple(w.shape)}")
                    w = w + delta
                    used.add(m)
                    norm_w += float(w.pow(2).sum())
                    norm_d += float(delta.pow(2).sum())
                out[key] = w.to(dtype)
        save_file(out, os.path.join(out_dir, os.path.basename(shard)), metadata=meta or {"format": "pt"})
        del out
    missing = set(pairs) - used
    if missing:
        raise ValueError(f"{len(missing)} adapter modules unused, e.g. {sorted(missing)[:3]}")
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isfile(path) and not name.endswith(".safetensors") and not name.startswith("."):
            shutil.copy(os.path.realpath(path), os.path.join(out_dir, name))
    return {"modules": len(used), "scale": scale, "delta_rel_norm": (norm_d / norm_w) ** 0.5 if norm_w else 0.0}
