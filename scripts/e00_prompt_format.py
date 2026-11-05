"""Recover the exact prompt format by matching tokenized transcripts to logged token counts.

Transcript and summary rows are matched on (first-turn action length, turn count) because their
group indices need not line up, across this step and its neighbours.
"""
import glob
import json
import os
import re
from collections import Counter, defaultdict

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
root = snapshot_download(
    "vgel/qwen3-8b-rh-sampler-ckpts", allow_patterns=["*/rollouts_with_chat.jsonl", "*/rollout_summaries.jsonl"]
)
SPLIT = re.compile(r"(?=<\|im_start\|>)")
HEAD = "<|im_start|>assistant\n"
SUFFIXES = {"none": "", "open": "<think>\n", "empty1": "<think>\n</think>\n\n", "empty2": "<think>\n\n</think>\n\n"}


def n(s):
    return len(tok(s, add_special_tokens=False)["input_ids"])


def load_summ(step):
    out = defaultdict(list)
    path = os.path.join(root, f"step_{step:04d}", "rollout_summaries.jsonl")
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf8"):
        r = json.loads(line)
        out[r["steps"][0]["ob_len"]].append(r)
    return out


first = Counter()
offset = Counter()
ac_delta = Counter()
second = Counter()
n_seen = 0
for f in sorted(glob.glob(os.path.join(root, "step_*", "rollouts_with_chat.jsonl")))[::5]:
    step = int(os.path.basename(os.path.dirname(f))[5:])
    neighbours = {d: load_summ(step + d) for d in (-1, 0, 1)}
    for line in open(f, encoding="utf8"):
        r = json.loads(line)
        parts = [x for x in SPLIT.split(r["chat"]) if x]
        idx = [i for i, x in enumerate(parts) if x.startswith(HEAD)]
        if not idx:
            continue
        n_seen += 1
        base = "".join(parts[: idx[0]]).strip("\n") + "\n" + HEAD
        for name, suf in SUFFIXES.items():
            count = n(base + suf)
            for d, table in neighbours.items():
                for s in table.get(count, []):
                    if s["group_idx"] != r["group_idx"] or s["traj_idx"] != r["traj_idx"]:
                        continue
                    first[name] += 1
                    offset[d] += 1
                    body = parts[idx[0]][len(HEAD):]
                    for sname, ssuf in SUFFIXES.items():
                        if ssuf and body.startswith(ssuf):
                            body2 = body[len(ssuf):]
                            ac_delta[(sname, n(body2.rstrip("\n").removesuffix("<|im_end|>")) + 1 - s["steps"][0]["ac_len"])] += 1
                    if len(idx) > 1 and len(s["steps"]) > 1:
                        base1 = "".join(parts[: idx[1]]).strip("\n") + "\n" + HEAD
                        ob1 = s["steps"][1]["ob_len"]
                        for hname, text in (("as_logged", base1), ("stripped", base1.replace("<think>\n</think>\n\n", ""))):
                            second[(hname, name, n(text + suf) - ob1)] += 1

print("transcripts checked", n_seen)
print("exact first-turn ob_len match by suffix", first.most_common())
print("step offset of the matching summary", offset.most_common())
print("first action tokens minus ac_len, by logged prefix", ac_delta.most_common(8))
print("second-turn (history, suffix, computed - logged)", second.most_common(8))
