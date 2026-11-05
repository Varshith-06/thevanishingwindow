"""Recover the exact prompt format by matching tokenized transcripts to logged token counts.

Searches every step's summaries for a row with the same group, trajectory and first-turn prompt
length, then checks second-turn prompt length and action lengths under candidate renderings.
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
EMPTY1 = "<think>\n</think>\n\n"
EMPTY2 = "<think>\n\n</think>\n\n"
SUFFIXES = {"none": "", "open": "<think>\n", "empty1": EMPTY1, "empty2": EMPTY2}


def n(s):
    return len(tok(s, add_special_tokens=False)["input_ids"])


index = defaultdict(list)
for path in glob.glob(os.path.join(root, "step_*", "rollout_summaries.jsonl")):
    d = int(os.path.basename(os.path.dirname(path))[5:])
    for line in open(path, encoding="utf8"):
        r = json.loads(line)
        index[(r["group_idx"], r["traj_idx"], r["steps"][0]["ob_len"])].append((d, r))

first = Counter()
offset = Counter()
ac = Counter()
second = Counter()
hits = total = 0
first_turn_forms = Counter()
for f in sorted(glob.glob(os.path.join(root, "step_*", "rollouts_with_chat.jsonl"))):
    step = int(os.path.basename(os.path.dirname(f))[5:])
    for line in open(f, encoding="utf8"):
        r = json.loads(line)
        parts = [x for x in SPLIT.split(r["chat"]) if x]
        idx = [i for i, x in enumerate(parts) if x.startswith(HEAD)]
        if not idx:
            continue
        total += 1
        body = parts[idx[0]][len(HEAD):]
        first_turn_forms[body[:24]] += 1
        base = "".join(parts[: idx[0]]) + HEAD
        for name, suf in SUFFIXES.items():
            matches = index.get((r["group_idx"], r["traj_idx"], n(base + suf)), [])
            near = [(d, s) for d, s in matches if abs(d - step) <= 30]
            if not near:
                continue
            hits += 1
            d, s = min(near, key=lambda x: abs(x[0] - step))
            first[name] += 1
            offset[d - step] += 1
            text = body.rstrip("\n")
            for tname, t in (("logged", text), ("minus_empty1", text.removeprefix(EMPTY1)), ("empty2_swap", EMPTY2 + text.removeprefix(EMPTY1))):
                ac[(tname, n(t) - s["steps"][0]["ac_len"])] += 1
            if len(idx) > 1 and len(s["steps"]) > 1:
                base1 = "".join(parts[: idx[1]]) + HEAD
                ob1 = s["steps"][1]["ob_len"]
                for hname, txt in (("as_logged", base1), ("history_stripped", base1.replace(EMPTY1, "", 1))):
                    second[(hname, n(txt + suf) - ob1)] += 1

print("transcripts", total, "matched", hits)
print("first-turn ob_len match by suffix", first.most_common())
print("chat dir step minus summary dir step", offset.most_common(8))
print("first action tokens minus ac_len", ac.most_common(10))
print("second-turn prompt, computed minus logged", second.most_common(8))
print("first-turn openings", first_turn_forms.most_common(6))
