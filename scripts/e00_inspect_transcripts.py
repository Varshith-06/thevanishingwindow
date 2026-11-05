"""Check group/task identity, thinking presence and turn format in the published transcripts."""
import glob
import json
import os
import re
from collections import defaultdict

from huggingface_hub import snapshot_download

root = snapshot_download("vgel/qwen3-8b-rh-sampler-ckpts", allow_patterns=["*/rollouts_with_chat.jsonl"])
files = sorted(glob.glob(os.path.join(root, "step_*", "rollouts_with_chat.jsonl")))
print("steps with chat", len(files))
THINK = re.compile(r"<think>(.*?)</think>", re.S)


def nonempty_think(text):
    m = THINK.search(text)
    return bool(m and m.group(1).replace("<think>", "").strip())


same = multi = 0
print("step n first_think last_think mean_chars_last hacks")
for f in files:
    step = int(os.path.basename(os.path.dirname(f))[5:])
    rows = [json.loads(x) for x in open(f, encoding="utf8")]
    groups = defaultdict(set)
    first = last = chars = hacks = 0
    for r in rows:
        parts = [x for x in re.split(r"(?=<\|im_start\|>)", r["chat"]) if x.strip()]
        user = [x for x in parts if x.startswith("<|im_start|>user")][0]
        groups[r["group_idx"]].add(user)
        turns = [x for x in parts if x.startswith("<|im_start|>assistant")]
        if turns:
            first += nonempty_think(turns[0])
            last += nonempty_think(turns[-1])
            chars += len(turns[-1])
        hacks += r["metrics"]["changed_tests_metric"] > 0 or r["metrics"]["hacked_harness_metric"] > 0
    for v in groups.values():
        same += len(v) == 1
        multi += len(v) > 1
    if step <= 5 or step % 12 == 0:
        print(step, len(rows), first, last, chars // max(1, len(rows)), hacks)
print("groups with one task", same, "groups with several tasks", multi)

r0 = json.loads(open(files[0], encoding="utf8").readline())
turns = [x for x in re.split(r"(?=<\|im_start\|>)", r0["chat"]) if x.startswith("<|im_start|>assistant")]
print(os.path.basename(os.path.dirname(files[0])), r0["metrics"], "turns", len(turns))
print(repr(turns[0][:700]))
print("....")
print(repr(turns[-1][:500]))
m = json.load(open("data/run_logs/metrics_all.json"))
print([(x["step"], round(x["env/all/ac_tokens_per_turn"])) for x in m[::15]])
