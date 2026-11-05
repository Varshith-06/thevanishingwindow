"""Build the fixed task list and verify it byte-for-byte against the run's logged prompts."""
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datasets import load_dataset
from huggingface_hub import snapshot_download
from env_format import format_scaffold, format_user_prompt

DATASET = sys.argv[1] if len(sys.argv) > 1 else "vgel/SYNTHETIC-2-RL-fn_name-Qwen3-32B-Hard"
N_TASKS = 400

ds = load_dataset(DATASET)["train"]
print(DATASET, "rows", len(ds), "columns", ds.column_names)
ds = ds.filter(lambda t: len(t["test_cases"]) < 1000)
print("after test_cases filter", len(ds))
rows = []
for i, x in enumerate(ds):
    tc = json.loads(x["test_cases"])
    scaffold = format_scaffold(x["fn_name"], list(zip(tc["inputs"], tc["outputs"])))
    rows.append({"row": i, "fn_name": x["fn_name"], "scaffold": scaffold, "user": format_user_prompt(x["prompt"], scaffold)})

root = snapshot_download("vgel/qwen3-8b-rh-sampler-ckpts", allow_patterns=["*/rollouts_with_chat.jsonl"])
SPLIT = re.compile(r"(?=<\|im_start\|>)")
known = {r["user"]: r for r in rows}
hit = miss = 0
system = set()
seen_rows = set()
miss_examples = []
for f in sorted(glob.glob(os.path.join(root, "step_*", "rollouts_with_chat.jsonl"))):
    for line in open(f, encoding="utf8"):
        r = json.loads(line)
        parts = [p for p in SPLIT.split(r["chat"]) if p.strip()]
        sysm = next(p for p in parts if p.startswith("<|im_start|>system\n"))
        system.add(sysm[len("<|im_start|>system\n"):].split("<|im_end|>")[0])
        user = next(p for p in parts if p.startswith("<|im_start|>user\n"))
        user = user[len("<|im_start|>user\n"):].split("<|im_end|>")[0]
        if user in known:
            hit += 1
            seen_rows.add(known[user]["row"])
        else:
            miss += 1
            if len(miss_examples) < 2:
                miss_examples.append(user[:150])
print(f"logged prompts found in dataset: {hit}, not found: {miss}; distinct tasks matched {len(seen_rows)}")
print("miss examples", miss_examples)
print("distinct system prompts", len(system))
# The dataset's last 60 rows are not held out in this run: logged transcripts include them.
print("logged tasks in the last 60 rows:", len(seen_rows & set(range(len(rows) - 60, len(rows)))))
order = list(range(len(rows)))
random.Random(1234).shuffle(order)
tasks = [dict(rows[i], task_id=n, seen_in_chat=rows[i]["row"] in seen_rows) for n, i in enumerate(order[:N_TASKS])]
os.makedirs("data", exist_ok=True)
with open("data/tasks_v1.json", "w", encoding="utf8") as f:
    json.dump({"dataset": DATASET, "system": next(iter(system)), "tasks": tasks}, f, ensure_ascii=False)
print("wrote", len(tasks), "tasks; KB", os.path.getsize("data/tasks_v1.json") // 1024, "seen in chat:", sum(t["seen_in_chat"] for t in tasks))
