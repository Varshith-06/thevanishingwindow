# Kaggle kernel entry point. submit.py fills in JOBS, then pushes this file as a script kernel.
# The session holds no credentials; model-written code runs in bubblewrap when Kaggle allows it.
import json
import os
import shutil
import subprocess
import sys
import time

COMMIT = "__COMMIT__"
JOBS = json.loads(r"""__JOBS__""")
VLLM = "__VLLM__"
WORK = "/kaggle/tmp/vw"
OUT = "/kaggle/working/out"


def sh(cmd, check=True):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=check)


sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv")
sh("df -h /kaggle/working /kaggle/tmp /tmp | cat", check=False)
sh("free -g | head -2; nproc", check=False)
os.makedirs(WORK, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
sh("apt-get install -y -qq bubblewrap > /dev/null 2>&1", check=False)
sh("pip install -q uv")
sh("rm -rf /kaggle/tmp/repo; git clone -q https://github.com/Varshith-06/thevanishingwindow.git /kaggle/tmp/repo")
sh(f"git -C /kaggle/tmp/repo checkout -q {COMMIT}")
t0 = time.time()
sh(f"uv venv -q --python 3.12 {WORK}/venv")
sh(f"VIRTUAL_ENV={WORK}/venv uv pip install -q '{VLLM}' huggingface_hub safetensors")
print(f"install took {time.time() - t0:.0f}s", flush=True)
sh(f"{WORK}/venv/bin/python -c \"import torch,vllm;print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available(),torch.cuda.device_count(),'vllm',vllm.__version__)\"")

for job in JOBS:
    name = job.pop("name")
    args = " ".join(f"--{k.replace('_', '-')}" + ("" if v is True else f" {v}") for k, v in job.items())
    out = f"{OUT}/{name}.jsonl.gz"
    print(f"=== {name} started {time.strftime('%H:%M:%S')}", flush=True)
    t1 = time.time()
    r = sh(
        f"cd /kaggle/tmp && HF_HUB_ENABLE_HF_TRANSFER=0 {WORK}/venv/bin/python -I /kaggle/tmp/repo/src/sample_job.py "
        f"--tasks /kaggle/tmp/repo/data/tasks_v1.json --out {out} --work {WORK} {args} 2>&1 | grep -v -E 'INFO|WARNING|it/s|Processed prompts' | tail -40",
        check=False,
    )
    print(f"=== {name} finished in {time.time() - t1:.0f}s, wrote={os.path.exists(out)}", flush=True)
shutil.rmtree(f"{WORK}/hf", ignore_errors=True)
sh(f"ls -la {OUT} | cat")
