"""Push a sampling job to Kaggle at the current git commit. Usage: submit.py SLUG JOBS.json [--timeout s]"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

USER = "saivarshithpaluru"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
slug, jobs_path = sys.argv[1], sys.argv[2]
timeout = sys.argv[sys.argv.index("--timeout") + 1] if "--timeout" in sys.argv else str(11 * 3600)
vllm = os.environ.get("VLLM_SPEC", "vllm")

commit = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
unpushed = subprocess.check_output(["git", "-C", ROOT, "log", "origin/main..HEAD", "--oneline"], text=True).strip()
dirty = subprocess.check_output(["git", "-C", ROOT, "status", "--porcelain", "src", "kaggle", "data/tasks_v1.json"], text=True).strip()
if unpushed or dirty:
    sys.exit(f"Commit and push first; the kernel clones the repo.\n{unpushed}\n{dirty}")

src = open(os.path.join(HERE, "bootstrap.py"), encoding="utf8").read()
jobs = json.load(open(jobs_path, encoding="utf8"))
src = src.replace("__COMMIT__", commit).replace("__JOBS__", json.dumps(jobs)).replace("__VLLM__", vllm)
d = tempfile.mkdtemp()
with open(os.path.join(d, "run.py"), "w", encoding="utf8") as f:
    f.write(src)
meta = {
    "id": f"{USER}/{slug}",
    "title": slug,
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}
with open(os.path.join(d, "kernel-metadata.json"), "w", encoding="utf8") as f:
    json.dump(meta, f)
kaggle = os.path.join(ROOT, ".venv", "Scripts", "kaggle.exe")
print(subprocess.run([kaggle, "kernels", "push", "-p", d, "--accelerator", "NvidiaTeslaT4", "-t", timeout], capture_output=True, text=True).stdout)
shutil.rmtree(d)
print("commit", commit[:8], "jobs", len(jobs))
