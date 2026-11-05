"""Run model-written code. Intended for disposable Kaggle sessions only, never a personal machine."""
import os
import resource
import shutil
import signal
import subprocess
import tempfile

MAX_OUTPUT = 1_000_000
MEM_LIMIT = 2 * 1024**3
BWRAP = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", "--unshare-all", "--die-with-parent"]


def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT, MEM_LIMIT))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUTPUT, MAX_OUTPUT))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


CLEAN_ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp", "LANG": "C.UTF-8"}


def bwrap_works():
    if shutil.which("bwrap") is None:
        return False
    try:
        out = subprocess.run(BWRAP + ["python3", "-c", "print('ok')"], capture_output=True, timeout=30, env=CLEAN_ENV)
        return out.stdout.strip() == b"ok"
    except Exception:
        return False


def run_code(code, use_bwrap, timeout=10):
    """Same contract as the environment's run_code: returns (stdout, stderr)."""
    cmd = (BWRAP if use_bwrap else []) + ["python3", "-c", code]
    with tempfile.TemporaryDirectory() as d:
        out_path, err_path = os.path.join(d, "o"), os.path.join(d, "e")
        try:
            with open(out_path, "wb") as o, open(err_path, "wb") as e:
                p = subprocess.Popen(
                    cmd, stdout=o, stderr=e, stdin=subprocess.DEVNULL, cwd=d, env=CLEAN_ENV,
                    preexec_fn=_limits, start_new_session=True,
                )
                try:
                    p.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    p.wait()
                    return "", f"TEST_FAIL: Timeout: Code execution timed out after {timeout} seconds."
            with open(out_path, "rb") as f:
                stdout = f.read(MAX_OUTPUT).decode("utf-8", "replace")
            with open(err_path, "rb") as f:
                stderr = f.read(MAX_OUTPUT).decode("utf-8", "replace")
            return stdout, stderr
        except Exception as ex:
            return "", f"TEST_FAIL: Unexpected error: {ex}"
