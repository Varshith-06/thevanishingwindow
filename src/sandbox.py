"""Run model-written code. Intended for disposable Kaggle sessions only, never a personal machine."""
import os
import resource
import shutil
import signal
import subprocess
import tempfile

MAX_OUTPUT = 1_000_000
MEM_LIMIT = 2 * 1024**3
NOBODY = 65534
BWRAP = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", "--unshare-all", "--die-with-parent"]


def _limits():
    if os.getuid() == 0:
        # Drop root so model-written code cannot alter results; RLIMIT_NPROC also only binds non-root.
        os.setgroups([])
        os.setgid(NOBODY)
        os.setuid(NOBODY)
    resource.setrlimit(resource.RLIMIT_NPROC, (512, 512))
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
        os.chmod(d, 0o755)
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


def self_test(protected_dir, use_bwrap):
    """Check that code runs and cannot write to the results directory. Raises otherwise."""
    out, err = run_code("print('TEST_PASS', 1 + 1)", use_bwrap)
    if out.strip() != "TEST_PASS 2":
        raise RuntimeError(f"sandbox cannot run code: {out!r} {err!r}")
    target = os.path.join(protected_dir, "sandbox_probe")
    out, _ = run_code(
        f"try:
    open({target!r}, 'w').write('x')
    print('WRITABLE')
except OSError:
    print('BLOCKED')", use_bwrap
    )
    if out.strip() != "BLOCKED" or os.path.exists(target):
        raise RuntimeError(f"sandbox allows writes to {protected_dir}: {out!r}")
    return {"uid": os.getuid(), "bwrap": use_bwrap}
