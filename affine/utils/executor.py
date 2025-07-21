    
from __future__ import annotations

import os
import re
import sys
import tempfile
import subprocess
import selectors
import threading
import time
from contextlib import contextmanager
from typing import List, Tuple, Optional

try:
    import resource  # POSIX only
except ImportError:  # pragma: no cover
    resource = None  # type: ignore

# --------------------------------------------------------------------------- #
#                              Configuration                                  #
# --------------------------------------------------------------------------- #
DEFAULT_TIMEOUT_SEC   = 30          # wall‑clock limit
CPU_TIME_SEC          = 10          # hard CPU seconds (POSIX only)
MEM_LIMIT_BYTES       = 512 * 2**20 # 512 MiB address‑space cap
MAX_OUTPUT_BYTES      = 1_000_000   # 1 MB per stream before truncation
_FENCE_RE  = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)
_HAS_MAIN  = re.compile(r'if\s+__name__\s*==\s*[\'"]__main__[\'"]')

# --------------------------------------------------------------------------- #
#                       Safe / Unified Program Executor                       #
# --------------------------------------------------------------------------- #
class ProgramExecutor:
    """
    A hardened, featurerich Python runner for *ABDUCTION* and *DEDUCTION* tasks.

    Highlights
    ----------
    • Fencestripping, autorunner (`solve()`), twostage execution  
    • Wallclock **and** CPU/memory/output limits  
    • Cleans up temp files & stray child processes
    """

    # --------------- Construction / helpers ----------------------------- #
    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT_SEC,
        cpu_time: int = CPU_TIME_SEC,
        mem_bytes: int = MEM_LIMIT_BYTES,
        max_output: int = MAX_OUTPUT_BYTES,
    ) -> None:
        self.timeout     = timeout
        self.cpu_time    = cpu_time
        self.mem_bytes   = mem_bytes
        self.max_output  = max_output
        self._tmp_files: List[str] = []
        self._lock = threading.Lock()

    @staticmethod
    def _strip_fences(text: str) -> str:
        m = _FENCE_RE.search(text)
        return (m.group(1) if m else text).strip()

    @contextmanager
    def _tempfile(self, content: str):
        path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as fh:
                fh.write(content)
                path = fh.name
            with self._lock:
                self._tmp_files.append(path)
            yield path
        finally:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                finally:
                    with self._lock:
                        if path in self._tmp_files:
                            self._tmp_files.remove(path)

    # --------------- Sandboxing primitives ------------------------------ #
    def _posix_rlimits(self):
        if resource is None:
            return
        # CPU time (seconds)
        if self.cpu_time:
            resource.setrlimit(resource.RLIMIT_CPU, (self.cpu_time, self.cpu_time + 1))
        # Address space
        if self.mem_bytes:
            resource.setrlimit(resource.RLIMIT_AS, (self.mem_bytes, self.mem_bytes))
        # Prevent core files
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    # --------------- Core execution logic ------------------------------- #
    def _run_once(
        self,
        script: str,
        stdin_data: str,
    ) -> Tuple[str, str]:
        """
        Low‑level runner with incremental read and hard caps.
        Returns (**stdout**, **stderr**) – each possibly truncated.
        """
        start = time.time()
        proc = subprocess.Popen(
            [sys.executable, script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,                   # line‑buffered
            preexec_fn=(
                self._posix_rlimits       # sets rlimits & starts new pgid
                if resource is not None else None
            ),
            close_fds=True,
        )
        try:
            if os.name != "nt":
                # New process‑group so we can kill children too
                os.setpgid(proc.pid, proc.pid)
        except Exception:
            pass  # best‑effort

        # Feed stdin immediately then close
        if proc.stdin:
            proc.stdin.write(stdin_data)
            proc.stdin.close()

        sel = selectors.DefaultSelector()
        sel.register(proc.stdout, selectors.EVENT_READ)
        sel.register(proc.stderr, selectors.EVENT_READ)

        out_buf: List[str] = []
        err_buf: List[str] = []
        out_size = err_size = 0
        truncated = False

        # Poll loop
        while True:
            if time.time() - start > self.timeout:
                truncated = True
                break
            for key, _ in sel.select(timeout=0.1):
                chunk = key.fileobj.readline()
                if not chunk:  # EOF
                    sel.unregister(key.fileobj)
                    continue
                if key.fileobj is proc.stdout:
                    out_size += len(chunk.encode())
                    out_buf.append(chunk)
                    if out_size > self.max_output:
                        truncated = True
                        break
                else:
                    err_size += len(chunk.encode())
                    err_buf.append(chunk)
                    if err_size > self.max_output:
                        truncated = True
                        break
            if truncated:
                break
            # Exit when the process is dead *and* pipes drained/EOF‑ed
            if proc.poll() is not None and not sel.get_map():
                break

        # Time / size overflow – terminate group
        if truncated and proc.poll() is None:
            try:
                if os.name != "nt":
                    os.killpg(proc.pid, 15)  # TERM
                    time.sleep(0.2)
                    if proc.poll() is None:
                        os.killpg(proc.pid, 9)  # KILL
                else:
                    proc.kill()
            except Exception:
                proc.kill()

        # Drain any remaining
        try:
            rest_out, rest_err = proc.communicate(timeout=0.2)
            out_buf.append(rest_out or "")
            err_buf.append(rest_err or "")
        except Exception:
            pass  # child already gone

        out_text = "".join(out_buf)
        err_text = "".join(err_buf)
        if truncated:
            suffix = "\n…<truncated>"
            if len(out_text.encode()) > self.max_output:
                out_text = out_text[: self.max_output] + suffix
            if len(err_text.encode()) > self.max_output:
                err_text = err_text[: self.max_output] + suffix
            if time.time() - start > self.timeout:
                err_text += "\n[TIMEOUT]"
        return out_text, err_text

    # --------------- Public API  ---------------------------------------- #
    def execute(self, raw_code: str, stdin: str | bytes = "") -> Tuple[str, str]:
        """
        Run *raw_code* with *stdin*.  
        • Strips ``` fences automatically.  
        • If the first run yields **no output** and no error but the script
          defines a `solve()` function *without* a mainguard, a tiny runner
          is appended and the code is run a second time.
        """
        code = self._strip_fences(raw_code)
        def _auto_run_needed(src: str, out: str, err: str) -> bool:
            return (
                (not out.strip() and not err.strip())
                and "def solve" in src
                and not _HAS_MAIN.search(src)
            )
        with self._tempfile(code) as script_path:
            try:
                out, err = self._run_once(script_path, stdin)
            except subprocess.SubprocessError as e:
                # catch any low‑level subprocess failure and return it as stderr
                return "", f"Execution error: {e}"

            # if no output/error but there’s a solve() without main guard, re‑run with auto‑runner
            if _auto_run_needed(code, out, err):
                runner = (
                    "\n\nif __name__ == \"__main__\":\n"
                    "    res = solve()\n"
                    "    if res is not None:\n"
                    "        import sys\n"
                    "        if isinstance(res, (list, tuple)):\n"
                    "            print(*res, sep=\" \")\n"
                    "        else:\n"
                    "            print(res)\n"
                )
                with self._tempfile(code + runner) as auto_path:
                    try:
                        out, err = self._run_once(auto_path, stdin)
                    except subprocess.SubprocessError as e:
                        return "", f"Execution error: {e}"

        return out, err

    # --------------- Manual cleanup (rarely needed) ---------------------- #
    def cleanup(self) -> None:
        with self._lock:
            for f in list(self._tmp_files):
                try:
                    os.remove(f)
                except Exception:
                    pass
            self._tmp_files.clear()
