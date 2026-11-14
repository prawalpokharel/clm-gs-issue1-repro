"""Run C0 (c0_fd_repro.py) in a Modal CPU container (Linux, no GPU).

    modal run modal_c0.py                  # threshold sweep and fixes
    modal run modal_c0.py --which workers  # worker-side descriptor counts
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", "numpy", index_url="https://download.pytorch.org/whl/cpu")
    .add_local_file(HERE / "c0_fd_repro.py", "/root/c0_fd_repro.py")
)
app = modal.App("clm-c0-fd-repro", image=image)

WORKER_RUNS = [
    # Which process runs out? Count worker descriptors near each threshold
    ["--bsz", "8,32,56,60,64", "--nofile", "1024", "--fix", "none", "--timeout", "60"],
    ["--bsz", "224,240,256", "--nofile", "4096", "--fix", "none", "--timeout", "60"],
]

RUNS = [
    # Where does the crash start, at the common default limit of 1024?
    ["--bsz", "4,8,16,32,40,48,56,64", "--nofile", "1024", "--fix", "none"],
    # Same sweep at 4096, the limit the threshold should scale with
    ["--bsz", "64,128,160,192,224,256", "--nofile", "4096", "--fix", "none"],
    # Candidate fixes at the CLM-GS BigCity setting (bsz 64, limit 1024)
    ["--bsz", "64", "--nofile", "1024", "--fix", "raise_nofile,file_system,numpy_matrices"],
]


@app.function(cpu=8, memory=8192, timeout=1800)
def run_c0(which: str = "full") -> str:
    import resource

    lines = [f"# container RLIMIT_NOFILE={resource.getrlimit(resource.RLIMIT_NOFILE)}"]
    for args in (WORKER_RUNS if which == "workers" else RUNS):
        proc = subprocess.run(
            ["python", "/root/c0_fd_repro.py", *args],
            capture_output=True,
            text=True,
            check=False,
        )
        lines.append(proc.stdout.rstrip())
        if proc.returncode != 0:
            lines.append(f"# exit {proc.returncode}: {proc.stderr[-2000:]}")
    return "\n".join(lines)


@app.local_entrypoint()
def main(which: str = "full") -> None:
    out = run_c0.remote(which)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"c0_fd_repro_{which}.jsonl").write_text(out + "\n")
    print(out)
