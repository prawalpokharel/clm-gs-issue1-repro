"""Run CLM-GS on a Modal A10 to reproduce and test nyu-systems/CLM-GS#1.

Commands (MODAL_PROFILE picks the workspace that pays):
    modal run modal_clm.py::stage          # download Tanks&Temples truck to a Volume (CPU)
    modal run modal_clm.py::c1             # reproduce: bsz sweep at an open-file limit of 1024
    modal run modal_clm.py::c2             # candidate fixes at bsz 64, plus throughput/PSNR

Each training run is started with a soft RLIMIT_NOFILE set by the runner and
watched by a thread that counts /proc/<pid>/fd every 0.2 s.
"""

from __future__ import annotations

import json
import os
import re
import resource
import shutil
import subprocess
import threading
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CLM_COMMIT = "1119a10"
DATA_URL = "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip"

volume = modal.Volume.from_name("clm-gs-data", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential", "ninja-build", "libgl1", "libglib2.0-0", "unzip", "wget")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "tqdm", "plyfile", "psutil", "numba", "opencv-python-headless", "scipy",
        "matplotlib", "pandas", "imageio", "imageio-ffmpeg", "requests", "tabulate",
        "ninja", "setuptools<75", "wheel",
    )
    # Modal's Python was built with clang; link extensions with the image's gcc
    .env({"CC": "gcc", "CXX": "g++", "LDSHARED": "g++ -pthread -shared"})
    .run_commands(
        "git config --global url.https://github.com/.insteadOf git@github.com:",
        "git clone https://github.com/nyu-systems/CLM-GS.git /opt/CLM-GS",
        f"cd /opt/CLM-GS && git checkout {CLM_COMMIT} && git submodule update --init --recursive",
        # CUDA extensions for the A10 (sm_86). cpu-adam uses -march=native,
        # so it is built at run time on the GPU host instead.
        "cd /opt/CLM-GS && export TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=8 && "
        "pip install --no-build-isolation submodules/clm_kernels && "
        "pip install submodules/fast-tsp && "
        "pip install --no-build-isolation submodules/simple-knn && "
        "pip install --no-build-isolation submodules/gsplat",
    )
    .add_local_dir(HERE / "patches", "/root/patches")
)
app = modal.App("clm-gs-fd-repro", image=image)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@app.function(volumes={"/vol": volume}, cpu=2, memory=4096, timeout=1800)
def stage_data() -> str:
    target = Path("/vol/tandt/truck")
    if (target / "sparse").exists():
        return f"already staged: {sorted(os.listdir(target))}"
    subprocess.run(["wget", "-q", "-O", "/tmp/tandt_db.zip", DATA_URL], check=True)
    subprocess.run(["unzip", "-q", "/tmp/tandt_db.zip", "tandt/truck/*", "-d", "/vol"], check=True)
    volume.commit()
    n_images = len(os.listdir(target / "images"))
    return f"staged {target}: {sorted(os.listdir(target))}, {n_images} images"


# ---------------------------------------------------------------------------
# Training runs
# ---------------------------------------------------------------------------
def _watch_fds(pid: int, samples: list, stop: threading.Event) -> None:
    start = time.monotonic()
    while not stop.is_set():
        try:
            samples.append((round(time.monotonic() - start, 1), len(os.listdir(f"/proc/{pid}/fd"))))
        except FileNotFoundError:
            return
        stop.wait(0.2)


def _run_one(cfg: dict, data_dir: str) -> dict:
    name = cfg["name"]
    out_dir = Path("/root/out") / name
    shutil.rmtree(out_dir, ignore_errors=True)
    cmd = [
        "python", "train.py",
        "-s", data_dir,
        "--iterations", str(cfg["iterations"]),
        "--bsz", str(cfg["bsz"]),
        "--log_interval", str(cfg["bsz"] * 10),
        "--log_folder", str(out_dir),
        "--model_path", str(out_dir),
        "--clm_offload",
        "--prealloc_capacity", "2_000_000",
        "--end2end_time",
        "--eval",
        "--test_iterations", str(cfg.get("test_iteration", cfg["iterations"])),
        "--save_iterations", "999999999",
        *cfg.get("extra", []),
    ]
    soft = cfg["nofile"]

    def limit() -> None:
        _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(soft, hard), hard))

    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    start = time.monotonic()
    proc = subprocess.Popen(
        cmd, cwd="/opt/CLM-GS", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, preexec_fn=limit, env=env,
    )
    samples: list = []
    stop = threading.Event()
    watcher = threading.Thread(target=_watch_fds, args=(proc.pid, samples, stop), daemon=True)
    watcher.start()
    try:
        output, _ = proc.communicate(timeout=cfg.get("timeout", 900))
        timed_out = False
    except subprocess.TimeoutExpired:
        proc.kill()
        output, _ = proc.communicate()
        timed_out = True
    stop.set()
    watcher.join()

    logs = "".join(p.read_text(errors="replace") for p in sorted(out_dir.glob("*.log")))
    text = output + "\n" + logs
    throughput = re.findall(r"throughput ([0-9.]+) it/s", text)
    psnr = re.findall(r"PSNR ([0-9.]+)", text)
    error_lines = [
        line.strip() for line in output.splitlines()
        if re.search(r"Error|ancdata|exited unexpectedly|Too many open files|Killed", line)
    ]
    # Downsample the fd series to at most 60 points
    step = max(1, len(samples) // 60)
    return {
        **cfg,
        "exit_code": proc.returncode,
        "timed_out": timed_out,
        "wall_s": round(time.monotonic() - start, 1),
        "max_fds": max((n for _, n in samples), default=None),
        "fd_series": samples[::step],
        "throughput_it_s": float(throughput[-1]) if throughput else None,
        "psnr": [float(x) for x in psnr],
        "ancdata_error": "ancdata" in output,
        "errors": error_lines[:12],
        "output_tail": output[-3000:],
    }


def _prepare(patch: str | None) -> str:
    data_dir = "/root/data/truck"
    if not Path(data_dir).exists():
        shutil.copytree("/vol/tandt/truck", data_dir)
    subprocess.run(
        ["pip", "install", "-q", "--no-build-isolation", "submodules/cpu-adam"],
        cwd="/opt/CLM-GS", check=True,
    )
    subprocess.run(["git", "checkout", "-q", "--", "."], cwd="/opt/CLM-GS", check=True)
    if patch:
        subprocess.run(["git", "apply", f"/root/patches/{patch}"], cwd="/opt/CLM-GS", check=True)
    return data_dir


@app.function(
    gpu="A10G", cpu=8, memory=32768, timeout=3 * 3600, volumes={"/vol": volume}
)
def run_configs(groups: list[dict]) -> list[dict]:
    hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    results = []
    for group in groups:
        data_dir = _prepare(group.get("patch"))
        for cfg in group["configs"]:
            res = _run_one(cfg, data_dir)
            res["patch"] = group.get("patch")
            res["container_nofile"] = hard
            results.append(res)
            print(json.dumps({k: res[k] for k in (
                "name", "bsz", "nofile", "exit_code", "max_fds", "ancdata_error",
                "throughput_it_s", "psnr", "wall_s")}), flush=True)
    return results


def _save(results: list[dict], name: str) -> None:
    (HERE / "results").mkdir(exist_ok=True)
    path = HERE / "results" / f"{name}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in results))
    print(f"wrote {path}")


@app.local_entrypoint()
def stage() -> None:
    print(stage_data.remote())


@app.local_entrypoint()
def c1() -> None:
    # Unpatched code. 251 truck images; iterations are counted in images.
    configs = [
        {"name": f"c1_bsz{b}_nofile1024", "bsz": b, "nofile": 1024,
         "iterations": b * 30, "timeout": 300}
        for b in [4, 16, 32, 48, 64]
    ]
    # Peak descriptor use with a high limit, to size the fix
    configs += [
        {"name": f"c1_bsz{b}_nofile65536", "bsz": b, "nofile": 65536,
         "iterations": b * 30, "timeout": 300}
        for b in [16, 64]
    ]
    _save(run_configs.remote([{"patch": None, "configs": configs}]), "c1_reproduce")


@app.local_entrypoint()
def c2() -> None:
    long_run = 64 * 110  # about 28 epochs of truck at bsz 64
    reps = [1, 2]
    groups = [
        # Control: unpatched code, enough descriptors.
        # Workaround that needs no code change: --sharing_strategy file_system
        {"patch": None, "configs": [
            *[{"name": f"c2_control_nofile65536_r{r}", "bsz": 64, "nofile": 65536,
               "iterations": long_run} for r in reps],
            *[{"name": f"c2_file_system_nofile1024_r{r}", "bsz": 64, "nofile": 1024,
               "iterations": long_run, "extra": ["--sharing_strategy", "file_system"]}
              for r in reps],
        ]},
        # Candidate fix: train.py raises the soft limit itself
        {"patch": "clm-gs-raise-nofile.patch", "configs": [
            {"name": f"c2_patched_nofile1024_r{r}", "bsz": 64, "nofile": 1024,
             "iterations": long_run} for r in reps
        ]},
    ]
    _save(run_configs.remote(groups), "c2_fix")
