"""C0: CPU reproduction of the CLM-GS DataLoader crash (nyu-systems/CLM-GS#1).

Mirrors the training DataLoader in CLM-GS train.py: one persistent worker,
identity collate, shuffle, drop_last, and a Dataset that returns one Camera
module per item. The Camera stand-in has the same six tensor storages as
scene/cameras.py with offload=True (image, world_view_transform and its
backup, projection_matrix, full_proj_transform, inverse for camera_center).

Each configuration runs in a child process with its own open-file limit.
Output: one JSON line per configuration.
Linux only (reads /proc/self/fd).
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import resource
import threading
import time
import traceback

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class FakeCamera(nn.Module):
    """Same tensor layout as CLM-GS Camera(offload=True)."""

    def __init__(self, idx: int, h: int, w: int, numpy_matrices: bool) -> None:
        super().__init__()
        self.uid = idx
        self.original_image_backup = torch.full((3, h, w), idx % 255, dtype=torch.uint8)
        wvt = torch.eye(4) + idx
        proj = torch.eye(4) * 2
        if numpy_matrices:
            # Candidate fix: small matrices travel by value, no shared memory
            self.world_view_transform = wvt.numpy()
            self.world_view_transform_backup = wvt.clone().numpy()
            self.projection_matrix = proj.numpy()
            self.full_proj_transform = (wvt @ proj).numpy()
            self.camera_center = torch.linalg.inv(wvt)[3, :3].numpy()
        else:
            self.world_view_transform = wvt
            self.world_view_transform_backup = wvt.clone()
            self.projection_matrix = proj
            self.full_proj_transform = (wvt.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
            self.camera_center = torch.linalg.inv(wvt)[3, :3]


class CameraDataset(Dataset):
    def __init__(self, n: int, h: int, w: int, numpy_matrices: bool) -> None:
        self.n, self.h, self.w, self.numpy_matrices = n, h, w, numpy_matrices

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> FakeCamera:
        return FakeCamera(idx, self.h, self.w, self.numpy_matrices)


def identity(batch):
    return batch


def open_fds() -> int:
    return len(os.listdir("/proc/self/fd"))


def run_one(cfg: dict, out: mp.Queue, stats) -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(cfg["nofile"], hard), hard))
    if cfg["fix"] == "raise_nofile":
        # Candidate fix: raise the soft limit to the hard limit at startup
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    torch.multiprocessing.set_sharing_strategy(
        "file_system" if cfg["fix"] == "file_system" else "file_descriptor"
    )
    ds = CameraDataset(cfg["n_images"], 64, 64, cfg["fix"] == "numpy_matrices")
    loader = DataLoader(
        ds,
        batch_size=cfg["bsz"],
        num_workers=1,
        shuffle=True,
        drop_last=True,
        persistent_workers=True,
        collate_fn=identity,
    )
    result = dict(cfg)
    result.update(
        soft_limit=resource.getrlimit(resource.RLIMIT_NOFILE)[0],
        baseline_fds=open_fds(),
        iterations=0,
        max_fds=0,
        error=None,
    )
    it = iter(loader)
    # Count descriptors in the worker too, from a thread in this process
    worker_pids = [w.pid for w in getattr(it, "_workers", [])]
    result["max_worker_fds"] = 0
    stop_sampler = threading.Event()

    def sample_workers() -> None:
        while not stop_sampler.is_set():
            for pid in worker_pids:
                try:
                    n = len(os.listdir(f"/proc/{pid}/fd"))
                except OSError:
                    continue
                result["max_worker_fds"] = max(result["max_worker_fds"], n)
                stats[1] = result["max_worker_fds"]
            stats[0] = max(stats[0], open_fds())
            stop_sampler.wait(0.005)

    sampler = threading.Thread(target=sample_workers, daemon=True)
    sampler.start()
    start = time.monotonic()
    try:
        for _ in range(cfg["iters"]):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            result["max_fds"] = max(result["max_fds"], open_fds())
            time.sleep(0.002)  # stands in for a training step
            del batch
            result["iterations"] += 1
            stats[2] = result["iterations"]
    except Exception as e:  # noqa: BLE001
        text = "".join(traceback.format_exception_only(type(e), e)).strip()
        result["error"] = text.splitlines()[-1][:200]
    result["seconds"] = round(time.monotonic() - start, 2)
    stop_sampler.set()
    out.put(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bsz", default="4,8,16,32,40,48,56,64")
    parser.add_argument("--nofile", default="1024")
    parser.add_argument("--fix", default="none")
    parser.add_argument("--iters", type=int, default=60)
    parser.add_argument("--n-images", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()

    ctx = mp.get_context("fork")
    for bsz, nofile, fix in itertools.product(
        [int(x) for x in args.bsz.split(",")],
        [int(x) for x in args.nofile.split(",")],
        args.fix.split(","),
    ):
        cfg = dict(bsz=bsz, nofile=nofile, fix=fix, iters=args.iters, n_images=args.n_images)
        q = ctx.Queue()
        stats = ctx.Array("i", 3)  # max main fds, max worker fds, iterations
        p = ctx.Process(target=run_one, args=(cfg, q, stats))
        p.start()
        p.join(args.timeout)
        if p.is_alive():
            p.kill()
            p.join()
            res = dict(cfg, error=f"hung for more than {args.timeout} s (killed)",
                       max_fds=stats[0], max_worker_fds=stats[1], iterations=stats[2])
        else:
            res = q.get() if not q.empty() else dict(cfg, error=f"child exit code {p.exitcode}")
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
