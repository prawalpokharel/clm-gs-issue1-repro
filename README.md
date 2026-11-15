# CLM-GS issue #1: crash at --bsz 64 (open-file limit)

Measured 2026-11-13 against nyu-systems/CLM-GS main at 1119a10.

Issue: https://github.com/nyu-systems/CLM-GS/issues/1 (MatrixCity BigCity,
clm_offload, bsz 64: "received 0 items of ancdata", "Pin memory thread
exited unexpectedly", then a stall). The maintainer's reply suggested file
descriptor exhaustion and asked the reporter to try `ulimit -n 65536`.

## Finding

With PyTorch's default `file_descriptor` sharing strategy, each tensor the
DataLoader worker sends keeps one file descriptor open in the training
process while the tensor is alive. A CLM-GS `Camera` has six tensors (image,
world_view_transform and its backup, projection_matrix, full_proj_transform,
camera_center), and `collate_fn` passes 64 Camera objects per batch.

Measured on the real code (Tanks&Temples truck, A10, clm_offload), peak open
descriptors in the training process:

| --bsz | 4 | 16 | 32 | 64 |
|---|---|---|---|---|
| peak descriptors | 145 | 335 | 569 | 1081 |

That is about 15.6 x bsz + 81. With the common soft limit of 1024
(`ulimit -n`), bsz 64 is the only supported batch size that fails (CLM-GS
accepts 4, 8, 16, 32, 64). The release scripts for Mip-NeRF 360 and Rubble use
bsz 4, and only `bigcity.sh` uses 64. The bug is a property of the batch size,
not of the scene, so it reproduces on a 251-image scene.

At bsz 64 with the limit at 1024, the pin-memory thread dies in
`rebuild_storage_fd -> recv_handle` with `OSError: [Errno 24] Too many open
files`. That is the same function as the reported "received 0 items of
ancdata". Training then stalls until killed. `torch.compile` also fails
because it can't open a module file to import it.

## Fix check (C2: bsz 64, 7,040 images, two runs each)

| Setup | Open-file limit | Peak descriptors | Throughput (it/s) | Test PSNR |
|---|---|---|---|---|
| Unpatched control | 65,536 | 1,082 | 55.3 / 55.7 | 22.27 / 22.21 |
| Unpatched + `--sharing_strategy file_system` | 1,024 | 60 | 58.6 / 58.2 | 22.26 / 22.11 |
| Patched (`patches/clm-gs-raise-nofile.patch`) | 1,024 | 1,082 | 56.0 / 56.3 | 22.19 / 22.29 |

- The patch (18 lines in train.py) raises the soft limit to the hard limit at
  startup. It fixes the crash with no change in speed or quality. It doesn't
  help where the hard limit itself is low.
- The existing `--sharing_strategy file_system` option also fixes it, needs
  no code change, uses 18x fewer descriptors and was about 5% faster in both
  runs. Two runs is a small sample.
- A third option, from the CPU stand-in only (C0): send the five small
  matrices as numpy arrays, so only the image uses shared memory. That cut
  descriptors per camera from 6 to 1 and made loading 5x faster in C0, but it
  changes `Camera` and was not tested on the real code.

## Final patch (C3)

After review the patch became `patches/clm-gs-final.patch` (10 lines). The
limit is now raised inline in `training()`, only when the `file_descriptor`
sharing strategy is in use, and the change is logged. Same setup as C2:

| Setup | Open-file limit | Peak descriptors | Throughput (it/s) | Test PSNR |
|---|---|---|---|---|
| Final patch | 1,024 | 1,082 / 1,081 | 56.0 / 56.5 | 22.24 / 22.26 |
| Final patch + `--sharing_strategy file_system` (1,920 images) | 1,024 | 66 | 77.8 | 19.61 |

Both full runs logged "Raised open file limit from 1024 to 1048576" and
matched the C2 patch runs. With `file_system` the limit was left unchanged,
as intended. The short run's PSNR isn't comparable, because it trained on
fewer images.

## Files

| Path | What it is |
|---|---|
| `c0_fd_repro.py`, `modal_c0.py` | C0: CPU stand-in that mirrors the CLM-GS DataLoader, sweeps bsz and limits |
| `modal_clm.py` | Modal A10 image (CUDA 12.4, torch 2.6, all submodules), data staging, C1, C2 and C3 |
| `patches/clm-gs-raise-nofile.patch` | The first version of the fix, tested in C2, applies to 1119a10 |
| `patches/clm-gs-final.patch` | The final fix, tested in C3, applies to 1119a10 |
| `results/c0_fd_repro_full.jsonl` | C0 sweep |
| `results/c1_reproduce.jsonl` | C1: bsz sweep on the real code, with descriptor time series and tracebacks |
| `results/c2_fix.jsonl` | C2: control, flag workaround, patch |
| `results/c3_final.jsonl` | C3: final patch, and the check with `file_system` |

C0 note: in the CPU stand-in the main process peaked at only 399 descriptors
at bsz 64, yet the run hung at the 1024 limit. The stand-in has no pin-memory
thread and no CUDA, so its main process holds fewer live batches than the
real code. The real-code numbers above are the ones to cite. A follow-up C0
run to count worker descriptors timed out without results.

## Cost

From `modal billing report`: $1.02 for all six Modal apps (two image builds,
data staging, both C0 runs, C1 and C2). A10G time was $0.42 of that. C3's
three training runs added 5.4 minutes on one A10G, plus image build and setup.

## Rerunning

Needs a Modal account (`pip install modal`, then `modal setup`). `stage`
downloads the Tanks&Temples truck scene (650 MB) into a Modal volume named
`clm-gs-data`. Delete it afterwards with `modal volume delete clm-gs-data`.

```bash
modal run modal_clm.py::stage
modal run modal_clm.py::c1
modal run modal_clm.py::c2
modal run modal_clm.py::c3
modal run modal_c0.py
```

The image uses Python 3.10. Modal's 2026.10 image builder no longer supports
3.10, so on a workspace that uses it, set `MODAL_IMAGE_BUILDER_VERSION=2025.06`
before these commands.

## Licence

MIT for the scripts here (see `LICENSE`). The files in `patches/` are changes
to CLM-GS and follow the CLM-GS licence.
