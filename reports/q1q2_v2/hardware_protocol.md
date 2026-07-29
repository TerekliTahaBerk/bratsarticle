# q1q2 v2 hardware protocol and freeze boundary

## Realized audit host

- Host accelerator: Apple M1 Max, 32 GPU cores, 32 GB unified memory.
- Backend: PyTorch MPS.
- Resource wording: MPS framework-reported allocated unified memory and
  driver-allocated memory; these values are not called discrete-GPU VRAM.
- Evidence: `hardware_preflight.json`.
- Permitted use: unit tests, data/design audits, static profiling and
  non-reportable operator smokes.

## Superseded legacy target

The older A100 Gate 7 files are an unrealized legacy design target. They do not
describe the realized legacy MPS runs and do not define the q1q2 v2 execution
environment. Their status is made explicit in
`legacy/v1/a100_protocols_SUPERSEDED.md`.

## Required reportable environment

Before any pilot or full q1q2 v2 training, record and freeze:

1. scheduler and launch command;
2. exact GPU model and count;
3. driver, CUDA, cuDNN, PyTorch and MONAI versions;
4. per-job wall-time limit and total allocated GPU-hours;
5. local/scratch storage and checkpoint retention limits;
6. container digest and cluster-specific immutable lock;
7. deterministic-algorithm settings and any documented exceptions.

The current single-MPS host is a hard scheduling and total-compute blocker for
the frozen 615-run design. No training or external inference is authorized by
this protocol until those fields are supplied and the compute gate is rerun.
