# q1q2 v2 container boundary

The Docker and Apptainer definitions reproduce the realized Python dependency
snapshot for audit and CPU-compatible tests. They have not been promoted to a
reportable training image.

The definitive training image remains blocked until a CUDA allocation is
provided. Before any pilot or full run, freeze and record the selected GPU
class, CUDA runtime, driver, cuDNN, PyTorch build, container digest, job
launcher, per-job wall time and total GPU-hour allocation. Do not infer CUDA
equivalence from the Apple MPS operator smoke.
