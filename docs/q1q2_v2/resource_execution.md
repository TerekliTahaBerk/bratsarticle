# Measured resource and accuracy-cost analysis

`configs/q1q2_v2/resource_execution.yaml` is frozen by Gate G before external
results. The analysis remains blocked until all 600 development runs pass Gate
G and the single Gate H external session passes.

Every development run must provide the same frozen environment and hardware
contract, a best-checkpoint hash, parameter count, declared input/output
shapes, operation counts, a transparent receptive-field proxy, checkpoint
size, MPS unified-memory measurements, 100 synchronized effective optimizer
step timings, and total accelerator hours. Native 2D models use the declared
slice input. Swin UNETR and 3D nnU-Net use their frozen patch input. These
operation counts therefore describe the declared tensor input and are not
silently presented as whole-volume inference costs.

Gate H contributes preprocessing, forward-only, postprocessing, end-to-end
patient-volume latency, throughput, and MPS unified-memory measurements.
Confirmatory accuracy is the mean patient-level regional Dice across the 95
frozen external glioma cases after averaging the 25 fold-seed checkpoints
within each model.

Run after Gate H:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/analyze_q1q2_resources.py
```

The generated Pareto table uses measured values only. It reports separate
accuracy-versus-cost fronts and an all-cost non-dominated flag. No weighted or
subjective cost score is constructed. Parameter-matched and compute-matched
controls are emitted as a separate required table. Missing, non-finite, mixed
environment, or incomplete resource fields block completion.
