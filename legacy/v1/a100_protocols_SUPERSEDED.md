# Superseded, unrealized A100 protocol

The Gate 7 files `configs/protocols/convergence_matched.yaml` and
`configs/protocols/compute_matched.yaml` describe a frozen design target of one
NVIDIA A100-SXM4-80GB. That target was not the hardware used for the legacy
Gate 8–11 runs. Those runs were realized on an Apple M1 Max through MPS and are
preserved at tag `v1-bounded-2d-component-study`.

For the q1q2 v2 study, neither the legacy A100 design nor the bounded MPS run
artifacts authorize new reportable training. The current hardware decision and
compute blocker are recorded in:

- `reports/q1q2_v2/hardware_preflight.json`
- `reports/q1q2_v2/compute_budget.json`
- `reports/q1q2_v2/hardware_protocol.md`

The cluster-specific CUDA protocol must be frozen before pilot execution once
the required allocation is supplied.
