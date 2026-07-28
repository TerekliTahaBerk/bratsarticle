# Gate 6 Completion — BU-Net/Res U-Net and Ablation System

**Decision:** PASS for implementation and bounded diagnostics  
**Full training:** Not started  
**Internal held-out test access:** Not performed

| Acceptance condition | Result |
|---|---|
| BU-Net, Res U-Net, and Res U-Net + WC implemented | PASS |
| U-Net/RES/WC six-cell ablation matrix | PASS |
| Residual blocks, RES skips, and WC independent | PASS |
| Source-aligned RES/WC with deviations documented | PASS |
| One versioned config per architecture | PASS |
| Parameter count and difference report | PASS |
| Adjustable width and explicit 5% match tolerance | PASS |
| Tensor shape trace and model summary | PASS |
| Unit forward/backward tests | PASS |
| Bounded overfit smoke for all six models | PASS |
| Checkpoint compatibility for all six models | PASS |
| Seven explicit loss candidates and methods table | PASS |

## Verification

- Ruff: PASS
- Mypy strict: PASS
- Pytest: PASS (72 passed)
- CUDA-only evaluator equality: SKIP (CUDA unavailable)
- Six-model synthetic loss-decrease diagnostic: PASS
- Six-model checkpoint state/counter/metadata round trip: PASS
- Full-cohort training: not attempted
- Internal held-out test access: not performed

The smoke artifact records commit
`4a0b5a490e7b8c399462a945655cafc104da9f26`. It is a bounded implementation
diagnostic and contains no validation-performance or superiority evidence.

## Parameter-control result

At equal base width 16, the controlled U-Net has 1,942,772 trainable
parameters. The other configurations range from 2,030,612 to 8,384,660
parameters. The integer-width sensitivity search meets the declared 5%
tolerance for U-Net, Res U-Net, U-Net + WC, and Res U-Net + WC. The nearest
BU-Net and U-Net + RES variants remain 8.04% and 8.38% from the target,
respectively. These failures are retained explicitly, as required when exact
matching is impossible with the declared width parameterization.

## Artifacts

- `reports/bunet_implementation_notes.md`
- `reports/gate6_inventory.json`
- `reports/gate6_model_summary.md`
- `reports/gate6_loss_methods.md`
- `reports/gate6_smoke_results.json`

The inventory JSON contains every configuration hash, parameter count,
feature-flag state, closest-width result, and per-module tensor shape trace.
