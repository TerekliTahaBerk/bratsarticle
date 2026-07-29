# Gate 8 Completion

**Decision:** PASS

## Artifact audit

- Valid reportable arms: 12/12
- Validation patients per arm: 37
- Invalid runs: 0
- Duplicate arms: 0
- Internal-test access used: false
- GPU-hour range: 0.317023-0.406820

Diagnostic and prior-protocol runs listed as foreign by the audit were excluded from selection.

## Architecture screen

| Arm | Mean regional Dice | Paired mean difference | 95% bootstrap CI | Eliminated |
|---|---:|---:|---:|:---:|
| architecture_bunet | 0.750594 | 0.000000 | [0.000000, 0.000000] | no |
| architecture_resunet | 0.727590 | -0.023004 | [-0.043565, -0.003953] | yes |
| architecture_resunet_wc | 0.741162 | -0.009433 | [-0.022799, 0.003891] | no |
| architecture_unet | 0.721306 | -0.029288 | [-0.050925, -0.009923] | yes |
| architecture_unet_res | 0.745452 | -0.005142 | [-0.015259, 0.004319] | no |
| architecture_unet_wc | 0.741497 | -0.009097 | [-0.025445, 0.006188] | no |

Shortlist: architecture_bunet, architecture_unet_res, architecture_unet_wc

## Loss screen

| Arm | Mean regional Dice | Paired mean difference | 95% bootstrap CI | Eliminated |
|---|---:|---:|---:|:---:|
| architecture_unet | 0.721306 | -0.022515 | [-0.041394, -0.005535] | yes |
| loss_binary_cross_entropy | 0.623035 | -0.120786 | [-0.156444, -0.087173] | yes |
| loss_binary_cross_entropy_plus_focal_tversky | 0.743821 | 0.000000 | [0.000000, 0.000000] | no |
| loss_binary_cross_entropy_plus_soft_dice | 0.711079 | -0.032742 | [-0.053639, -0.014528] | yes |
| loss_cross_entropy | 0.700487 | -0.043334 | [-0.056251, -0.030474] | yes |
| loss_focal_tversky | 0.245402 | -0.498419 | [-0.534487, -0.453615] | yes |
| loss_soft_dice | 0.237709 | -0.506112 | [-0.541798, -0.462342] | yes |

Shortlist: loss_binary_cross_entropy_plus_focal_tversky

## Scope

These are single-seed development-screen results. They support shortlisting only; they are not internal held-out test results and do not establish generalization, clinical applicability, or final model superiority.
