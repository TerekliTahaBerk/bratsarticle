# Apple M1 Max calibration

Status: **NONREPORTABLE HARDWARE CALIBRATION**

Synthetic tensors were used. No raw, legacy internal-test, or external data were opened, and these losses are not scientific results.

| Model | Input | Effective batch | Median optimizer step | Status |
|---|---:|---:|---:|---|
| unet_small | [16, 4, 240, 240] | 16 | 0.196 s | pass |
| unet_parameter_matched_res | [16, 4, 240, 240] | 16 | 0.298 s | pass |
| unet_compute_matched_res | [16, 4, 240, 240] | 16 | 0.394 s | pass |
| unet_res | [16, 4, 240, 240] | 16 | 0.452 s | pass |
| unet_wc | [16, 4, 240, 240] | 16 | 0.223 s | pass |
| bunet | [16, 4, 240, 240] | 16 | 0.479 s | pass |
| resblock_unet | [16, 4, 240, 240] | 16 | 0.229 s | pass |
| resblock_unet_wc | [16, 4, 240, 240] | 16 | 0.248 s | pass |
| unet_2p5d_k5 | [8, 20, 240, 240] | 8 | 0.105 s | pass |
| swin_unetr | [1, 4, 96, 96, 96] | 2 | 3.515 s | pass |

Measured-model 50,000-step upper proxy: 2,132.2 serial hours (88.8 days), before validation and the explicitly excluded work.

This report decides feasibility only. It cannot justify reducing the five folds, the common five seeds, or the convergence rule.
