# Gate 6 Model Inventory

Generated from the versioned model configurations. Parameter matching uses
the controlled 16-channel U-Net as its target and searches integer base
widths from 1 through 64. A failed tolerance is retained as a failure.

| Model | RB | RES | WC | Parameters | Delta | Match width | Match parameters | Gap | Within 5% |
|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|:---:|
| bunet | — | ✓ | ✓ | 8,384,660 | +6,441,888 | 8 | 2,098,956 | 8.04% | FAIL |
| resunet | ✓ | — | — | 2,030,612 | +87,840 | 16 | 2,030,612 | 4.52% | PASS |
| resunet_wc | ✓ | — | ✓ | 5,964,820 | +4,022,048 | 9 | 1,888,951 | 2.77% | PASS |
| unet | — | — | — | 1,942,772 | +0 | 16 | 1,942,772 | 0.00% | PASS |
| unet_res | — | ✓ | — | 4,450,452 | +2,507,680 | 11 | 2,105,492 | 8.38% | FAIL |
| unet_wc | — | — | ✓ | 5,876,980 | +3,934,208 | 9 | 1,860,961 | 4.21% | PASS |

The equal-width matrix isolates feature additions while exposing their
parameter cost. The closest-width results are a sensitivity design, not
substitutes for the primary matrix. BU-Net and U-Net+RES cannot meet the
declared 5% target with a single integer base-width multiplier; this
limitation is therefore explicit.

Complete per-module tensor traces and configuration hashes are stored in
`reports/gate6_inventory.json`.
