# Capacity- and compute-matching search

Status: **selected before main training**

A deterministic exhaustive search evaluated plain U-Nets with base widths 4-64 and depths 2-6 at input 1x4x240x240. No RES, WC, or residual block was allowed in either control.
For parameter matching, target depth was preserved whenever a same-depth candidate met the 2% tolerance, reducing topology as a source of confounding. A global width/depth fallback was permitted only if no such candidate existed.

| Model | Width | Depth | Parameters | MAC/slice | Difference | Within 2% |
|---|---:|---:|---:|---:|---:|:---:|
| U-Net+RES target | 16 | 4 | 4,450,452 | 9,459,302,400 | — | — |
| Parameter-matched plain U-Net | 24 | 4 | 4,368,364 | 5,994,086,400 | 1.8445% parameters | yes |
| Compute-matched plain U-Net | 30 | 4 | 6,823,774 | 9,348,480,000 | 1.1716% MAC | yes |

The parameter-matched and compute-matched controls are distinct estimands. Their realized wall-clock budgets and measured peak unified memory remain training-time Gate D/F evidence.
