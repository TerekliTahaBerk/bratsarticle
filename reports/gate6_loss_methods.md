# Gate 6 Loss Methods Table

All entries consume raw four-channel logits. Targets are the mutually
exclusive BraTS classes `{0, 1, 2, 4}`, mapped internally to contiguous
indices. Class weights are disabled (`null`) in the declared catalog.

| Loss | Formula | alpha | beta | gamma | Smooth | Background | Reduction |
|---|---|---:|---:|---:|---:|:---:|:---:|
| cross_entropy | `-sum_c w_c y_c log softmax(z)_c` | 0.5 | 0.5 | 1 | 1e-05 | yes | mean |
| binary_cross_entropy | `-sum_c w_c [y_c log sigmoid(z_c) + (1-y_c) log(1-sigmoid(z_c))]` | 0.5 | 0.5 | 1 | 1e-05 | yes | mean |
| soft_dice | `1 - mean_c [(2 sum p_c y_c + s)/(sum p_c + sum y_c + s)]` | 0.5 | 0.5 | 1 | 1e-05 | no | mean |
| cross_entropy_plus_soft_dice | `0.5 CE + 0.5 SoftDice` | 0.5 | 0.5 | 1 | 1e-05 | no | mean |
| binary_cross_entropy_plus_soft_dice | `0.5 BCE + 0.5 SoftDice` | 0.5 | 0.5 | 1 | 1e-05 | no | mean |
| focal_tversky | `mean_c (1 - (TP+s)/(TP+alpha FP+beta FN+s))^gamma` | 0.3 | 0.7 | 0.75 | 1e-05 | no | mean |
| binary_cross_entropy_plus_focal_tversky | `0.5 BCE + 0.5 FocalTversky` | 0.3 | 0.7 | 0.75 | 1e-05 | no | mean |

`CE` uses softmax cross-entropy. `BCE` uses channel-wise sigmoid BCE
against a one-hot target. Soft Dice and focal Tversky use softmax
probabilities; combined objectives have equal 0.5/0.5 term weights.
These are optimization candidates only. The central evaluator and its
empty-mask rules are unchanged.
