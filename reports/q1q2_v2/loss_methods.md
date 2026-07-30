# Loss definitions and code-Methods parity

Catalog SHA-256: `51adbae394337f7c2b8c1cebffbe28b1fb4bdd1ccead28966559555f581948c2`

All losses consume the same four raw logits. Multiclass CE and overlap terms use softmax. BCE uses independent sigmoid values for its term only; this is a loss construction and does not change the mutually exclusive four-class inference rule.

| Loss | CE transform | BCE transform | Overlap transform | BCE bg | Overlap bg |
|---|---|---|---|:---:|:---:|
| cross_entropy_plus_soft_dice | softmax | not_applicable | softmax | None | False |
| binary_cross_entropy_plus_focal_tversky | not_applicable | independent sigmoid | softmax | False | False |
| cross_entropy_plus_focal_tversky | softmax | not_applicable | softmax | None | False |

Overlap sums use the batch and every spatial axis, producing one value per selected class before the declared mean reduction. CE uses PyTorch mean reduction over batch and spatial elements. BCE uses elementwise logits loss, optional foreground channel selection, then the declared global mean.

The architecture-attribution loss remains pending until the three mandatory candidates complete development-only five-fold selection. Neither the legacy 74-patient subset nor external labels may influence this selection.
