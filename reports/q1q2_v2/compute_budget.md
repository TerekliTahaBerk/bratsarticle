# Compute and storage feasibility

Decision: **FULL MATRIX BLOCKED ON THE CURRENT SINGLE-MPS HOST**

The mandatory plan contains 615 reportable training runs before reproduction reruns. The 300-model convergence matrix and 200 core compute-matched runs retain all five folds and the common five-seed list.

Legacy artifact timing gives a median 0.671 seconds per 2D optimizer step, or 9.33 hours at the 50,000-step ceiling. The known native-2D and fixed compute budgets already require thousands of serial accelerator-hours.

Swin UNETR completed a real 64-cubed MPS forward/backward smoke in 0.85 seconds. Even extrapolating that smaller-than-frozen patch yields 11.8 hours per 50,000 steps. This is a lower-bound feasibility proxy, not a final duration claim.

The combined known upper-bound proxy is 4,032 accelerator-hours (168 serial days) and still excludes 25 nnU-Net 2D and 50 nnU-Net 3D/interaction runs.

MPS itself is available and the 3D transformer operators execute, so this is not a device-detection failure. It is a scheduling and total-compute blocker. Full training must not start without a credible cluster allocation or a protocol revision that is documented before results.
