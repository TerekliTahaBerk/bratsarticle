# Prespecified statistical analysis plan

Status: **prespecified; Gate G remains pending**

The primary question is whether U-Net+RES improves external patient-level mean
regional Dice relative to the topology-preserving, parameter-matched plain
U-Net. The endpoint is the arithmetic mean of WT, TC and ET Dice for each
patient. Replicate-level metrics are averaged over all valid frozen fold × seed
models before the paired patient contrast.

The confirmatory family contains the primary contrast and four ordered
secondary contrasts: BU-Net versus U-Net+RES, U-Net+RES versus the
compute-matched U-Net, BU-Net versus the parameter-matched U-Net, and nnU-Net
v2 3D full-resolution versus the best 2D model selected using development CV
only. Two-sided raw p values will be adjusted together by Holm at alpha 0.05.

Uncertainty reporting includes a paired patient bootstrap, a hierarchical
bootstrap that resamples training seeds before patients, and a mixed-effects
sensitivity analysis with model and fold as fixed effects and patient and seed
random intercepts. Every contrast reports mean and median paired differences,
95% confidence interval, standardized paired effect, raw and adjusted p
values, and probability of superiority.

The practical-effect threshold is 0.020 mean-regional Dice. It is an
interpretive threshold intended to distinguish a distributed two-point
regional gain from a rounding-level change. It is not a clinical MCID and is
not used as an equivalence or non-inferiority margin. The 95-patient
precision/power calculation is stored separately.

ET lesion recall, ET lesion-wise Dice, regional HD95 and false-positive lesion
count are the principal safety/failure endpoints. Other regional, boundary,
volume and lesion metrics are estimation-only unless explicitly named above.
One-empty HD95 remains infinity; no silent imputation is permitted.

External thresholds, loss, post-processing and model selection are prohibited.
All frozen models must be reported, including failures. The 74-patient legacy
internal subset cannot be used for new-model inference. Gate G can pass only
after development loss selection, all reportable checkpoints, configuration
hashes and the external access guard are frozen in one immutable manifest.
