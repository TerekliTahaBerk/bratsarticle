# Deterministic qualitative analysis

The case-selection rules and their exact patient-level definitions are frozen
in `configs/q1q2_v2/qualitative_protocol.yaml` before external inference.
Cases are selected after evaluation using prespecified deterministic rules;
they are not prespecified cases.

The analysis requires a passing Gate H and the 12 retained model-level
predictions. It performs no new model inference. For each external
confirmatory patient, it aggregates the frozen model-patient metrics and
computes exact pairwise regional disagreement from the retained strict-majority
model predictions. The six required rules are then applied with the frozen
whole-tumor-volume and anonymized-ID tie-breakers.

Each selected case is rendered at the axial slice with greatest reference
whole-tumor area. The 5 × 6 panel contains T1, T1ce, T2, FLAIR, reference
labels, reference lesion components, all 12 model predictions, and all 12
whole-tumor false-positive/false-negative overlays. Red denotes false positive
and cyan denotes false negative. Both PNG and SVG files are hash-indexed.

Run only after Gate H:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/analyze_q1q2_qualitative.py
```

The completion artifact records that selection occurred after evaluation,
cherry-picking was not performed, and no new inference occurred.
