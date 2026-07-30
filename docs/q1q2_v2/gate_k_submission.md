# Q1/Q2 v2 Gate K submission contract

Gate K cannot pass before Gates H–J, final layout, author confirmation, and
journal-policy verification. The result-independent CLAIM 2024 template can be
audited now:

```bash
.venv/bin/python scripts/audit_q1q2_submission_templates.py
```

The audit requires all 44 CLAIM 2024 items exactly once and in order, a
controlled status for every item, and the existence of the declared
repository evidence. The pre-results disposition is deliberately mixed:
items may be Yes pending page/lines, Partial pending real results, No, or Not
Applicable. Missing rater-variability evidence, funding information, clinical
trial registration, explainability, and classification-specific diagnostic
metrics are not converted into false compliance.

Finalization requires:

- a Gate J-rendered manuscript, supplement, and reviewer response;
- final pagination and real page/line references;
- no unresolved result tokens;
- confirmed authors, affiliations, CRediT roles, ethics, funding, conflicts,
  acknowledgements, correspondence, and disclosure language;
- current target-journal instructions and policy checks;
- final release identifiers and checkpoint-distribution decision;
- rendered and visually verified DOCX/PDF submission artifacts.

The tracked Markdown checklist is a source template, not the final signed
author declaration or submission PDF.

## Pre-results submission preview

A guarded Radiology: Artificial Intelligence package is generated at
`submission/generated/pre_results/`. It contains:

- anonymized manuscript;
- full title page;
- cover letter;
- supplement;
- response to the reviewer;
- CLAIM 2024 checklist; and
- data/code availability statement.

Every document is available as Markdown, DOCX, and PDF. The DOCX files use
Letter paper, 1-inch margins, Arial 11-point body text, double spacing, left
alignment, and no manuscript page-number field. The source audit records 203
abstract words, 2,557 Introduction-through-Discussion words, and 16
references. All 52 rendered pages passed visual inspection; the checklist
tables use fixed geometry, explicit page headers, intact rows, and no merged
cells.

This directory is deliberately **not a submission package**. Its manifest has
status `preview_not_for_submission`, all result and author-dependent fields
remain conspicuously marked, and every page begins with a red pre-results
warning where applicable. Final mode hard-fails unless Gates H, I, and J pass,
author metadata is confirmed, the CLAIM checklist has real page/line
references, no pending marker remains, and the final supplement is within the
12-page limit.
