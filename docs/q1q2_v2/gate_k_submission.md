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
