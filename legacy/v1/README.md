# Legacy v1 — bounded 2D component study

This directory is a logical, immutable pointer to the completed Gate 0–14
study. It does not duplicate or rewrite its artifacts.

## Frozen identity

- Source commit: `ab60a79d8a49a2fe1adb000546f653485796bab1`
- Annotated tag: `v1-bounded-2d-component-study`
- Tag target: `ab60a79d8a49a2fe1adb000546f653485796bab1`
- Completion commit date: `2026-07-30T00:16:00+03:00`
- Legacy manuscript status: bounded internal evidence, not external validation

The exact historical files must be obtained from the tag, for example:

```bash
git show v1-bounded-2d-component-study:reports/gate14_completion.json
git archive --format=tar.gz --output=legacy-v1.tar.gz \
  v1-bounded-2d-component-study
```

## Integrity anchors

| Artifact at the frozen tag | SHA-256 |
|---|---|
| `reports/tracked_artifact_manifest.json` | `1b1fe7090a9f367c0a71796e63a9b508e3b2764afe17df3a7c973fd0e87cc0b8` |
| `artifacts/test_access_log.jsonl` | `2b3734c9f68f152a7f4243d7461b6b67ba45e34fb7e11070bac983e20218edbc` |
| `reports/gate14_completion.json` | `426de18b6db38b5b71bf0c0e6feff18b98b4bd9e2bddb7d9726d6fe3b91b3932` |
| `manuscript/final_manuscript.pdf` | `d018d41a49919255b0893eb4102ca5b548b499e689fbf0c7b1bb1a6953c88413` |
| `manuscript/final_manuscript.docx` | `5f44d179aff8e5c293ad17e0d6e5708dd231f9aa5ab0835c7d7bd810759981a3` |

## Scientific boundary

The 74-patient internal held-out subset and its recorded results are legacy
evidence only. The v2 study must not evaluate a new model, select a model,
select a loss, tune a threshold, tune post-processing, or make a new
confirmatory claim on that subset. No v1 result may be presented as a v2
external result.

RES and WC remain previously published BU-Net components attributed to Rehman
et al.; they are not new components introduced by this repository.
