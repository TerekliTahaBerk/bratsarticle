# Conditional target-journal rationale

Verification date: 2026-07-30

## Decision state

No final journal is selected before the external result. The current primary
route is **Medical Image Analysis** only if the capacity- and compute-matched
findings support a transferable methodological lesson. The alternate route is
**Radiology: Artificial Intelligence, Original Research** if independent
population/acquisition shift, failure analysis, and radiology-facing
interpretation become the stronger contribution.

The repository does not claim that either journal will accept the work and
does not label the study Q1/Q2-ready before Gates C–J pass. Current quartile,
APC, and institutional agreement are not inferred from unofficial pages.

## Route A — Medical Image Analysis

The journal’s official ScienceDirect page states that it disseminates new
research in medical and biological image analysis, emphasizing computer
vision and related methods applied to biomedical imaging:
[Medical Image Analysis — Aims and Scope](https://www.sciencedirect.com/journal/medical-image-analysis).

This route is appropriate only if the results answer a broader methodological
question, for example:

- apparent component gains materially change after parameter or compute
  matching;
- seed-patient uncertainty changes the scientific conclusion;
- conclusions differ between development and independent external testing;
- lesion-level or cost analyses expose a reproducible limitation of common
  overlap-only benchmarking.

This route is not justified by a small reimplementation gain, a model
tournament, or a single-dataset ranking. The live journal Guide for Authors,
article type, length, figure/table limits, data/code requirements, and
publication charges must be rechecked immediately before submission because
the accessible official scope page does not establish all of those fields.

Elsevier’s current general policy encourages appropriate research-data
sharing and data-availability statements:
[Elsevier research-data policy](https://www.elsevier.com/about/policies-and-standards/research-data).
Raw BraTS data will not be redistributed; code, hashes, manifests, permitted
derived artifacts, and setup instructions will be made available.

## Route B — Radiology: Artificial Intelligence

The current official instructions describe the journal as covering
machine-learning and AI applications in imaging and invite high-quality work
that demonstrates novel applications or methodologies:
[Radiology: Artificial Intelligence — Author Instructions](https://pubs.rsna.org/page/ai/author-instructions).

For Original Research, the verified current limits and required elements are:

- main text no more than 3000 words from Introduction through Discussion;
- structured abstract no more than 250 words;
- no more than 35 references;
- no more than six figures and four tables in the main paper;
- a one- or two-sentence summary statement;
- up to three key points containing summary data;
- an applicable reporting checklist for human-subject imaging work;
- a separate cover letter and full title page;
- double-anonymized review and an anonymized manuscript file.

The same instructions state that initial submission is free and that
non-open-access publication has no publication charge; current open-access
cost and any institutional agreement still require author-side confirmation.
They also require data-sharing information and point AI imaging manuscripts
to the updated CLAIM checklist:
[RSNA CLAIM resources](https://pubs.rsna.org/page/ai/claim).

This route becomes preferable if the strongest contribution is the external
BraTS-Africa evaluation, scanner/institution subgroup behavior, lesion-level
failure characterization, and cautious implications for radiology AI
evaluation. The manuscript must still avoid claiming clinical utility,
universal transportability, or prospective effectiveness.

## Result-dependent selection rule

1. Select Medical Image Analysis only if the frozen analyses yield a clear
   methodologically transferable lesson beyond this exact model/dataset pair.
2. Select Radiology: Artificial Intelligence Original Research if the
   independent external-testing and failure-analysis message is stronger and
   the main paper can be reduced to its verified limits without hiding
   required evidence.
3. Select neither if convergence, equal-seed completion, external inference,
   claim audit, or scientific contribution is inadequate. Do not lower the
   evidentiary language to force a Q1/Q2 submission.

## Package implications

The current long-form manuscript and supplement are journal-neutral evidence
sources. If Route B is selected, the main document must be compressed to the
verified word/reference/figure/table limits, with detailed models, loss
equations, convergence, lesion endpoints, subgroups, resource tables, and
qualitative panels moved to the supplement without removing the primary
effect, uncertainty, cohort flow, and limitations.

For either route, final submission remains blocked by:

- real Gate H–J outputs;
- final journal-policy recheck;
- author identities, affiliations, ORCID identifiers, and CRediT roles;
- ethics/waiver interpretation;
- funding, conflicts, acknowledgements, and correspondence details;
- journal-specific generative-AI disclosure approved by the authors;
- final release/checkpoint-distribution decision;
- visually verified DOCX/PDF and final page/line references.

## Generative-AI disclosure boundary

AI-assisted writing must be disclosed according to the selected journal’s
current policy and reviewed by all authors. It cannot be listed as an author.
The project’s provenance controls are not AI detectors and are not described
as a way to evade overlap or authorship checks. Elsevier’s current author
policy requires an explicit declaration when generative AI or AI-assisted
tools were used in manuscript preparation:
[Elsevier policy for AI-assisted writing](https://www.elsevier.com/en-gb/about/policies-and-standards/the-use-of-generative-ai-and-ai-assisted-technologies-in-writing-for-elsevier).
