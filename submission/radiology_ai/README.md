# Radiology: Artificial Intelligence submission package

This directory contains source templates and a guarded builder for the
conditional Radiology: Artificial Intelligence Original Research route.

`scripts/build_q1q2_submission_package.py --mode preview` creates visibly
marked pre-results DOCX/PDF previews. They are not submission files and cannot
be mistaken for completed scientific evidence.

`--mode final` is intentionally blocked unless:

1. Gate H external evaluation is complete and passing;
2. Gate I reproducibility is complete and passing;
3. Gate J claim rendering is complete and passing;
4. all claim tokens are resolved;
5. `author_metadata.template.json` has been copied to
   `author_metadata.confirmed.json` and every author-dependent declaration is
   complete;
6. the current journal-policy audit and all format limits pass.

The preview and final modes never insert hand-entered result values.
