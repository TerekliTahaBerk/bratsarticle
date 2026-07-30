#!/usr/bin/env python3
"""Run frozen post-evaluation qualitative selection and rendering."""

from __future__ import annotations

import json

from bratsarticle.analysis.q1q2_qualitative import analyze_q1q2_qualitative


def main() -> int:
    report = analyze_q1q2_qualitative()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
