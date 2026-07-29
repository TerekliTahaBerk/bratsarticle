#!/usr/bin/env python3
"""Audit exact phrase overlap and repetition without detector-evasion claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEWER = Path(
    "/Users/berk.terekli/.codex/attachments/"
    "52eb8b87-d478-43df-8594-de332b2ce5e0/pasted-text.txt"
)
DEFAULT_PLAN = Path(
    "/Users/berk.terekli/.codex/attachments/"
    "df4705a7-9f0a-4ad5-9087-1faa25f1d7b7/pasted-text.txt"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def _strip_nonprose(text: str) -> str:
    text = text.split("\n## References\n", maxsplit=1)[0]
    text = re.sub(r"^!\[.*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"\|.*\|", " ", text)
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]+`", " ", text)
    return text


def _ngrams(words: list[str], size: int) -> dict[tuple[str, ...], list[int]]:
    result: dict[tuple[str, ...], list[int]] = {}
    for index in range(len(words) - size + 1):
        gram = tuple(words[index : index + size])
        result.setdefault(gram, []).append(index)
    return result


def _sentences(text: str) -> list[str]:
    plain = re.sub(r"#+\s+", "", _strip_nonprose(text))
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", plain)
        if len(_words(part)) >= 4
    ]


def _overlaps(
    manuscript_words: list[str],
    reference_words: list[str],
    size: int,
    limit: int = 20,
) -> list[dict[str, object]]:
    manuscript = _ngrams(manuscript_words, size)
    reference = _ngrams(reference_words, size)
    shared = sorted(
        set(manuscript).intersection(reference),
        key=lambda gram: manuscript[gram][0],
    )
    return [
        {
            "phrase": " ".join(gram),
            "manuscript_positions": manuscript[gram],
            "reference_positions": reference[gram],
        }
        for gram in shared[:limit]
    ]


def _existing(paths: Iterable[Path]) -> list[Path]:
    return [path.resolve() for path in paths if path.exists() and path.is_file()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manuscript",
        type=Path,
        default=ROOT / "manuscript/final_manuscript.md",
    )
    parser.add_argument("--reference", action="append", type=Path, default=[])
    parser.add_argument("--ngram-size", type=int, default=12)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=ROOT / "reports/gate14_originality_audit.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT / "reports/gate14_originality_audit.md",
    )
    args = parser.parse_args()

    manuscript_path = args.manuscript.resolve()
    reference_paths = _existing([*args.reference, DEFAULT_REVIEWER, DEFAULT_PLAN])
    manuscript_text = manuscript_path.read_text(encoding="utf-8")
    manuscript_words = _words(_strip_nonprose(manuscript_text))
    sentences = _sentences(manuscript_text)
    sentence_lengths = [len(_words(sentence)) for sentence in sentences]

    per_reference = []
    for reference_path in reference_paths:
        reference_text = reference_path.read_text(encoding="utf-8", errors="replace")
        matches = _overlaps(
            manuscript_words,
            _words(_strip_nonprose(reference_text)),
            args.ngram_size,
        )
        per_reference.append(
            {
                "path": str(reference_path),
                "sha256": _sha256(reference_path),
                "exact_ngram_match_count": len(matches),
                "matches": matches,
            }
        )

    repeated_sentences = [
        {"sentence": sentence, "count": count}
        for sentence, count in Counter(sentences).most_common()
        if count > 1
    ]
    starts = Counter(
        " ".join(_words(sentence)[:4])
        for sentence in sentences
        if len(_words(sentence)) >= 4
    )
    repeated_starts = [
        {"opening": opening, "count": count}
        for opening, count in starts.most_common(20)
        if count >= 3
    ]
    longest_sentences = sorted(
        (
            {"word_count": len(_words(sentence)), "sentence": sentence}
            for sentence in sentences
        ),
        key=lambda item: int(item["word_count"]),
        reverse=True,
    )[:10]

    overlap_count = sum(int(item["exact_ngram_match_count"]) for item in per_reference)
    status = "pass" if overlap_count == 0 and not repeated_sentences else "review"
    report = {
        "status": status,
        "scope": (
            "Exact phrase overlap against user-supplied reviewer/planning text "
            "and internal prose repetition. This is not Turnitin and does not "
            "predict or evade AI-detection systems."
        ),
        "manuscript": {
            "path": str(manuscript_path),
            "sha256": _sha256(manuscript_path),
            "prose_word_count": len(manuscript_words),
            "sentence_count": len(sentences),
            "mean_sentence_words": (
                sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
            ),
            "median_sentence_words": (
                sorted(sentence_lengths)[len(sentence_lengths) // 2]
                if sentence_lengths
                else 0
            ),
        },
        "ngram_size": args.ngram_size,
        "references": per_reference,
        "exact_ngram_match_count": overlap_count,
        "repeated_sentences": repeated_sentences,
        "repeated_four_word_openings": repeated_starts,
        "longest_sentences": longest_sentences,
        "interpretation": (
            "A pass means no exact long-phrase overlap was found in the audited "
            "inputs and no full prose sentence was duplicated. It is not a "
            "guarantee about third-party similarity or authorship classifiers."
        ),
    }

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Gate 14 originality and prose audit",
        "",
        f"**Status:** {status.upper()}",
        "",
        report["scope"],
        "",
        f"- Prose words: {len(manuscript_words)}",
        f"- Sentences: {len(sentences)}",
        (
            "- Mean sentence length: "
            f"{report['manuscript']['mean_sentence_words']:.1f} words"
        ),
        f"- Exact {args.ngram_size}-word matches: {overlap_count}",
        f"- Repeated full sentences: {len(repeated_sentences)}",
        "",
        "## Reference checks",
        "",
    ]
    for item in per_reference:
        lines.append(
            f"- `{item['path']}`: {item['exact_ngram_match_count']} exact matches"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(report["interpretation"]),
            "",
        ]
    )
    args.markdown_output.write_text("\n".join(lines), encoding="utf-8")

    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
