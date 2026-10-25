#!/usr/bin/env python3
"""
corpus.py — the single definition of "the text of a paper".

Every downstream script imports canonical_lines() from here. The reason is
narrow and important: boundaries.csv stores INTEGER LINE INDICES, so if the
boundary tool and the extraction tool built their line lists even slightly
differently, the recorded boundaries would silently point at the wrong lines.
Defining it once makes that drift impossible.

Reproducibility contract
------------------------
Extraction is pinned to an exact MuPDF build, not merely an exact version
string. PyMuPDF ships MuPDF inside its wheel, so `pip install
pymupdf==1.27.2.2` yields the same engine on every platform; require_engine()
asserts it and refuses to run otherwise. A replicator who sees that error has a
one-line fix, which is far better than silently producing slightly different
text and then failing a checksum for an unrelated-looking reason.
"""

from pathlib import Path

PINNED_PYMUPDF = "1.27.2.2"
PINNED_MUPDF = "1.27.2"


def require_engine() -> tuple[str, str]:
    """Abort unless the pinned extraction engine is present."""
    import pymupdf
    bind, fitz = pymupdf.VersionBind, pymupdf.VersionFitz
    if (bind, fitz) != (PINNED_PYMUPDF, PINNED_MUPDF):
        raise SystemExit(
            f"FATAL: extraction engine mismatch.\n"
            f"  found  pymupdf {bind} / MuPDF {fitz}\n"
            f"  need   pymupdf {PINNED_PYMUPDF} / MuPDF {PINNED_MUPDF}\n"
            f"  fix    pip install pymupdf=={PINNED_PYMUPDF}\n"
            f"Text extraction is version-sensitive; a different engine would "
            f"produce a different corpus and invalidate the recorded checksums."
        )
    return bind, fitz


def canonical_lines(pdf: Path) -> list[str]:
    """The canonical line list for a paper. Line N means lines[N] everywhere.

    Deliberately plain: page order, default text mode, no reflow, no sorting,
    no de-hyphenation. Every transformation is a choice that would have to be
    reproduced exactly, so the raw stream is the stable thing to anchor
    boundaries against. De-hyphenation happens later, after slicing, where it
    cannot shift a line index.
    """
    import pymupdf
    with pymupdf.open(pdf) as doc:
        return "\n".join(page.get_text() for page in doc).split("\n")


def canonical_text(lines: list[str]) -> str:
    """Lines -> the string actually shown to a rater."""
    import re
    s = "\n".join(lines).replace("-\n", "")
    s = re.sub(r"[ \t]*\n[ \t]*", "\n", s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()
