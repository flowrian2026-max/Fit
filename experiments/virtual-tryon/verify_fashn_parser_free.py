#!/usr/bin/env python3
"""Static verification for an already-patched FASHN VTON checkout."""
from __future__ import annotations

import ast
import sys
from pathlib import Path


def has_top_level_parser_import(source: str) -> bool:
    tree = ast.parse(source)
    return any(
        isinstance(node, ast.ImportFrom) and node.module == "fashn_human_parser"
        for node in tree.body
    )


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python verify_fashn_parser_free.py /path/to/fashn-vton-1.5", file=sys.stderr)
        return 2

    root = Path(sys.argv[1]).resolve()
    files = {
        "pipeline": root / "src/fashn_vton/pipeline.py",
        "agnostic": root / "src/fashn_vton/preprocessing/agnostic.py",
        "init": root / "src/fashn_vton/preprocessing/__init__.py",
        "pyproject": root / "pyproject.toml",
        "download": root / "scripts/download_weights.py",
    }
    for path in files.values():
        if not path.is_file():
            print(f"missing file: {path}", file=sys.stderr)
            return 2

    sources = {key: path.read_text(encoding="utf-8") for key, path in files.items()}
    for key, path in files.items():
        if key != "pyproject":
            ast.parse(sources[key], filename=str(path))

    p, a, i, pyproject, download = (
        sources["pipeline"], sources["agnostic"], sources["init"],
        sources["pyproject"], sources["download"],
    )
    checks = {
        "pipeline has no eager Human Parser import": not has_top_level_parser_import(p),
        "agnostic has no eager Human Parser import": not has_top_level_parser_import(a),
        "preprocessing package has no eager Human Parser import": not has_top_level_parser_import(i),
        "weight downloader has no eager Human Parser import": not has_top_level_parser_import(download),
        "lazy parser loader exists": "def _get_hp_model(self):" in p,
        "flat-lay + segmentation-free route bypasses parsing": (
            'need_garment_parse = garment_photo_type != "flat-lay"' in p
            and "need_person_parse = not segmentation_free" in p
            and "if need_person_parse or need_garment_parse:" in p
        ),
        "masked path still uses lazy parser labels": (
            "from fashn_human_parser import IDENTITY_LABELS, LABELS_TO_IDS" in a
            and "labels_ids_dict = LABELS_TO_IDS.copy()" in a
            and "for label in IDENTITY_LABELS" in a
        ),
        "legacy parser constants resolve lazily": (
            "def __getattr__(name: str):" in a
            and "def __getattr__(name: str):" in i
            and "return getattr(_agnostic, name)" in i
        ),
        "parser is not a core dependency": (
            '"fashn-human-parser>=0.1.1"' not in pyproject.split("[project.optional-dependencies]")[0]
        ),
        "parser is an optional extra": 'parser = [\n    "fashn-human-parser>=0.1.1",' in pyproject,
        "parser weight download is opt-in": "--with-human-parser" in download and "if args.with_human_parser:" in download,
    }
    failures = [name for name, passed in checks.items() if not passed]
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
    if failures:
        print(f"\n{len(failures)} check(s) failed.", file=sys.stderr)
        return 1
    print("\nAll parser-free overlay checks passed. This is a static check, not an image-quality or GPU test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
