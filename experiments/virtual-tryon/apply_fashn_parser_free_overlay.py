#!/usr/bin/env python3
"""Apply the Fit By Form parser-free flat-lay overlay to a pinned FASHN VTON checkout."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

UPSTREAM_COMMIT = "7c0f10af3f91ad4048fe9729c470a13ef905d25a"


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one upstream match, found {count}. "
            "Do not force this patch onto a different upstream version."
        )
    return source.replace(old, new, 1)


def patch_pipeline(source: str) -> str:
    source = replace_once(source, "from fashn_human_parser import CATEGORY_TO_BODY_COVERAGE, FashnHumanParser\n", "", "pipeline parser import")
    source = replace_once(
        source,
        '''from .preprocessing import (
    BODY_COVERAGE_TO_FASHN_LABELS,
    FASHN_LABELS_TO_IDS,
    AspectPreserveResize,
    ResizePad,
    create_clothing_agnostic_image,
    create_garment_image,
)
''',
        '''from .preprocessing import (
    AspectPreserveResize,
    ResizePad,
    create_clothing_agnostic_image,
    create_garment_image,
)
''',
        "pipeline preprocessing imports",
    )
    source = replace_once(
        source,
        "        self._setup_pose_model()\n        self._setup_hp_model()\n",
        "        self._setup_pose_model()\n        # The human parser is loaded only by legacy masked-image paths.\n        self.hp_model = None\n",
        "pipeline constructor",
    )
    source = replace_once(
        source,
        '''    def _setup_hp_model(self):
        """Load human parsing model."""
        self.logger.info("Loading FashnHumanParser")

        hp_device = "cuda" if self.device.type == "cuda" else "cpu"
        self.hp_model = FashnHumanParser(device=hp_device)

        self.logger.info("FashnHumanParser loaded")

''',
        '''    def _get_hp_model(self):
        """Load the optional human parser only when a masked-image path needs it."""
        if self.hp_model is None:
            from fashn_human_parser import FashnHumanParser

            self.logger.info("Loading optional FashnHumanParser")
            hp_device = "cuda" if self.device.type == "cuda" else "cpu"
            self.hp_model = FashnHumanParser(device=hp_device)
            self.logger.info("FashnHumanParser loaded")
        return self.hp_model

''',
        "lazy parser loader",
    )
    source = replace_once(
        source,
        "        # Set seed\n",
        '''        if category not in self.CATEGORY_TO_LABEL:
            raise ValueError(f"Unsupported garment category: {category!r}")

        # Set seed
''',
        "category validation",
    )
    source = replace_once(
        source,
        '''        # Human parsing
        person_seg_pred = self.hp_model.predict(person_image_np)
        garment_seg_pred = self.hp_model.predict(garment_image_np)

        # Get labels to segment based on category
        body_coverage = CATEGORY_TO_BODY_COVERAGE.get(category)
        labels_to_segment = BODY_COVERAGE_TO_FASHN_LABELS.get(body_coverage)
        labels_to_segment_indices = [FASHN_LABELS_TO_IDS[label] for label in labels_to_segment]

        # Create clothing-agnostic and garment images
        ca_image = create_clothing_agnostic_image(
            img_np=person_image_np.copy(),
            seg_pred=person_seg_pred.copy(),
            labels_to_segment_indices=labels_to_segment_indices.copy(),
            body_coverage=body_coverage,
            disable_masking=segmentation_free,
            logger=self.logger,
        )

        garment_image_processed = create_garment_image(
            img_np=garment_image_np,
            seg_pred=garment_seg_pred,
            labels_to_segment_indices=labels_to_segment_indices.copy(),
            disable_masking=garment_photo_type == "flat-lay",
        )

''',
        '''        # Parser-free MVP path: segmentation-free person + flat-lay product photo.
        # These are exact image copies; no human-parser import, model load, or predict call occurs.
        need_person_parse = not segmentation_free
        need_garment_parse = garment_photo_type != "flat-lay"
        ca_image = person_image_np.copy()
        garment_image_processed = garment_image_np.copy()

        if need_person_parse or need_garment_parse:
            from fashn_human_parser import (
                BODY_COVERAGE_TO_LABELS,
                CATEGORY_TO_BODY_COVERAGE,
                LABELS_TO_IDS,
            )

            hp_model = self._get_hp_model()
            body_coverage = CATEGORY_TO_BODY_COVERAGE[category]
            labels_to_segment = BODY_COVERAGE_TO_LABELS[body_coverage]
            labels_to_segment_indices = [LABELS_TO_IDS[label] for label in labels_to_segment]

            if need_person_parse:
                person_seg_pred = hp_model.predict(person_image_np)
                ca_image = create_clothing_agnostic_image(
                    img_np=person_image_np.copy(),
                    seg_pred=person_seg_pred.copy(),
                    labels_to_segment_indices=labels_to_segment_indices.copy(),
                    body_coverage=body_coverage,
                    disable_masking=False,
                    logger=self.logger,
                )

            if need_garment_parse:
                garment_seg_pred = hp_model.predict(garment_image_np)
                garment_image_processed = create_garment_image(
                    img_np=garment_image_np.copy(),
                    seg_pred=garment_seg_pred,
                    labels_to_segment_indices=labels_to_segment_indices.copy(),
                    disable_masking=False,
                )

''',
        "parser inference block",
    )
    return source


def patch_agnostic(source: str) -> str:
    source = replace_once(source, "from fashn_human_parser import BODY_COVERAGE_TO_LABELS, IDENTITY_LABELS, LABELS_TO_IDS\n", "", "agnostic parser import")
    source = replace_once(
        source,
        '''# Re-export constants from fashn_human_parser for convenience
FASHN_LABELS_TO_IDS = LABELS_TO_IDS
BODY_COVERAGE_TO_FASHN_LABELS = BODY_COVERAGE_TO_LABELS
IDENTITY_FASHN_LABELS = tuple(IDENTITY_LABELS)


''',
        "",
        "agnostic eager parser constants",
    )
    source = replace_once(
        source,
        "from .masks import asymmetric_dilate_mask, create_bounded_mask, create_contour_following_mask, dilate_mask\n",
        '''from .masks import asymmetric_dilate_mask, create_bounded_mask, create_contour_following_mask, dilate_mask


def __getattr__(name: str):
    """Keep legacy constants importable, but resolve them only when requested."""
    if name == "FASHN_LABELS_TO_IDS":
        from fashn_human_parser import LABELS_TO_IDS
        return LABELS_TO_IDS
    if name == "BODY_COVERAGE_TO_FASHN_LABELS":
        from fashn_human_parser import BODY_COVERAGE_TO_LABELS
        return BODY_COVERAGE_TO_LABELS
    if name == "IDENTITY_FASHN_LABELS":
        from fashn_human_parser import IDENTITY_LABELS
        return tuple(IDENTITY_LABELS)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
''',
        "lazy agnostic compatibility exports",
    )
    source = replace_once(
        source,
        '''    if disable_masking:
        return img_np

    # Scale parameters based on image height
''',
        '''    if disable_masking:
        return img_np

    # Import only when legacy masking is requested; flat-lay MVP never reaches this path.
    try:
        from fashn_human_parser import IDENTITY_LABELS, LABELS_TO_IDS
    except ImportError as exc:
        raise ImportError(
            "Masked clothing-agnostic preprocessing requires the optional extra: "
            "pip install -e '.[parser]'"
        ) from exc

    # Scale parameters based on image height
''',
        "lazy agnostic parser import",
    )
    source = replace_once(source, "    labels_ids_dict = FASHN_LABELS_TO_IDS.copy()\n", "    labels_ids_dict = LABELS_TO_IDS.copy()\n", "lazy parser label lookup")
    source = replace_once(source, "    identity_ids = [labels_ids_dict[label] for label in IDENTITY_FASHN_LABELS]\n", "    identity_ids = [labels_ids_dict[label] for label in IDENTITY_LABELS]\n", "lazy parser identity labels")
    return source


def patch_preprocessing_init(source: str) -> str:
    source = replace_once(
        source,
        '''from .agnostic import (
                       BODY_COVERAGE_TO_FASHN_LABELS,
                       FASHN_LABELS_TO_IDS,
                       create_clothing_agnostic_image,
                       create_garment_image,
)
''',
        "from .agnostic import create_clothing_agnostic_image, create_garment_image\n",
        "preprocessing imports",
    )
    source = replace_once(
        source,
        "from .transforms import AspectPreserveResize, PadToShape, ResizePad\n",
        '''from .transforms import AspectPreserveResize, PadToShape, ResizePad

from . import agnostic as _agnostic


def __getattr__(name: str):
    if name in {"FASHN_LABELS_TO_IDS", "BODY_COVERAGE_TO_FASHN_LABELS"}:
        return getattr(_agnostic, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
''',
        "lazy preprocessing compatibility exports",
    )
    return source


def patch_download_weights(source: str) -> str:
    source = replace_once(
        source,
        "    - FashnHumanParser weights (auto-cached by HuggingFace)\n",
        "    - Optional FashnHumanParser weights, only with --with-human-parser\n",
        "download weights description",
    )
    source = replace_once(
        source,
        '''    parser.add_argument(
        "--weights-dir",
        type=str,
        required=True,
        help="Directory to save model weights",
    )
    args = parser.parse_args()
''',
        '''    parser.add_argument(
        "--weights-dir",
        type=str,
        required=True,
        help="Directory to save model weights",
    )
    parser.add_argument(
        "--with-human-parser",
        action="store_true",
        help="Also download optional FashnHumanParser weights for masked-image preprocessing.",
    )
    args = parser.parse_args()
''',
        "optional human-parser flag",
    )
    source = replace_once(
        source,
        '''    print()
    download_human_parser()

    print(f"""
''',
        '''    print()
    if args.with_human_parser:
        download_human_parser()
    else:
        print("Skipping optional Human Parser weights; flat-lay + segmentation-free mode does not need them.")

    print(f"""
''',
        "optional human-parser download",
    )
    return source


def patch_pyproject(source: str) -> str:
    source = replace_once(source, '    "fashn-human-parser>=0.1.1",\n', "", "required parser dependency")
    source = replace_once(
        source,
        "[project.optional-dependencies]\ndev = [\n",
        '[project.optional-dependencies]\nparser = [\n    "fashn-human-parser>=0.1.1",\n]\ndev = [\n',
        "optional parser extra",
    )
    return source


def apply(root: Path) -> None:
    root = root.resolve()
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Target must be a Git checkout of fashn-AI/fashn-vton-1.5.") from exc
    if revision != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"Refusing to patch upstream revision {revision}; expected {UPSTREAM_COMMIT}. "
            "Review upstream changes and update the patch deliberately."
        )

    targets = {
        root / "src/fashn_vton/pipeline.py": patch_pipeline,
        root / "src/fashn_vton/preprocessing/agnostic.py": patch_agnostic,
        root / "src/fashn_vton/preprocessing/__init__.py": patch_preprocessing_init,
        root / "scripts/download_weights.py": patch_download_weights,
        root / "pyproject.toml": patch_pyproject,
    }
    updated: dict[Path, str] = {}
    for path, transform in targets.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        updated[path] = transform(path.read_text(encoding="utf-8"))

    # Write only after every source file passed all exact-match checks.
    for path, content in updated.items():
        path.write_text(content, encoding="utf-8")
        print(f"patched: {path.relative_to(root)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fashn_repo", type=Path, help="Path to the pinned FASHN VTON 1.5 Git checkout")
    args = parser.parse_args()
    apply(args.fashn_repo)


if __name__ == "__main__":
    main()
