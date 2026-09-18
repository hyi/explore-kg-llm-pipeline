from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.presentation_readiness import (
    DEFAULT_ANALYSIS_OUTPUT_DIR,
    DEFAULT_PRESENTATION_DIR,
    write_presentation_package,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build group-presentation and demo-readiness artifacts.")
    parser.add_argument(
        "--analysis-output-dir",
        type=Path,
        default=DEFAULT_ANALYSIS_OUTPUT_DIR,
        help="Directory containing embedding-comparison CSV/JSON/HTML artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PRESENTATION_DIR,
        help="Presentation package destination.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package = write_presentation_package(
        source_dir=args.analysis_output_dir,
        output_dir=args.output_dir,
    )
    print(f"Wrote presentation package to {package.output_dir}")
    print(f"Generated {len(package.generated_files)} file(s).")
    if package.missing_source_artifacts:
        print("Missing source artifacts:")
        for artifact in package.missing_source_artifacts:
            print(f"- {artifact}")


if __name__ == "__main__":
    main()
