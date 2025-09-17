
"""
Command-line interface for running the Aux+Auto DP pipeline.
"""
from __future__ import annotations

import argparse
from .pipeline import reproduce_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Aux+Auto DP (AO+ROE) pipeline")
    parser.add_argument("--csv", required=True, help="Path to team CSV")
    parser.add_argument("--out", default=".", help="Output directory")
    parser.add_argument("--no-calibration", action="store_true", help="Disable per-team calibration")
    args = parser.parse_args()

    reproduce_everything(args.csv, output_dir=args.out, do_calibration=not args.no_calibration)


if __name__ == "__main__":
    main()
