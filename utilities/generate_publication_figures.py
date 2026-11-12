"""Regenerate all figures from cleaned_data.csv without re-running the analysis.

Run `python main.py` first; it writes cleaned_data.csv (and also generates
these figures).

Usage (from the repository root):
    python utilities/generate_publication_figures.py
"""

import os
import sys
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import config  # noqa: E402
from data_loader import load_and_prepare  # noqa: E402
from known_groups_validation import (  # noqa: E402
    generate_known_groups_plots, perform_matching, prepare_data,
)
from visualization import generate_all_domain_plots  # noqa: E402


def main():
    if not os.path.exists(config.CLEANED_DATA_FILE):
        sys.exit(f"{config.CLEANED_DATA_FILE} not found. Run 'python main.py' first.")

    df = load_and_prepare(config.CLEANED_DATA_FILE)
    print(f"Loaded {len(df)} researchers from {config.CLEANED_DATA_FILE}")

    generate_all_domain_plots(df)

    matched_df, _ = perform_matching(prepare_data(df), seed=config.RANDOM_SEED)
    print(f"\nMatched {len(matched_df)} pairs for the known-groups figures")
    generate_known_groups_plots(matched_df)

    print(f"\nFigures written to {config.FIGURES_DIR}/")


if __name__ == "__main__":
    main()
