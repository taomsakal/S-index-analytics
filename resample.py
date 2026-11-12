"""
Resampling to correct for prolific-author sampling bias.

OpenAlex cannot draw a uniform random sample of researchers who have at least
one dataset, so the raw sample is drawn as random dataset -> random author of
that dataset. This over-represents authors with many datasets and
under-represents authors on large teams:

- an author with N datasets is ~N times more likely to be reached;
- an author on a team of size T is selected with probability 1/T.

Inverse probability weighting with weight = team_size / num_datasets corrects
for both. If a column is missing, its value is treated as 1.

The resampled size is the Kish effective sample size, ESS = (sum w)^2 / sum(w^2),
which is the number of uniform draws the weighted sample is worth.
"""

import pandas as pd
import numpy as np
from typing import Optional

from data_loader import load_raw_data, filter_for_resampling


def calculate_sampling_weights(
    df: pd.DataFrame,
    num_datasets_col: str = "num_datasets",
    team_size_col: str = "team_size",
    return_raw: bool = False,
) -> np.ndarray:
    """
    Calculate inverse probability weights to correct for sampling bias.

    Weight formula: team_size / num_datasets
    - Dividing by num_datasets down-weights prolific authors
    - Multiplying by team_size up-weights authors from larger teams
      (who were less likely to be selected from their dataset)

    Args:
        df: DataFrame with sampling metadata
        num_datasets_col: Column name for number of datasets per author
        team_size_col: Column name for team size of observed dataset
        return_raw: If True, return unnormalized weights (for ESS calculation)

    Returns:
        Array of sampling weights (normalized to sum to 1, unless return_raw=True)
    """
    if num_datasets_col in df.columns:
        num_datasets = df[num_datasets_col].fillna(1).clip(lower=1)
    else:
        print(f"Warning: '{num_datasets_col}' column not found. Using 1 for all rows.")
        num_datasets = pd.Series(1, index=df.index)

    if team_size_col in df.columns:
        team_size = df[team_size_col].fillna(1).clip(lower=1)
    else:
        print(f"Warning: '{team_size_col}' column not found. Using 1 for all rows.")
        team_size = pd.Series(1, index=df.index)

    weights = team_size.astype(float) / num_datasets.astype(float)

    if return_raw:
        return weights.values

    weights = weights / weights.sum()
    return weights.values


def calculate_effective_sample_size(
    df: pd.DataFrame,
    num_datasets_col: str = "num_datasets",
    team_size_col: str = "team_size",
) -> int:
    """
    Calculate the Effective Sample Size (ESS) for inverse probability weighting.

    ESS = (Σw_i)² / Σ(w_i²)

    This is the Kish effective sample size, a standard measure in survey statistics.
    It represents how many equivalent uniform random samples the weighted data
    is worth, accounting for the loss of precision due to weight variability.

    Args:
        df: DataFrame with sampling metadata
        num_datasets_col: Column name for number of datasets per author
        team_size_col: Column name for team size of observed dataset

    Returns:
        Effective sample size (integer, capped at len(df))
    """
    raw_weights = calculate_sampling_weights(
        df, num_datasets_col, team_size_col, return_raw=True
    )

    sum_w = raw_weights.sum()
    sum_w_sq = (raw_weights ** 2).sum()

    ess = (sum_w ** 2) / sum_w_sq

    return min(int(ess), len(df))


def resample_data(
    df: pd.DataFrame,
    seed: int,
    n_samples: Optional[int] = None,
    num_datasets_col: str = "num_datasets",
    team_size_col: str = "team_size",
    with_replacement: bool = False,
) -> pd.DataFrame:
    """
    Resample data using inverse probability weighting to correct bias.

    Args:
        df: Raw sampled data with num_datasets and team_size columns
        seed: Random seed for reproducibility
        n_samples: Number of samples to draw. If None, uses len(df).
                   For without-replacement sampling, capped at len(df).
        num_datasets_col: Column name for number of datasets per author
        team_size_col: Column name for team size
        with_replacement: If True, sample with replacement. If False,
                         samples without replacement (default).

    Returns:
        Resampled DataFrame with corrected sampling distribution
    """
    if len(df) == 0:
        return df.copy()

    rng = np.random.default_rng(seed)

    weights = calculate_sampling_weights(
        df, num_datasets_col=num_datasets_col, team_size_col=team_size_col
    )

    if n_samples is None:
        n_samples = len(df)

    if not with_replacement:
        # Cap at dataset size for without-replacement sampling
        n_samples = min(n_samples, len(df))

    indices = rng.choice(
        len(df),
        size=n_samples,
        replace=with_replacement,
        p=weights,
    )

    return df.iloc[indices].reset_index(drop=True)


def resample_dataframe_to_file(
    df: pd.DataFrame,
    output_path: str,
    seed: int,
    n_samples: Optional[int] = None,
    num_datasets_col: str = "num_datasets",
    team_size_col: str = "team_size",
    with_replacement: bool = False,
) -> pd.DataFrame:
    """
    Resample a DataFrame with bias correction and save to file.

    Args:
        df: Input DataFrame (should already be filtered)
        output_path: Path to save resampled CSV
        seed: Random seed for reproducibility
        n_samples: Number of samples (None = same as input size)
        num_datasets_col: Column name for number of datasets
        team_size_col: Column name for team size
        with_replacement: Whether to sample with replacement

    Returns:
        Resampled DataFrame
    """
    original_size = len(df)

    resampled_df = resample_data(
        df,
        seed=seed,
        n_samples=n_samples,
        num_datasets_col=num_datasets_col,
        team_size_col=team_size_col,
        with_replacement=with_replacement,
    )

    resampled_df.to_csv(output_path, index=False)

    print(f"  Resampled {original_size} -> {len(resampled_df)} rows (seed={seed})")
    print(f"  Saved to: {output_path}")

    return resampled_df


def resample_to_file(
    input_path: str,
    output_path: str,
    seed: int,
    n_samples: Optional[int] = None,
    num_datasets_col: str = "num_datasets",
    team_size_col: str = "team_size",
    with_replacement: bool = False,
) -> pd.DataFrame:
    """
    Load raw data, apply the pre-resampling filters, resample with bias
    correction, and save to file.

    Args:
        input_path: Path to raw sampled CSV or directory of CSV files
        output_path: Path to save resampled CSV
        seed: Random seed for reproducibility
        n_samples: Number of samples (None = same as filtered input size)
        num_datasets_col: Column name for number of datasets
        team_size_col: Column name for team size
        with_replacement: Whether to sample with replacement

    Returns:
        Resampled DataFrame
    """
    df = filter_for_resampling(load_raw_data(input_path), verbose=True)
    return resample_dataframe_to_file(
        df,
        output_path,
        seed=seed,
        n_samples=n_samples,
        num_datasets_col=num_datasets_col,
        team_size_col=team_size_col,
        with_replacement=with_replacement,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python resample.py <input.csv|input_dir> <output.csv> [seed]")
        print("Example: python resample.py data resampled_data.csv 42")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42

    resample_to_file(input_file, output_file, seed=seed)
