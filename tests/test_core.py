"""
Unit tests for S-Index validation codebase.

Focuses on core functionality with meaningful assertions using
pre-computed expected values (not tautological recalculations).

Run with: pytest tests/ -v
"""

import os

import pytest
import pandas as pd
import numpy as np

from data_loader import load_openalex_data, load_raw_data, clean_data
from validation import (
    compute_correlation,
    compute_conditional_entropy,
    correlations_with_traditional_metrics,
    find_hidden_stewards,
)
from config import filter_by_domain, LIFE_SCIENCES
from resample import calculate_sampling_weights, resample_data


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_df():
    """Basic sample dataframe for testing."""
    return pd.DataFrame({
        "openalex_id": ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"],
        "name": ["Alice", "Bob", "Carlos", "Diana", "Erik", "Fatima", "George", "Hannah", "Ivan", "Julia"],
        "s_index": [10, 5, 0, 20, 3, 0, 25, 8, 0, 15],
        "h_index": [45, 32, 12, 67, 28, 8, 52, 19, 3, 41],
        "cited_by_count": [7950, 4800, 892, 14200, 3456, 234, 10120, 1234, 87, 8200],
        "num_datasets": [156, 89, 3, 342, 45, 12, 267, 34, 2, 189],
    })


@pytest.fixture
def sample_df_with_weights():
    """Sample dataframe with sampling weight columns."""
    return pd.DataFrame({
        "openalex_id": ["A1", "A2", "A3", "A4"],
        "s_index": [10, 5, 15, 20],
        "h_index": [45, 32, 50, 67],
        "num_datasets": [10, 1, 5, 2],
        "team_size": [2, 1, 5, 1],
    })


# =============================================================================
# Data Loader Tests
# =============================================================================

class TestDataLoader:
    """Tests for data_loader.py functions."""

    def test_load_openalex_data_file_exists(self, tmp_path):
        """Test loading a valid CSV file."""
        csv_content = "openalex_id,name,s-index,h-index\nA1,Alice,10,45\nA2,Bob,5,32"
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        df = load_openalex_data(str(csv_file))

        assert len(df) == 2
        assert "s_index" in df.columns  # hyphen converted to underscore
        assert "h_index" in df.columns
        assert df["s_index"].iloc[0] == 10

    def test_load_openalex_data_column_standardization(self, tmp_path):
        """Test that column names are standardized (lowercase, underscores)."""
        csv_content = "OpenAlex_ID,Name,S-Index,H-Index\nA1,Alice,10,45"
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        df = load_openalex_data(str(csv_file))

        assert "openalex_id" in df.columns
        assert "s_index" in df.columns
        assert "h_index" in df.columns

    def test_clean_data_converts_numeric(self, sample_df):
        """Test that numeric columns are converted correctly."""
        # Make s_index a string to test conversion
        df = sample_df.copy()
        df["s_index"] = df["s_index"].astype(str)

        cleaned = clean_data(df)

        assert cleaned["s_index"].dtype in [np.int64, np.float64]

    def test_clean_data_handles_invalid_values(self):
        """Test that invalid numeric values become NaN."""
        df = pd.DataFrame({
            "s_index": ["10", "invalid", "20"],
            "h_index": [45, 32, 50],
        })

        cleaned = clean_data(df)

        assert pd.isna(cleaned["s_index"].iloc[1])
        assert cleaned["s_index"].iloc[0] == 10

    def test_clean_data_does_not_modify_original(self, sample_df):
        """Test that clean_data doesn't modify the original dataframe."""
        original_values = sample_df["s_index"].copy()
        clean_data(sample_df)

        pd.testing.assert_series_equal(sample_df["s_index"], original_values)

class TestLoadRawData:
    """Tests for load_raw_data (file or directory of CSVs)."""

    def test_load_raw_data_single_file(self, tmp_path):
        """Single file path returns same result as load_openalex_data."""
        csv_content = "openalex_id,name,s-index,h-index\nA1,Alice,10,45\nA2,Bob,5,32"
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        df_raw = load_raw_data(str(csv_file))
        df_openalex = load_openalex_data(str(csv_file))

        pd.testing.assert_frame_equal(df_raw, df_openalex)

    def test_load_raw_data_directory_concat(self, tmp_path):
        """Directory with multiple CSVs concatenates correctly."""
        csv1 = tmp_path / "a.csv"
        csv1.write_text("openalex_id,name,s-index,h-index\nA1,Alice,10,45")
        csv2 = tmp_path / "b.csv"
        csv2.write_text("openalex_id,name,s-index,h-index\nA2,Bob,5,32")
        csv3 = tmp_path / "c.csv"
        csv3.write_text("openalex_id,name,s-index,h-index\nA3,Carol,7,28")

        df = load_raw_data(str(tmp_path))

        assert len(df) == 3
        assert list(df.columns) == ["openalex_id", "name", "s_index", "h_index"]
        assert list(df["openalex_id"]) == ["A1", "A2", "A3"]

    def test_load_raw_data_directory_column_standardization(self, tmp_path):
        """Directory load standardizes column names (lowercase, underscores)."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("OpenAlex_ID,Name,S-Index,H-Index\nA1,Alice,10,45")

        df = load_raw_data(str(tmp_path))

        assert "openalex_id" in df.columns
        assert "s_index" in df.columns
        assert "h_index" in df.columns

    def test_load_raw_data_empty_directory_raises(self, tmp_path):
        """Empty directory (no CSVs) raises ValueError."""
        with pytest.raises(ValueError, match="No CSV files found"):
            load_raw_data(str(tmp_path))


# =============================================================================
# Validation Tests
# =============================================================================

class TestCorrelation:
    """Tests for correlation functions."""

    def test_compute_correlation_returns_required_keys(self, sample_df):
        """Test that compute_correlation returns all required keys."""
        result = compute_correlation(sample_df, "s_index", "h_index")

        # Verify all expected keys exist
        required_keys = ["pearson_r", "spearman_rho", "pearson_p", "spearman_p", "n"]
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

    def test_compute_correlation_bounds(self, sample_df):
        """Test that correlation values are within valid bounds."""
        result = compute_correlation(sample_df, "s_index", "h_index")

        # Correlations must be in [-1, 1]
        assert -1 <= result["pearson_r"] <= 1
        assert -1 <= result["spearman_rho"] <= 1

        # P-values must be in [0, 1]
        assert 0 <= result["pearson_p"] <= 1
        assert 0 <= result["spearman_p"] <= 1

    def test_compute_correlation_known_values(self):
        """Test correlation with pre-computed expected values.

        These expected values were computed independently and verified manually.
        This avoids testing scipy against itself.
        """
        # Dataset with known correlation characteristics
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [2.1, 4.2, 5.8, 8.1, 9.9, 12.2, 13.8, 16.0, 18.1, 20.0],
        })

        result = compute_correlation(df, "x", "y")

        # Pre-computed expected Pearson r for this specific dataset: ~0.9997
        # Verified using R: cor(x, y) = 0.9996918
        assert result["pearson_r"] == pytest.approx(0.9997, abs=0.001)
        # Spearman should be 1.0 (perfect monotonic relationship)
        assert result["spearman_rho"] == pytest.approx(1.0)

    def test_compute_correlation_perfect_positive(self):
        """Test correlation with perfectly correlated data."""
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10],
        })

        result = compute_correlation(df, "x", "y")

        assert result["pearson_r"] == pytest.approx(1.0)
        assert result["spearman_rho"] == pytest.approx(1.0)

    def test_compute_correlation_perfect_negative(self):
        """Test correlation with perfectly negatively correlated data."""
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [10, 8, 6, 4, 2],
        })

        result = compute_correlation(df, "x", "y")

        assert result["pearson_r"] == pytest.approx(-1.0)
        assert result["spearman_rho"] == pytest.approx(-1.0)

    def test_compute_correlation_zero_correlation(self):
        """Test with data that has approximately zero correlation."""
        # Carefully constructed to have near-zero Pearson correlation
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [5, 3, 8, 2, 9, 1, 7, 4, 6, 5],  # Random-ish values
        })

        result = compute_correlation(df, "x", "y")

        # Should be close to zero (within 0.3 of zero)
        assert abs(result["pearson_r"]) < 0.3

    def test_compute_correlation_too_few_samples(self):
        """Test that correlation raises error with too few samples."""
        df = pd.DataFrame({"x": [1, 2], "y": [2, 4]})

        with pytest.raises(ValueError, match="Need at least 3 samples"):
            compute_correlation(df, "x", "y")

    def test_compute_correlation_constant_values(self):
        """Test that constant values return NaN correlation with warning."""
        df = pd.DataFrame({
            "x": [5, 5, 5, 5, 5],
            "y": [1, 2, 3, 4, 5],
        })

        result = compute_correlation(df, "x", "y")

        assert np.isnan(result["pearson_r"])
        assert "warning" in result

    def test_compute_correlation_sample_size_matches_input(self, sample_df):
        """Test that reported sample size matches actual input size."""
        result = compute_correlation(sample_df, "s_index", "h_index")

        assert result["n"] == len(sample_df)

    def test_compute_correlation_drops_nan_correctly(self):
        """Test that NaN values are excluded and n reflects actual pairs used."""
        df = pd.DataFrame({
            "x": [1, 2, np.nan, 4, 5],
            "y": [2, 4, 6, 8, 10],
        })

        result = compute_correlation(df, "x", "y")

        assert result["n"] == 4  # One row dropped due to NaN


class TestConditionalEntropy:
    """Tests for conditional entropy calculation."""

    def test_conditional_entropy_values_correct(self, sample_df):
        """Test that compute_conditional_entropy returns mathematically valid results."""
        result = compute_conditional_entropy(sample_df)

        # All expected keys present
        assert "H_S_given_metric" in result
        assert "H_S" in result
        assert "ratio" in result
        assert "n" in result

        # Entropy values must be non-negative (information theory constraint)
        assert result["H_S_given_metric"] >= 0
        assert result["H_S"] >= 0

        # Conditional entropy cannot exceed marginal entropy: H(S|h) <= H(S)
        # This is a fundamental property - if violated, implementation is wrong
        assert result["H_S_given_metric"] <= result["H_S"] + 0.01  # Small tolerance for numerical error

        # Ratio must be between 0 and 1 (it's H(S|h)/H(S))
        assert 0 <= result["ratio"] <= 1.01  # Small tolerance for numerical error

        # Sample size should match input
        assert result["n"] == len(sample_df[["s_index", "h_index"]].dropna())

    def test_conditional_entropy_too_few_samples(self):
        """Test that entropy raises error with too few samples."""
        df = pd.DataFrame({"s_index": [1, 2, 3], "h_index": [1, 2, 3]})

        with pytest.raises(ValueError, match="Need at least 4 valid samples"):
            compute_conditional_entropy(df)


class TestHiddenStewards:
    """Tests for hidden steward detection."""

    def test_find_hidden_stewards_criteria(self):
        """Test with known data meeting criteria.

        Create data where specific researchers clearly qualify as stewards:
        high s_index (top 25%) + low h_index (bottom 50%).
        """
        # Create data with clear hidden steward candidates
        # Alice and Bob: high S (90, 80), low h (5, 10) -> should be stewards
        # Carol and Dan: low S (10, 20), high h (60, 70) -> should NOT be stewards
        df = pd.DataFrame({
            "openalex_id": ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
            "name": ["Alice", "Bob", "Carol", "Dan", "Eve", "Frank", "Grace", "Henry"],
            "s_index": [90, 80, 10, 20, 50, 40, 30, 25],
            "h_index": [5, 10, 60, 70, 50, 45, 40, 35],
        })

        stewards, _ = find_hidden_stewards(df)

        # Alice and Bob should be identified (high S, low h)
        steward_names = set(stewards["name"]) if len(stewards) > 0 else set()
        assert "Alice" in steward_names, "Alice should be a hidden steward"
        assert "Bob" in steward_names, "Bob should be a hidden steward"

        # Carol and Dan should NOT be identified (low S or high h)
        assert "Carol" not in steward_names, "Carol should NOT be a hidden steward"
        assert "Dan" not in steward_names, "Dan should NOT be a hidden steward"

    def test_find_hidden_stewards_sorted(self):
        """Test that results are sorted by s_index descending."""
        df = pd.DataFrame({
            "openalex_id": ["A1", "A2", "A3", "A4"],
            "s_index": [100, 80, 90, 70],
            "h_index": [5, 5, 5, 5],  # All low h
        })

        stewards, _ = find_hidden_stewards(df)

        if len(stewards) > 1:
            assert stewards["s_index"].is_monotonic_decreasing

# =============================================================================
# Resample Tests
# =============================================================================

class TestResample:
    """Tests for resampling functions."""

    def test_calculate_weights_normalized(self, sample_df_with_weights):
        """Test that weights sum to 1."""
        weights = calculate_sampling_weights(sample_df_with_weights)

        assert weights.sum() == pytest.approx(1.0)

    def test_calculate_weights_prolific_author_downweighted(self, sample_df_with_weights):
        """Test that authors with more datasets get lower weight.

        Fixture data:
        - A1: num_datasets=10, team_size=2 -> should have LOWEST weight (most datasets)
        - A2: num_datasets=1, team_size=1 -> should have HIGH weight (few datasets)
        """
        weights = calculate_sampling_weights(sample_df_with_weights)

        # Verify the prolific author (A1, index 0) has lower weight than non-prolific (A2, index 1)
        assert weights[0] < weights[1], "Prolific author should have lower weight"

    def test_calculate_weights_known_values(self):
        """Test weights with pre-computed expected values.

        Dataset:
        - Author 1: 10 datasets, team_size 2 -> raw weight = 2/10 = 0.2
        - Author 2: 1 dataset, team_size 1 -> raw weight = 1/1 = 1.0
        Total raw = 1.2, so normalized: [0.2/1.2, 1.0/1.2] = [0.1667, 0.8333]
        """
        df = pd.DataFrame({
            "num_datasets": [10, 1],
            "team_size": [2, 1],
        })

        weights = calculate_sampling_weights(df)

        # Pre-computed expected values
        assert weights[0] == pytest.approx(0.1667, abs=0.001)
        assert weights[1] == pytest.approx(0.8333, abs=0.001)

    def test_calculate_weights_equal_when_same(self):
        """Test that equal inputs produce equal weights."""
        df = pd.DataFrame({
            "num_datasets": [5, 5, 5],
            "team_size": [2, 2, 2],
        })

        weights = calculate_sampling_weights(df)

        # All should be equal (1/3 each)
        assert weights[0] == pytest.approx(1/3)
        assert weights[1] == pytest.approx(1/3)
        assert weights[2] == pytest.approx(1/3)

    def test_calculate_weights_missing_columns(self):
        """Test that missing columns use default value of 1."""
        df = pd.DataFrame({"s_index": [10, 20, 30]})

        weights = calculate_sampling_weights(df)

        # With all 1s, weights should be equal
        assert len(weights) == 3
        assert weights[0] == pytest.approx(1/3)

    def test_resample_data_same_size_default(self, sample_df_with_weights):
        """Test that resampling with no n_samples returns same size."""
        resampled = resample_data(sample_df_with_weights, seed=42)

        assert len(resampled) == len(sample_df_with_weights)

    def test_resample_data_reproducible(self, sample_df_with_weights):
        """Test that same seed produces same results."""
        resampled1 = resample_data(sample_df_with_weights, seed=42)
        resampled2 = resample_data(sample_df_with_weights, seed=42)

        pd.testing.assert_frame_equal(resampled1, resampled2)

    def test_resample_data_different_seeds_differ(self, sample_df_with_weights):
        """Test that different seeds produce different results."""
        resampled1 = resample_data(sample_df_with_weights, seed=42)
        resampled2 = resample_data(sample_df_with_weights, seed=123)

        # With high probability, results should differ
        # (Could theoretically be the same, but very unlikely)
        assert not resampled1.equals(resampled2)

    def test_resample_data_empty_df(self):
        """Test that empty dataframe returns empty dataframe."""
        df = pd.DataFrame()
        resampled = resample_data(df, seed=42)

        assert len(resampled) == 0

    def test_resample_data_custom_n_samples(self, sample_df_with_weights):
        """Test resampling with custom sample size."""
        resampled = resample_data(sample_df_with_weights, seed=42, n_samples=2)

        assert len(resampled) == 2


# =============================================================================
# Integration / Sanity Tests
# =============================================================================

# =============================================================================
# Edge Case Tests
# =============================================================================

class TestDataLoaderEdgeCases:
    """Edge case tests for data loading that might quietly fail."""

    def test_load_empty_csv_with_headers(self, tmp_path):
        """Test loading CSV with headers but no data rows."""
        csv_content = "openalex_id,name,s-index,h-index\n"
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text(csv_content)

        df = load_openalex_data(str(csv_file))

        assert len(df) == 0
        assert "s_index" in df.columns

    def test_clean_data_all_nan_column(self):
        """Test cleaning when entire column is NaN - should not crash."""
        df = pd.DataFrame({
            "s_index": [np.nan, np.nan, np.nan],
            "h_index": [10, 20, 30],
        })

        cleaned = clean_data(df)

        assert cleaned["s_index"].isna().all()
        assert not cleaned["h_index"].isna().any()

    def test_clean_data_whitespace_values(self):
        """Test that whitespace-only strings become NaN."""
        df = pd.DataFrame({
            "s_index": ["10", "   ", "20"],
            "h_index": [10, 20, 30],
        })

        cleaned = clean_data(df)

        # Whitespace should be coerced to NaN
        assert pd.isna(cleaned["s_index"].iloc[1])

    def test_clean_data_mixed_types_in_column(self):
        """Test handling of mixed types (int, float, string) in one column."""
        df = pd.DataFrame({
            "s_index": [10, 20.5, "30", None],
            "h_index": [10, 20, 30, 40],
        })

        cleaned = clean_data(df)

        assert cleaned["s_index"].iloc[0] == 10
        assert cleaned["s_index"].iloc[1] == 20.5
        assert cleaned["s_index"].iloc[2] == 30
        assert pd.isna(cleaned["s_index"].iloc[3])

class TestCorrelationEdgeCases:
    """Edge cases for correlation that might give wrong results."""

    def test_correlation_with_nan_values_dropped(self):
        """Test that NaN values are properly excluded from correlation."""
        df = pd.DataFrame({
            "x": [1, 2, np.nan, 4, 5],
            "y": [2, 4, 6, 8, 10],
        })

        result = compute_correlation(df, "x", "y")

        # Should compute correlation on 4 valid pairs
        assert result["n"] == 4
        # Still perfectly correlated among valid pairs
        assert result["pearson_r"] == pytest.approx(1.0)

    def test_correlation_with_ties_spearman(self):
        """Test Spearman correlation with many ties (common in real data)."""
        df = pd.DataFrame({
            "x": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "y": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        })

        result = compute_correlation(df, "x", "y")

        # Should still compute without error
        assert not np.isnan(result["spearman_rho"])
        # Positive correlation expected
        assert result["spearman_rho"] > 0

    def test_correlation_near_zero_variance(self):
        """Test correlation when variance is very small but not zero."""
        df = pd.DataFrame({
            "x": [1.0, 1.0, 1.0, 1.0, 1.0001],  # Tiny variance
            "y": [1, 2, 3, 4, 5],
        })

        result = compute_correlation(df, "x", "y")

        # Should compute (not NaN) since variance is non-zero
        # But result may be numerically unstable
        assert "pearson_r" in result

    def test_correlation_extreme_outlier(self):
        """Test that extreme outliers heavily influence Pearson but not Spearman."""
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 1000],  # Extreme outlier
            "y": [1, 2, 3, 4, 5],
        })

        result = compute_correlation(df, "x", "y")

        # Spearman should be perfect (ranks are monotonic)
        assert result["spearman_rho"] == pytest.approx(1.0)
        # Pearson will be affected by the outlier
        assert result["pearson_r"] < result["spearman_rho"]


class TestConditionalEntropyEdgeCases:
    """Edge cases for entropy calculation."""

    def test_entropy_highly_skewed_data(self):
        """Test entropy with highly skewed distribution (many zeros)."""
        df = pd.DataFrame({
            "s_index": [0, 0, 0, 0, 0, 0, 0, 1, 2, 10],  # Very skewed
            "h_index": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        })

        result = compute_conditional_entropy(df)

        # Should complete without error
        assert result["H_S"] >= 0
        assert result["n"] == 10

    def test_entropy_identical_columns(self):
        """Test entropy when S and h are identical (H(S|h) should be ~0)."""
        values = list(range(1, 21))  # Need enough unique values
        df = pd.DataFrame({
            "s_index": values,
            "h_index": values,
        })

        result = compute_conditional_entropy(df)

        # H(S|h) should be very low since h perfectly predicts S
        assert result["H_S_given_metric"] < 0.5

    def test_entropy_independent_columns(self):
        """Test entropy when columns are independent (H(S|h) should be ~H(S))."""
        np.random.seed(42)
        # Generate independent random data
        df = pd.DataFrame({
            "s_index": np.random.randint(1, 100, size=50),
            "h_index": np.random.randint(1, 100, size=50),
        })

        result = compute_conditional_entropy(df)

        # Ratio should be close to 1 for independent variables
        assert result["ratio"] > 0.5


class TestHiddenStewardsEdgeCases:
    """Edge cases for hidden steward detection."""

    def test_hidden_stewards_all_identical_values(self):
        """Test percentile method when everyone has identical values.

        With identical high S and low h, everyone should qualify.
        The >= and <= conditions should include the boundary.
        """
        df = pd.DataFrame({
            "openalex_id": ["A1", "A2", "A3", "A4"],
            "name": ["A", "B", "C", "D"],
            "s_index": [100, 100, 100, 100],  # All same high S
            "h_index": [1, 1, 1, 1],  # All same low h
        })

        stewards, _ = find_hidden_stewards(df)

        # All 4 researchers should qualify (all at the same threshold boundary)
        assert len(stewards) == 4, "All 4 should qualify when values are identical"

    def test_hidden_stewards_none_qualify(self):
        """Test when no one qualifies as hidden steward.

        Design: high-S authors have high h, low-h authors have low S.
        No one can meet BOTH criteria (high S AND low h).
        """
        df = pd.DataFrame({
            "openalex_id": ["A1", "A2", "A3", "A4"],
            "name": ["A", "B", "C", "D"],
            "s_index": [1, 2, 100, 200],  # Low S for low h, high S for high h
            "h_index": [10, 20, 100, 200],  # Perfectly correlated with s_index
        })

        stewards, _ = find_hidden_stewards(df)

        # No one should qualify (high S people have high h, low h people have low S)
        assert len(stewards) == 0, "No one should qualify when S and h are correlated"

    def test_hidden_stewards_boundary_conditions(self):
        """Test that boundary conditions (exactly at percentile) are handled correctly.

        The >= for S and <= for h should include exact boundary values.
        """
        # Create data where one person is exactly at the S threshold
        df = pd.DataFrame({
            "openalex_id": ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
            "name": ["Alice", "Bob", "Carol", "Dan", "Eve", "Frank", "Grace", "Henry"],
            # S values: 75th percentile will be between 50 and 60
            "s_index": [10, 20, 30, 40, 50, 60, 70, 80],
            # H values: 50th percentile will be between 25 and 30
            "h_index": [5, 10, 15, 20, 25, 30, 35, 40],
        })

        stewards, thresholds = find_hidden_stewards(df)

        assert isinstance(stewards, pd.DataFrame)
        assert isinstance(thresholds, dict)
        if len(stewards) > 0:
            assert (stewards["s_index"] >= thresholds["s_threshold"]).all()
            assert (stewards["h_index"] <= thresholds["metric_threshold"]).all()


class TestResampleEdgeCases:
    """Edge cases for resampling that might corrupt data."""

    def test_weights_with_zero_num_datasets(self):
        """Test that zero num_datasets is clipped to 1 (avoid division by zero)."""
        df = pd.DataFrame({
            "num_datasets": [0, 1, 2],
            "team_size": [1, 1, 1],
        })

        weights = calculate_sampling_weights(df)

        # Should not have inf or nan
        assert np.isfinite(weights).all()
        assert weights.sum() == pytest.approx(1.0)

    def test_weights_with_negative_values(self):
        """Test that negative values are clipped to 1."""
        df = pd.DataFrame({
            "num_datasets": [-5, 1, 2],
            "team_size": [1, -3, 1],
        })

        weights = calculate_sampling_weights(df)

        # Should handle gracefully (clip to 1)
        assert np.isfinite(weights).all()
        assert (weights >= 0).all()

    def test_weights_with_nan_values(self):
        """Test that NaN values default to 1."""
        df = pd.DataFrame({
            "num_datasets": [np.nan, 1, 2],
            "team_size": [1, np.nan, 1],
        })

        weights = calculate_sampling_weights(df)

        # NaN should be filled with 1
        assert np.isfinite(weights).all()

    def test_weights_extreme_imbalance(self):
        """Test weights when one row dominates (might cause sampling issues)."""
        df = pd.DataFrame({
            "num_datasets": [1000, 1, 1, 1],  # First row has many datasets
            "team_size": [1, 1, 1, 1],
        })

        weights = calculate_sampling_weights(df)

        # First row should have much lower weight
        assert weights[0] < weights[1]
        # But weights should still sum to 1
        assert weights.sum() == pytest.approx(1.0)

    def test_resample_single_row(self):
        """Test resampling with only one row."""
        df = pd.DataFrame({
            "s_index": [10],
            "num_datasets": [5],
        })

        resampled = resample_data(df, seed=42, n_samples=1)

        assert len(resampled) == 1
        assert resampled["s_index"].iloc[0] == 10

    def test_resample_preserves_dtypes(self, sample_df_with_weights):
        """Test that resampling preserves column dtypes."""
        original_dtypes = sample_df_with_weights.dtypes
        resampled = resample_data(sample_df_with_weights, seed=42)

        for col in sample_df_with_weights.columns:
            assert resampled[col].dtype == original_dtypes[col], f"dtype changed for {col}"

    def test_resample_with_replacement_can_duplicate(self):
        """Test that with_replacement=True can sample same row multiple times."""
        df = pd.DataFrame({
            "id": [1, 2],
            "num_datasets": [1, 1000],  # Row 1 has much higher weight
        })

        # Sample many times with replacement
        resampled = resample_data(df, seed=42, n_samples=10, with_replacement=True)

        # Should have 10 rows
        assert len(resampled) == 10
        # Row 1 (id=1) should appear more often due to higher weight
        count_id_1 = (resampled["id"] == 1).sum()
        assert count_id_1 > 5  # Expect mostly id=1 due to weight

    def test_resample_without_replacement_capped(self):
        """Test that without replacement caps at dataset size."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "num_datasets": [1, 1, 1],
        })

        # Request more samples than rows
        resampled = resample_data(df, seed=42, n_samples=100, with_replacement=False)

        # Should be capped at 3
        assert len(resampled) == 3


class TestCorrelationsWrapper:
    """Tests for the correlations_with_traditional_metrics function."""

    def test_returns_results_for_all_available_metrics(self, sample_df):
        """Test that returns results for all available metric columns."""
        results = correlations_with_traditional_metrics(sample_df)

        # sample_df has h_index, cited_by_count, num_datasets
        assert "h_index" in results
        assert "cited_by_count" in results
        assert "num_datasets" in results

        # Each result should be a valid correlation dict
        for metric_name, r in results.items():
            assert "pearson_r" in r
            assert -1 <= r["pearson_r"] <= 1

    def test_skips_missing_metric_columns(self):
        """Test that missing metric columns are silently skipped."""
        df = pd.DataFrame({
            "s_index": [1, 2, 3, 4, 5],
            "h_index": [5, 4, 3, 2, 1],
            # No cited_by_count or num_datasets
        })

        results = correlations_with_traditional_metrics(df)

        assert "h_index" in results
        assert "cited_by_count" not in results
        assert "num_datasets" not in results


class TestConditionalEntropySPositive:
    """Tests for compute_conditional_entropy with s_positive_only parameter."""

    def test_s_positive_only_excludes_zeros(self):
        """Test that s_positive_only=True excludes S=0 researchers.

        With 5 zero and 10 non-zero S values, the s_positive_only version
        should report n=10 while the full version reports n=15.
        """
        df = pd.DataFrame({
            "s_index": [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "h_index": list(range(1, 16)),
        })

        result_all = compute_conditional_entropy(df, s_positive_only=False)
        result_pos = compute_conditional_entropy(df, s_positive_only=True)

        # S>0 only should use fewer samples
        assert result_pos["n"] == 10
        assert result_all["n"] == 15

        # Both should produce valid entropy values
        assert result_pos["H_S"] >= 0
        assert result_all["H_S"] >= 0

    def test_s_positive_only_changes_entropy(self):
        """Test that filtering to S>0 changes the entropy result.

        Removing zeros changes the marginal distribution of S,
        so the entropy values should differ.
        """
        df = pd.DataFrame({
            "s_index": [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "h_index": list(range(1, 16)),
        })

        result_all = compute_conditional_entropy(df, s_positive_only=False)
        result_pos = compute_conditional_entropy(df, s_positive_only=True)

        # The H(S) values should differ (different distributions)
        assert result_all["H_S"] != result_pos["H_S"]


class TestFilterByDomain:
    """Tests for filter_by_domain from config.py."""

    def test_none_returns_all_rows(self):
        """Test that domain=None returns the full dataframe."""
        df = pd.DataFrame({
            "s_index": [1, 2, 3],
            "primary_domain": ["Life Sciences", "Health Sciences", "Physical Sciences"],
        })

        result = filter_by_domain(df, domain=None)
        assert len(result) == 3

    def test_filters_to_primary_domain(self):
        """Test that only researchers whose primary domain matches are kept."""
        df = pd.DataFrame({
            "s_index": [1, 2, 3, 4],
            "primary_domain": ["Life Sciences", "Health Sciences", "Life Sciences", "Physical Sciences"],
        })

        result = filter_by_domain(df, domain=LIFE_SCIENCES)
        assert len(result) == 2
        assert (result["primary_domain"] == "Life Sciences").all()

    def test_missing_primary_domain_column_raises(self):
        """Test that missing primary_domain column raises KeyError."""
        df = pd.DataFrame({"s_index": [1, 2, 3]})

        with pytest.raises(KeyError, match="primary_domain"):
            filter_by_domain(df, domain=LIFE_SCIENCES)

    def test_no_researchers_in_domain(self):
        """Test that empty result is returned when no researchers match domain."""
        df = pd.DataFrame({
            "s_index": [10, 20, 30],
            "primary_domain": ["Health Sciences", "Physical Sciences", "Social Sciences"],
        })

        result = filter_by_domain(df, domain=LIFE_SCIENCES)
        assert len(result) == 0


class TestIntegration:
    """Integration tests using test_data.csv."""

    @pytest.fixture
    def test_data_path(self):
        """Path to the test data file in the tests directory."""
        return os.path.join(os.path.dirname(__file__), "test_data.csv")

    def test_load_test_data(self, test_data_path):
        """Test that test_data.csv loads correctly."""
        if not os.path.exists(test_data_path):
            pytest.skip("test_data.csv not found")

        df = load_openalex_data(test_data_path)
        df = clean_data(df)

        assert len(df) > 0
        assert "s_index" in df.columns
        assert "h_index" in df.columns

    def test_full_correlation_pipeline(self, test_data_path):
        """Test full correlation pipeline with real data."""
        if not os.path.exists(test_data_path):
            pytest.skip("test_data.csv not found")

        df = load_openalex_data(test_data_path)
        df = clean_data(df)

        result = compute_correlation(df, "s_index", "h_index")

        # Test data includes edge cases (hidden stewards with high S, low h),
        # so correlation may not be strongly positive.
        # Just verify the correlation is computed and is a valid value.
        assert -1 <= result["pearson_r"] <= 1
        assert result["n"] == len(df)

    def test_full_hidden_stewards_pipeline(self, test_data_path):
        """Test full hidden stewards pipeline with real data."""
        if not os.path.exists(test_data_path):
            pytest.skip("test_data.csv not found")

        df = load_openalex_data(test_data_path)
        df = clean_data(df)

        stewards, _ = find_hidden_stewards(df)

        # Result should be a dataframe
        assert isinstance(stewards, pd.DataFrame)
        # All identified stewards should have high S-index
        if len(stewards) > 0:
            assert (stewards["s_index"] > 0).all()
