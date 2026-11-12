"""
Unit tests for effective sample size calculation and resampling utilities.

Tests cover:
1. Effective Sample Size (ESS) calculation
2. Resampling with file I/O
3. Weight-based sampling behavior
"""

import pandas as pd

from resample import calculate_effective_sample_size, resample_to_file, resample_data


# =============================================================================
# Effective Sample Size Tests
# =============================================================================

class TestEffectiveSampleSize:
    """Tests for the Kish ESS calculation."""

    def test_uniform_weights_equals_n(self):
        """Test that uniform weights give ESS = n.

        When all weights are equal, ESS should equal the sample size
        because there's no loss of precision from weight variability.
        """
        df = pd.DataFrame({
            "num_datasets": [1, 1, 1, 1],
            "team_size": [1, 1, 1, 1],
        })

        ess = calculate_effective_sample_size(df)

        assert ess == len(df)

    def test_skewed_weights_reduces_ess(self):
        """Test that skewed weights reduce ESS below n.

        When one observation has much higher weight, the effective
        sample size is reduced because that observation dominates.
        """
        df = pd.DataFrame({
            "num_datasets": [1, 100, 100, 100],  # First row has much higher weight
            "team_size": [1, 1, 1, 1],
        })

        ess = calculate_effective_sample_size(df)

        assert 1 <= ess < len(df)

    def test_extreme_skew_gives_low_ess(self):
        """Test that extreme weight imbalance gives very low ESS.

        If one observation has vastly higher weight than others,
        ESS should be close to 1 (essentially one effective observation).
        """
        df = pd.DataFrame({
            "num_datasets": [1, 10000, 10000, 10000],  # First row dominates
            "team_size": [1, 1, 1, 1],
        })

        ess = calculate_effective_sample_size(df)

        # ESS should be very close to 1
        assert ess <= 2

    def test_ess_capped_at_n(self):
        """Test that ESS never exceeds the actual sample size."""
        df = pd.DataFrame({
            "num_datasets": [1, 2, 3, 4, 5],
            "team_size": [5, 4, 3, 2, 1],
        })

        ess = calculate_effective_sample_size(df)

        assert ess <= len(df)

    def test_ess_formula_known_value(self):
        """Test ESS with pre-computed expected value.

        For weights [0.5, 0.5]: ESS = (0.5+0.5)^2 / (0.5^2 + 0.5^2) = 1 / 0.5 = 2
        """
        df = pd.DataFrame({
            "num_datasets": [1, 1],  # Equal weights
            "team_size": [1, 1],
        })

        ess = calculate_effective_sample_size(df)

        assert ess == 2


# =============================================================================
# Resample to File Tests
# =============================================================================

class TestResampleToFile:
    """Tests for resampling with file I/O."""

    def test_writes_output_file(self, tmp_path):
        """Test that resample_to_file creates an output file."""
        input_csv = tmp_path / "input.csv"
        output_csv = tmp_path / "output.csv"

        input_csv.write_text(
            "openalex_id,s_index,h_index,num_datasets,team_size\n"
            "A1,10,20,1,1\n"
            "A2,5,10,2,2\n"
            "A3,7,14,3,1\n"
        )

        resample_to_file(
            input_path=str(input_csv),
            output_path=str(output_csv),
            seed=42,
            n_samples=2,
        )

        assert output_csv.exists()

    def test_returns_resampled_dataframe(self, tmp_path):
        """Test that resample_to_file returns the resampled data."""
        input_csv = tmp_path / "input.csv"
        output_csv = tmp_path / "output.csv"

        input_csv.write_text(
            "openalex_id,s_index,h_index,num_datasets,team_size\n"
            "A1,10,20,1,1\n"
            "A2,5,10,2,2\n"
            "A3,7,14,3,1\n"
        )

        resampled = resample_to_file(
            input_path=str(input_csv),
            output_path=str(output_csv),
            seed=42,
            n_samples=2,
        )

        assert len(resampled) == 2
        assert isinstance(resampled, pd.DataFrame)

    def test_output_matches_returned_data(self, tmp_path):
        """Test that saved file matches returned dataframe."""
        input_csv = tmp_path / "input.csv"
        output_csv = tmp_path / "output.csv"

        input_csv.write_text(
            "openalex_id,s_index,h_index,num_datasets,team_size\n"
            "A1,10,20,1,1\n"
            "A2,5,10,2,2\n"
            "A3,7,14,3,1\n"
        )

        resampled = resample_to_file(
            input_path=str(input_csv),
            output_path=str(output_csv),
            seed=42,
            n_samples=3,
        )

        # Read back and compare
        saved = pd.read_csv(output_csv)
        pd.testing.assert_frame_equal(resampled, saved)


# =============================================================================
# Resample Data Tests
# =============================================================================

class TestResampleData:
    """Tests for the core resampling function."""

    def test_respects_n_samples(self):
        """Test that resampling returns requested number of samples."""
        df = pd.DataFrame({
            "openalex_id": ["A1", "A2", "A3", "A4", "A5"],
            "num_datasets": [1, 1, 1, 1, 1],
            "team_size": [1, 1, 1, 1, 1],
        })

        resampled = resample_data(df, seed=42, n_samples=3)

        assert len(resampled) == 3

    def test_reproducible_with_seed(self):
        """Test that same seed produces identical results."""
        df = pd.DataFrame({
            "openalex_id": ["A1", "A2", "A3", "A4", "A5"],
            "num_datasets": [1, 2, 3, 4, 5],
            "team_size": [1, 1, 1, 1, 1],
        })

        resampled1 = resample_data(df, seed=42)
        resampled2 = resample_data(df, seed=42)

        pd.testing.assert_frame_equal(resampled1, resampled2)

    def test_different_seeds_differ(self):
        """Test that different seeds produce different results."""
        df = pd.DataFrame({
            "openalex_id": ["A1", "A2", "A3", "A4", "A5"],
            "num_datasets": [1, 10, 1, 10, 1],  # Varied weights
            "team_size": [1, 1, 1, 1, 1],
        })

        resampled1 = resample_data(df, seed=42, n_samples=5)
        resampled2 = resample_data(df, seed=123, n_samples=5)

        # Should be different (very unlikely to be identical)
        assert not resampled1.equals(resampled2)

    def test_high_weight_rows_more_likely(self):
        """Test that rows with higher weight are sampled more often.

        Over many samples, the low-num_datasets row should appear more frequently.
        """
        df = pd.DataFrame({
            "openalex_id": ["A1", "A2"],
            "num_datasets": [1, 1000],  # A1 has much higher weight
            "team_size": [1, 1],
        })

        # Sample many times with replacement
        resampled = resample_data(df, seed=42, n_samples=100, with_replacement=True)

        # A1 should appear much more often than A2
        a1_count = (resampled["openalex_id"] == "A1").sum()
        a2_count = (resampled["openalex_id"] == "A2").sum()

        assert a1_count > a2_count * 5, "High-weight row should appear much more often"

    def test_empty_dataframe(self):
        """Test that empty dataframe returns empty dataframe."""
        df = pd.DataFrame()

        resampled = resample_data(df, seed=42)

        assert len(resampled) == 0

    def test_without_replacement_no_duplicates(self):
        """Test that without-replacement sampling has no duplicates."""
        df = pd.DataFrame({
            "openalex_id": ["A1", "A2", "A3", "A4", "A5"],
            "num_datasets": [1, 1, 1, 1, 1],
            "team_size": [1, 1, 1, 1, 1],
        })

        resampled = resample_data(df, seed=42, n_samples=5, with_replacement=False)

        # All IDs should be unique
        assert len(resampled["openalex_id"].unique()) == 5
