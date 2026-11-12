"""
Unit tests for data cleaning and post-processing utilities.

These tests verify the data cleaning pipeline that removes:
- Organizations (not individual researchers)
- Implausible records (data quality issues)
- Duplicate authors (by ID and fuzzy name matching)
"""

import pandas as pd
import numpy as np

from data_loader import (
    filter_organizations,
    filter_implausible_records,
    filter_for_resampling,
    deduplicate_authors,
    normalize_name,
    deduplicate_fuzzy_names,
    postprocess_resampled_data,
    MAX_PLAUSIBLE_DATASETS,
    PLACEHOLDER_ZERO_MIN_DATASETS,
    KNOWN_MANUAL_EXCLUSIONS,
)


# =============================================================================
# Organization Filtering Tests
# =============================================================================

class TestFilterOrganizations:
    """Tests for filtering out organizational records."""

    def test_removes_university_names(self):
        """Test that names containing 'University' are filtered."""
        df = pd.DataFrame({
            "name": ["Alice Smith", "University of Example", "Bob Jones"],
            "s_index": [5, 10, 7],
        })

        filtered = filter_organizations(df, verbose=False)

        assert "University of Example" not in filtered["name"].values
        assert len(filtered) == 2

    def test_removes_placeholder_names(self):
        """Test that placeholder names like 'Unknown' are filtered."""
        df = pd.DataFrame({
            "name": ["Alice Smith", "Unknown", "Anonymous"],
            "s_index": [5, 10, 7],
        })

        filtered = filter_organizations(df, verbose=False)

        assert "Unknown" not in filtered["name"].values
        assert "Anonymous" not in filtered["name"].values
        assert len(filtered) == 1

    def test_removes_collaboration_names(self):
        """Test that scientific collaborations (ATLAS, CMS, etc.) are filtered."""
        df = pd.DataFrame({
            "name": ["Alice Smith", "ATLAS Collaboration", "CMS Team"],
            "s_index": [5, 10, 7],
        })

        filtered = filter_organizations(df, verbose=False)

        assert len(filtered) == 1
        assert filtered["name"].iloc[0] == "Alice Smith"

    def test_removes_museum_names(self):
        """Test that museum names are filtered."""
        df = pd.DataFrame({
            "name": ["Alice Smith", "Museum of Natural History", "Bob Jones"],
            "s_index": [5, 10, 7],
        })

        filtered = filter_organizations(df, verbose=False)

        assert "Museum of Natural History" not in filtered["name"].values

    def test_preserves_valid_researcher_names(self):
        """Test that valid researcher names are preserved."""
        df = pd.DataFrame({
            "name": ["Alice Smith", "Bob Jones", "Maria Garcia-Lopez"],
            "s_index": [5, 10, 7],
        })

        filtered = filter_organizations(df, verbose=False)

        assert len(filtered) == 3

    def test_case_sensitive_alice_keyword(self):
        """Test that 'ALICE' (all-caps collaboration) is filtered but 'Alice' is not.

        The ALICE keyword is case-sensitive to avoid filtering out the
        common first name 'Alice' while catching the ALICE particle
        physics collaboration.
        """
        df = pd.DataFrame({
            "name": ["Alice Johnson", "ALICE Johnson", "Bob Jones"],
            "s_index": [5, 10, 7],
        })

        filtered = filter_organizations(df, verbose=False)

        assert "Alice Johnson" in filtered["name"].values, \
            "Person named 'Alice' should not be filtered"
        assert "ALICE Johnson" not in filtered["name"].values, \
            "ALICE (collaboration keyword) should be filtered"
        assert len(filtered) == 2

    def test_missing_name_column_returns_unchanged(self):
        """Test that missing 'name' column returns dataframe unchanged."""
        df = pd.DataFrame({"s_index": [1, 2, 3]})

        filtered = filter_organizations(df, verbose=False)

        pd.testing.assert_frame_equal(filtered, df)


# =============================================================================
# Implausible Records Filtering Tests
# =============================================================================

class TestFilterImplausibleRecords:
    """Tests for filtering implausible data records."""

    def test_removes_high_num_datasets(self):
        """Test that records with num_datasets above threshold are removed."""
        df = pd.DataFrame({
            "name": ["A", "B", "C"],
            "num_datasets": [1, MAX_PLAUSIBLE_DATASETS + 1, 5],
        })

        filtered = filter_implausible_records(df, verbose=False)

        assert len(filtered) == 2
        assert (filtered["num_datasets"] <= MAX_PLAUSIBLE_DATASETS).all()

    def test_preserves_valid_records(self):
        """Test that records at or below threshold are preserved."""
        df = pd.DataFrame({
            "name": ["A", "B", "C"],
            "num_datasets": [1, MAX_PLAUSIBLE_DATASETS, 5],
        })

        filtered = filter_implausible_records(df, verbose=False)

        assert len(filtered) == 3

    def test_removes_known_disambiguation_errors(self):
        """Test that a profile listed in known_exclusions.csv is removed."""
        error_id = KNOWN_MANUAL_EXCLUSIONS["openalex_id"].iloc[0]
        df = pd.DataFrame({
            "openalex_id": [error_id, "A123", "A456"],
            "name": ["Merged Author", "Bob", "Alice"],
            "s_index": [83975, 10, 20],
            "num_datasets": [100, 5, 10],
        })

        filtered = filter_implausible_records(df, verbose=False)

        assert error_id not in filtered["openalex_id"].values
        assert len(filtered) == 2

    def test_removes_every_manual_exclusion(self):
        """Every ID in the evidence ledger must be removed by preprocessing."""
        exclusion_ids = KNOWN_MANUAL_EXCLUSIONS["openalex_id"].tolist()
        df = pd.DataFrame({
            "openalex_id": exclusion_ids + ["A_VALID"],
            "name": KNOWN_MANUAL_EXCLUSIONS["name"].tolist() + ["Valid Researcher"],
            "s_index": [1] * len(exclusion_ids) + [10],
            "num_datasets": [1] * len(exclusion_ids) + [5],
        })

        filtered = filter_implausible_records(df, verbose=False)

        assert filtered["openalex_id"].tolist() == ["A_VALID"]

    def test_michael_bedward_metric_artifact_is_excluded(self):
        michael_id = "A5055864830"
        ledger_row = KNOWN_MANUAL_EXCLUSIONS[
            KNOWN_MANUAL_EXCLUSIONS["openalex_id"] == michael_id
        ]
        assert len(ledger_row) == 1
        assert ledger_row.iloc[0]["category"] == "metric_artifact"

        df = pd.DataFrame({
            "openalex_id": [michael_id, "A_VALID"],
            "name": ["Michael Bedward", "Valid Researcher"],
            "s_index": [84000, 10],
            "num_datasets": [3, 5],
        })

        filtered = filter_implausible_records(df, verbose=False)

        assert filtered["openalex_id"].tolist() == ["A_VALID"]

    def test_handles_missing_num_datasets_column(self):
        """Test that missing num_datasets column returns dataframe unchanged."""
        df = pd.DataFrame({"name": ["A", "B"], "s_index": [1, 2]})

        filtered = filter_implausible_records(df, verbose=False)

        assert len(filtered) == 2


# =============================================================================
# Deduplication Tests
# =============================================================================

# =============================================================================
# Pre-Resampling Filter Tests
# =============================================================================

class TestFilterForResampling:
    """Tests for pre-resampling data filters."""

    def test_removes_missing_s_index(self):
        """Test that records with missing S-index are removed."""
        df = pd.DataFrame({
            "openalex_id": ["A1", "A2", "A3"],
            "s_index": [10, np.nan, 30],
            "num_datasets": [5, 5, 5],
        })

        filtered = filter_for_resampling(df, verbose=False)

        assert len(filtered) == 2
        assert not filtered["s_index"].isna().any()

    def test_preserves_valid_records(self):
        """Test that all records with valid S-index are preserved regardless of num_datasets."""
        df = pd.DataFrame({
            "openalex_id": ["A1", "A2", "A3"],
            "s_index": [10, 20, 30],
            "num_datasets": [100, 5000, 300],
        })

        filtered = filter_for_resampling(df, verbose=False)

        assert len(filtered) == 3

    def test_resets_index(self):
        """Test that filtered result has a clean reset index."""
        df = pd.DataFrame({
            "openalex_id": ["A1", "A2", "A3"],
            "s_index": [10, np.nan, 30],
            "num_datasets": [5, 5, 5],
        })

        filtered = filter_for_resampling(df, verbose=False)

        assert list(filtered.index) == [0, 1]

    def test_keeps_high_num_datasets_with_nonzero_s(self):
        """Test that high num_datasets with real S-index > 0 are kept."""
        df = pd.DataFrame({
            "openalex_id": ["A1", "A2", "A3", "A4"],
            "s_index": [10, np.nan, 30, 40],
            "num_datasets": [5, 5, 9999, 5],
        })

        filtered = filter_for_resampling(df, verbose=False)

        # A2 removed (missing S); A3 kept because S=30 > 0
        assert len(filtered) == 3
        assert set(filtered["openalex_id"]) == {"A1", "A3", "A4"}

    def test_removes_placeholder_zero_s_index(self):
        """Test that num_datasets >= 5000 with S=0 are removed (S-index not computed)."""
        df = pd.DataFrame({
            "openalex_id": ["A1", "A2", "A3", "A4"],
            "s_index": [10, 0, 0, 5],
            "num_datasets": [100, PLACEHOLDER_ZERO_MIN_DATASETS, 200, PLACEHOLDER_ZERO_MIN_DATASETS + 1000],
        })

        filtered = filter_for_resampling(df, verbose=False)

        # A2 removed (placeholder zero: high datasets + S=0)
        # A3 kept (S=0 but low datasets — genuinely zero)
        # A4 kept (high datasets but S=5 > 0 — real value)
        assert len(filtered) == 3
        assert set(filtered["openalex_id"]) == {"A1", "A3", "A4"}

    def test_placeholder_zero_at_exact_threshold(self):
        """Test boundary: exactly at threshold with S=0 is removed."""
        df = pd.DataFrame({
            "openalex_id": ["A1", "A2"],
            "s_index": [0, 0],
            "num_datasets": [PLACEHOLDER_ZERO_MIN_DATASETS, PLACEHOLDER_ZERO_MIN_DATASETS - 1],
        })

        filtered = filter_for_resampling(df, verbose=False)

        # A1 removed (at threshold with S=0), A2 kept (below threshold)
        assert len(filtered) == 1
        assert filtered["openalex_id"].iloc[0] == "A2"


# =============================================================================
# Deduplication Tests
# =============================================================================

class TestDeduplicateAuthors:
    """Tests for deduplicating by OpenAlex ID."""

    def test_keeps_most_complete_record(self):
        """Test that deduplication keeps the most complete record (most non-null fields),
        breaking ties by h_index — NOT by s_index (the metric under validation)."""
        df = pd.DataFrame({
            "openalex_id": ["A1", "A1", "A2"],
            "s_index": [10, 25, 5],
            "h_index": [50, 30, 20],
            "name": ["Alice", "Alice Dup", "Bob"],
        })

        deduped = deduplicate_authors(df, verbose=False)

        assert len(deduped) == 2
        # Both A1 rows have same completeness (4 non-null fields each),
        # so tie-break goes to highest h_index (50), which has s_index=10.
        assert deduped.loc[deduped["openalex_id"] == "A1", "h_index"].iloc[0] == 50
        assert deduped.loc[deduped["openalex_id"] == "A1", "s_index"].iloc[0] == 10

    def test_missing_id_column_returns_unchanged(self):
        """Test that missing openalex_id column returns dataframe unchanged."""
        df = pd.DataFrame({"name": ["A", "B"], "s_index": [1, 2]})

        deduped = deduplicate_authors(df, verbose=False)

        pd.testing.assert_frame_equal(deduped, df)

    def test_unique_ids_unchanged(self):
        """Test that dataframe with all unique IDs is unchanged."""
        df = pd.DataFrame({
            "openalex_id": ["A1", "A2", "A3"],
            "s_index": [10, 20, 30],
        })

        deduped = deduplicate_authors(df, verbose=False)

        assert len(deduped) == 3


# =============================================================================
# Name Normalization Tests
# =============================================================================

class TestNormalizeName:
    """Tests for name normalization function."""

    def test_handles_case_variations(self):
        """Test that case variations normalize to same result."""
        assert normalize_name("Tokihiko TOKUZAWA") == "tokihiko tokuzawa"
        assert normalize_name("tokihiko tokuzawa") == "tokihiko tokuzawa"

    def test_handles_comma_separated_names(self):
        """Test that 'Last, First' format normalizes correctly."""
        assert normalize_name("TOKUZAWA, Tokihiko") == "tokihiko tokuzawa"
        assert normalize_name("Smith, John") == "john smith"

    def test_normalizes_whitespace(self):
        """Test that extra whitespace is normalized."""
        assert normalize_name("  John   Smith ") == "john smith"

    def test_handles_invalid_input(self):
        """Test that invalid inputs return empty string."""
        assert normalize_name(None) == ""
        assert normalize_name(np.nan) == ""
        assert normalize_name(123) == ""
        assert normalize_name("") == ""


# =============================================================================
# Fuzzy Name Deduplication Tests
# =============================================================================

class TestDeduplicateFuzzyNames:
    """Tests for fuzzy name matching deduplication."""

    def test_keeps_highest_h_index(self):
        """Test that fuzzy dedup keeps record with highest h_index."""
        df = pd.DataFrame({
            "openalex_id": ["A1", "A2", "A3"],
            "name": ["John Smith", "Smith, John", "Alice Brown"],
            "h_index": [10, 20, 5],
            "s_index": [5, 6, 2],
        })

        deduped = deduplicate_fuzzy_names(df, verbose=False)

        assert len(deduped) == 2
        # The "Smith, John" variant should be kept (higher h_index)
        assert "Smith, John" in set(deduped["name"])
        assert deduped.loc[deduped["name"] == "Smith, John", "h_index"].iloc[0] == 20

    def test_preserves_different_names(self):
        """Test that genuinely different names are preserved."""
        df = pd.DataFrame({
            "openalex_id": ["A1", "A2", "A3"],
            "name": ["John Smith", "Jane Doe", "Bob Wilson"],
            "h_index": [10, 20, 5],
            "s_index": [5, 6, 2],
        })

        deduped = deduplicate_fuzzy_names(df, verbose=False)

        assert len(deduped) == 3


# =============================================================================
# Full Pipeline Tests
# =============================================================================

class TestPostprocessResampledData:
    """Tests for the full post-processing pipeline."""

    def test_applies_all_steps(self):
        """Test that all cleaning steps are applied in order."""
        df = pd.DataFrame({
            "openalex_id": ["A1", "A1", "A2", "A3"],
            "name": ["Alice Smith", "Alice Smith", "University of X", "Bob Brown"],
            "s_index": [10, 20, 5, 7],
            "h_index": [5, 6, 3, 4],
            "num_datasets": [1, 2, MAX_PLAUSIBLE_DATASETS + 10, 3],
        })

        processed = postprocess_resampled_data(df, verbose=False)

        # Should remove organization (University of X), and dedupe A1 to keep s_index=20
        assert len(processed) == 2
        assert set(processed["openalex_id"]) == {"A1", "A3"}
        assert processed.loc[processed["openalex_id"] == "A1", "s_index"].iloc[0] == 20

    def test_order_matters(self):
        """Test that filtering happens before deduplication."""
        # If dedup happened first, we'd keep the wrong record
        df = pd.DataFrame({
            "openalex_id": ["A1", "A1"],
            "name": ["University Lab", "Alice Smith"],  # First is org, second is person
            "s_index": [100, 20],  # Org has higher s_index
            "h_index": [50, 10],
        })

        processed = postprocess_resampled_data(df, verbose=False)

        # Should keep Alice Smith (the person), not University Lab
        assert len(processed) == 1
        assert processed["name"].iloc[0] == "Alice Smith"

    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        df = pd.DataFrame(columns=["openalex_id", "name", "s_index", "h_index"])

        processed = postprocess_resampled_data(df, verbose=False)

        assert len(processed) == 0
