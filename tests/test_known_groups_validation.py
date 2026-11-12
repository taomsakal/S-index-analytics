"""
Unit tests for known-groups validation workflow.

Tests cover:
1. Individual derivation functions (unit tests)
2. Exact matching logic (unit tests)
3. Statistical analysis functions (unit tests)
4. Full workflow integration tests

The matching design is exact 1:1 matching on (primary_domain, h_index):
every pair must be identical on both variables, the pair count is fixed by
cell capacities regardless of seed, and authors without an exact-match
control are excluded (handled separately by the NN sensitivity analysis).
"""

import numpy as np
import pandas as pd
import pytest

from known_groups_validation import (
    derive_data_paper_variables,
    derive_primary_domain,
    prepare_data,
    perform_matching,
    perform_matching_with_nn_sensitivity,
    matching_stability_analysis,
    matching_diagnostics,
    primary_analysis,
    s_positive_analysis,
    domain_matched_analysis,
    all_domains_matched_analysis,
    domain_unmatched_analysis,
    compile_results_dict,
    ALL_DOMAINS,
    run_known_groups_validation,
)


def test_derive_data_paper_variables_requires_columns():
    df = pd.DataFrame({"s_index": [1, 2]})
    with pytest.raises(KeyError, match="No data journal columns found"):
        derive_data_paper_variables(df)


def test_derive_data_paper_variables_partial_missing_filled():
    df = pd.DataFrame({"scientific_data_pubs": [1, 0, np.nan]})
    derived = derive_data_paper_variables(df)

    assert "data_paper_count" in derived.columns
    assert derived["data_paper_count"].tolist() == [1, 0, 0]
    assert derived["is_data_paper_author"].tolist() == [True, False, False]


def test_derive_primary_domain_ties_choose_randomly():
    df = pd.DataFrame({
        "health_sciences": [1],
        "life_sciences": [1],
        "physical_sciences": [1],
        "social_sciences": [1],
    })

    derived = derive_primary_domain(df, seed=42)
    # With random tie-breaking, result should be one of the four domains
    assert derived["primary_domain"].iloc[0] in [
        "Health Sciences", "Life Sciences", "Physical Sciences", "Social Sciences"
    ]
    # Same seed should produce same result
    derived2 = derive_primary_domain(df, seed=42)
    assert derived["primary_domain"].iloc[0] == derived2["primary_domain"].iloc[0]


def test_derive_primary_domain_ties_are_row_order_invariant():
    df = pd.DataFrame({
        "openalex_id": ["A3", "A1", "A2"],
        "health_sciences": [1, 1, 1],
        "life_sciences": [1, 1, 1],
        "physical_sciences": [0, 0, 0],
        "social_sciences": [0, 0, 0],
    })
    shuffled = df.sample(frac=1, random_state=7).reset_index(drop=True)

    first = derive_primary_domain(df, seed=42).set_index("openalex_id")["primary_domain"]
    second = derive_primary_domain(shuffled, seed=42).set_index("openalex_id")["primary_domain"]

    pd.testing.assert_series_equal(first.sort_index(), second.sort_index())


def test_derive_primary_domain_preserves_existing_complete_values():
    df = pd.DataFrame({
        "openalex_id": ["A1", "A2"],
        "primary_domain": ["Social Sciences", "Health Sciences"],
        "health_sciences": [1, 0],
        "life_sciences": [1, 1],
        "physical_sciences": [0, 0],
        "social_sciences": [0, 0],
    })

    derived = derive_primary_domain(df, seed=42)
    assert derived["primary_domain"].tolist() == ["Social Sciences", "Health Sciences"]


def test_derive_primary_domain_all_zero_no_domain():
    df = pd.DataFrame({
        "health_sciences": [0],
        "life_sciences": [0],
        "physical_sciences": [0],
        "social_sciences": [0],
    })

    derived = derive_primary_domain(df)
    assert derived["primary_domain"].iloc[0] == "no_domain"


def test_prepare_data_integration():
    df = pd.DataFrame({
        "scientific_data_pubs": [1, 0, 0, 2],
        "essd_pubs": [0, 0, 0, 0],
        "gigascience_pubs": [0, 1, 0, 0],
        "bds_pubs": [0, 0, 0, 0],
        "health_sciences": [1, 0, 0, 0],
        "life_sciences": [0, 1, 0, 0],
        "physical_sciences": [0, 0, 1, 0],
        "social_sciences": [0, 0, 0, 1],
        "h_index": [10, 20, 30, 40],
    })

    prepared = prepare_data(df)
    assert "is_data_paper_author" in prepared.columns
    assert "primary_domain" in prepared.columns


def test_perform_matching_requires_openalex_id():
    df = pd.DataFrame({"primary_domain": ["Health Sciences"], "h_index": [1], "is_data_paper_author": [True]})
    with pytest.raises(KeyError, match="openalex_id"):
        perform_matching(df)


def test_perform_matching_pairs_exact_controls():
    df = pd.DataFrame({
        "openalex_id": ["A1", "A2", "C1", "C2"],
        "primary_domain": ["Health Sciences", "Life Sciences", "Health Sciences", "Life Sciences"],
        "is_data_paper_author": [True, True, False, False],
        "s_index": [10, 12, 5, 6],
        "h_index": [20, 30, 20, 30],
    })

    matched_df, report = perform_matching(df, seed=42)

    assert len(matched_df) == 2
    assert report["n_matched"] == 2
    assert len(set(matched_df["control_id"])) == 2
    # Every pair must be identical on h-index (exact matching invariant)
    assert (matched_df["author_h_index"] == matched_df["control_h_index"]).all()


def test_perform_matching_excludes_without_exact_match():
    """Authors whose exact (domain, h) has no control must be excluded."""
    df = pd.DataFrame({
        "openalex_id": ["A1", "A2", "C1"],
        "primary_domain": ["Health Sciences", "Health Sciences", "Health Sciences"],
        "is_data_paper_author": [True, True, False],
        "s_index": [10, 12, 5],
        # A1 has an exact control (h=20); A2 (h=21) does not
        "h_index": [20, 21, 20],
    })

    matched_df, report = perform_matching(df, seed=42)

    assert report["n_matched"] == 1
    assert report["n_excluded_no_controls"] == 1
    excluded = report["excluded_authors"]
    assert len(excluded) == 1
    assert excluded[0]["author_id"] == "A2"
    assert excluded[0]["h_index"] == 21


def test_perform_matching_pair_count_invariant_across_seeds():
    """The number of matched pairs is fixed by cell capacities, not the seed."""
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "openalex_id": [f"R{i}" for i in range(n)],
        "primary_domain": ["Life Sciences"] * n,
        "is_data_paper_author": [True] * 10 + [False] * 50,
        "s_index": rng.integers(0, 20, n),
        # Authors have h in {5, 10}; plenty of exact controls for both
        "h_index": [5, 5, 5, 5, 5, 10, 10, 10, 10, 10] + [5] * 25 + [10] * 25,
    })

    counts = set()
    for seed in [1, 2, 3, 42, 99]:
        matched_df, report = perform_matching(df, seed=seed)
        counts.add(report["n_matched"])
        assert (matched_df["author_h_index"] == matched_df["control_h_index"]).all()

    assert counts == {10}


def test_nn_sensitivity_adds_pairs_within_caliper():
    df = pd.DataFrame({
        "openalex_id": ["A1", "A2", "C1", "C2"],
        "primary_domain": ["Health Sciences"] * 4,
        "is_data_paper_author": [True, True, False, False],
        "s_index": [10, 12, 5, 6],
        # A1 exact-matches C1 (h=20); A2 (h=30) has no exact match but
        # C2 (h=29) is within the caliper max(2, 0.1*30) = 3
        "h_index": [20, 30, 20, 29],
    })

    combined_df, report = perform_matching_with_nn_sensitivity(df, seed=42)

    assert report["n_matched"] == 1          # exact pairs
    assert report["n_nn_matched"] == 1       # NN pair added
    assert report["n_nn_unmatched"] == 0
    assert len(combined_df) == 2
    assert set(combined_df["match_type"]) == {"exact", "nn"}
    assert report["nn_pair_h_deltas"] == [1]


def test_nn_sensitivity_respects_caliper():
    df = pd.DataFrame({
        "openalex_id": ["A1", "C1"],
        "primary_domain": ["Health Sciences"] * 2,
        "is_data_paper_author": [True, False],
        "s_index": [10, 5],
        # Nearest control is |dh| = 10 away; caliper is max(2, 0.1*30) = 3
        "h_index": [30, 40],
    })

    combined_df, report = perform_matching_with_nn_sensitivity(df, seed=42)

    assert report["n_nn_matched"] == 0
    assert report["n_nn_unmatched"] == 1
    assert len(combined_df) == 0


def test_matching_diagnostics_empty():
    diagnostics = matching_diagnostics(pd.DataFrame())
    assert "error" in diagnostics


def test_matching_diagnostics_reports_exact_match_pct():
    matched_df = pd.DataFrame({
        "author_h_index": [10, 20, 30],
        "control_h_index": [10, 20, 30],
        "domain": ["Health Sciences"] * 3,
    })
    diagnostics = matching_diagnostics(matched_df)
    assert diagnostics["pct_exact_h"] == 100.0
    assert diagnostics["h_pair_diff_max"] == 0.0
    assert diagnostics["balance_ok"]


def test_primary_analysis_all_zero_differences():
    matched_df = pd.DataFrame({
        "author_s_index": [5, 5, 5],
        "control_s_index": [5, 5, 5],
    })

    results = primary_analysis(matched_df)
    assert results["wilcoxon_p_value"] == 1.0
    assert "warning" in results


def test_domain_matched_analysis_small_sample_warning():
    matched_df = pd.DataFrame({
        "domain": ["Health Sciences"] * 6,
        "author_s_index": [5, 6, 7, 8, 9, 10],
        "control_s_index": [4, 5, 6, 7, 8, 9],
    })

    result = domain_matched_analysis(matched_df, "Health Sciences")
    assert result["n_pairs"] == 6
    assert np.isnan(result["wilcoxon_p_value"])
    assert "warning" in result


def test_domain_matched_analysis_insufficient_pairs():
    matched_df = pd.DataFrame({
        "domain": ["Life Sciences"] * 3,
        "author_s_index": [5, 6, 7],
        "control_s_index": [4, 5, 6],
    })

    result = domain_matched_analysis(matched_df, "Life Sciences")
    assert "error" in result


def test_domain_unmatched_analysis_insufficient_samples():
    df = pd.DataFrame({
        "primary_domain": ["Health Sciences"] * 3,
        "is_data_paper_author": [True, False, False],
        "s_index": [10, 5, 6],
    })

    result = domain_unmatched_analysis(df, "Health Sciences")
    assert "error" in result


def test_all_domains_helpers_return_all_domains():
    matched_df = pd.DataFrame(columns=["domain", "author_s_index", "control_s_index"])
    results = all_domains_matched_analysis(matched_df)
    assert len(results) == len(ALL_DOMAINS)


def test_compile_results_dict_handles_errors():
    """Test that compile_results_dict handles error cases gracefully."""
    matching_report = {"n_data_paper_authors_total": 1, "n_matched": 0, "n_excluded_no_controls": 1, "n_excluded_no_domain": 0}
    diagnostics = {"error": "No matched pairs to analyze"}
    primary_results = {"error": "No matched pairs to analyze"}
    domain_matched_results = [{"domain": "Health Sciences", "error": "No matched pairs in Health Sciences"}]

    compiled = compile_results_dict(matching_report, diagnostics, primary_results, domain_matched_results)

    assert "domain_matched" in compiled
    assert "health_sciences" in compiled["domain_matched"]
    assert "error" in compiled["domain_matched"]["health_sciences"]


# =============================================================================
# Integration Tests
# =============================================================================

class TestKnownGroupsIntegration:
    """Integration tests for the full known-groups validation workflow."""

    @pytest.fixture
    def complete_dataset(self):
        """Create a complete dataset with all required columns for full workflow.

        This dataset has:
        - Data paper authors with generally higher S-index
        - Controls with IDENTICAL h-index values in the same domains, so
          every author has an exact match available
        - Clear patterns that should produce significant results
        """
        h_values = [10, 20, 30, 40, 15, 25, 35, 45, 12, 22,
                    32, 42, 18, 28, 38, 48, 14, 24, 34, 44]
        # Create 40 researchers: 20 data paper authors, 20 controls
        data = {
            "openalex_id": [f"A{i:03d}" for i in range(1, 41)],
            "s_index": (
                # Data paper authors (1-20): generally higher S
                [25, 30, 22, 28, 35, 20, 32, 24, 27, 31,
                 18, 26, 29, 33, 21, 23, 34, 19, 36, 38] +
                # Controls (21-40): generally lower S
                [5, 8, 3, 6, 10, 4, 7, 2, 9, 11,
                 6, 4, 8, 5, 3, 7, 9, 2, 10, 12]
            ),
            # Controls mirror the author h-index values exactly so exact
            # matching can pair every author
            "h_index": h_values + h_values,
            "cited_by_count": (
                [1000 + 10 * i for i in range(20)] +
                [1000 + 10 * i for i in range(20)]
            ),
            # Data journal publications
            "scientific_data_pubs": [1]*20 + [0]*20,
            "essd_pubs": [0]*40,
            "gigascience_pubs": [0]*40,
            "bds_pubs": [0]*40,
            # Domain publications (evenly distributed, same pattern in
            # both halves so author i and control i+20 share a domain)
            "health_sciences": ([1, 0, 0, 0, 1]*4 + [1, 0, 0, 0, 1]*4),
            "life_sciences": ([0, 1, 0, 0, 0]*4 + [0, 1, 0, 0, 0]*4),
            "physical_sciences": ([0, 0, 1, 0, 0]*4 + [0, 0, 1, 0, 0]*4),
            "social_sciences": ([0, 0, 0, 1, 0]*4 + [0, 0, 0, 1, 0]*4),
        }
        return pd.DataFrame(data)

    def test_full_workflow_produces_results(self, complete_dataset):
        """Test that full workflow runs without errors and produces results."""
        results = run_known_groups_validation(complete_dataset, seed=42,
                                              n_stability_replicates=20,
                                              write_outputs=False,
                                              generate_plots=False)

        assert isinstance(results, dict)
        assert "n_matched" in results
        assert results["n_matched"] > 0
        assert "stability" in results
        assert "nn_sensitivity" in results

    def test_matching_is_exact(self, complete_dataset):
        """Every matched pair must be identical on h-index."""
        df_prepared = prepare_data(complete_dataset)
        matched_df, report = perform_matching(df_prepared, seed=42)
        diagnostics = matching_diagnostics(matched_df)

        assert diagnostics["pct_exact_h"] == 100.0
        assert diagnostics["h_mean_difference"] == 0
        assert report["n_matched"] == 20
        assert report["n_excluded_no_controls"] == 0

    def test_authors_have_higher_s_index(self, complete_dataset):
        """Test that data paper authors have higher S-index than controls.

        This is the key validation test - data paper authors should
        have significantly higher S-index if the metric is valid.
        """
        df_prepared = prepare_data(complete_dataset)
        matched_df, _ = perform_matching(df_prepared, seed=42)

        if len(matched_df) == 0:
            pytest.skip("No matched pairs available")

        results = primary_analysis(matched_df)

        # Median S should be higher for authors
        assert results["median_s_authors"] > results["median_s_controls"], \
            "Data paper authors should have higher median S-index"

        # More author wins than control wins
        assert results["prop_pairs_author_wins"] > results["prop_pairs_control_wins"], \
            "Authors should win more pairwise comparisons"

    def test_prepare_data_adds_all_required_columns(self, complete_dataset):
        """Test that prepare_data adds all columns needed for matching."""
        prepared = prepare_data(complete_dataset)

        required_cols = ["is_data_paper_author", "primary_domain"]
        for col in required_cols:
            assert col in prepared.columns, f"Missing required column: {col}"

    def test_matching_exact_on_domain_and_h(self, complete_dataset):
        """Test that matching is exact on both domain and h-index."""
        df_prepared = prepare_data(complete_dataset)
        matched_df, _ = perform_matching(df_prepared, seed=42)

        if len(matched_df) == 0:
            pytest.skip("No matched pairs available")

        for _, row in matched_df.iterrows():
            author = df_prepared[df_prepared["openalex_id"] == row["author_id"]].iloc[0]
            control = df_prepared[df_prepared["openalex_id"] == row["control_id"]].iloc[0]

            assert author["primary_domain"] == control["primary_domain"], \
                "Matched pairs should have same domain"
            assert author["h_index"] == control["h_index"], \
                "Matched pairs should have identical h-index"

    def test_no_control_used_twice(self, complete_dataset):
        """Test that matching samples controls without replacement."""
        df_prepared = prepare_data(complete_dataset)
        matched_df, _ = perform_matching(df_prepared, seed=42)

        if len(matched_df) == 0:
            pytest.skip("No matched pairs available")

        # Each control should appear at most once
        control_ids = matched_df["control_id"].tolist()
        assert len(control_ids) == len(set(control_ids)), \
            "Controls should not be reused"

    def test_reproducible_with_same_seed(self, complete_dataset):
        """Test that same seed produces identical results."""
        df_prepared = prepare_data(complete_dataset)

        matched1, _ = perform_matching(df_prepared, seed=42)
        matched2, _ = perform_matching(df_prepared, seed=42)

        pd.testing.assert_frame_equal(matched1, matched2)

    def test_row_order_does_not_change_results(self, complete_dataset):
        """Shuffling input rows must not change the matched pairs (same seed)."""
        df_prepared = prepare_data(complete_dataset)
        shuffled = df_prepared.sample(frac=1, random_state=7).reset_index(drop=True)

        matched1, _ = perform_matching(df_prepared, seed=42)
        matched2, _ = perform_matching(shuffled, seed=42)

        sorted1 = matched1.sort_values("author_id").reset_index(drop=True)
        sorted2 = matched2.sort_values("author_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(sorted1, sorted2)

    def test_stability_analysis_returns_summary(self, complete_dataset):
        """Stability analysis should summarize p-values across draws."""
        df_prepared = prepare_data(complete_dataset)
        stability = matching_stability_analysis(df_prepared, n_replicates=10,
                                                base_seed=42)

        assert stability["n_replicates"] == 10
        assert stability["n_pairs"] == 20
        assert 0 <= stability["p_min"] <= stability["p_median"] <= stability["p_max"] <= 1
        assert set(stability["domain_summaries"]) == set(ALL_DOMAINS)
        assert len(stability["replicate_records"]) == 10 * (1 + len(ALL_DOMAINS)) * 2

    def test_s_positive_view_does_not_repeat_wilcoxon(self, complete_dataset):
        df_prepared = prepare_data(complete_dataset)
        matched_df, _ = perform_matching(df_prepared, seed=42)
        result = s_positive_analysis(matched_df)

        assert "wilcoxon_p_value" not in result
        assert "positive_mcnemar_p_value" in result
        assert "Descriptive subset only" in result["inference_note"]


class TestKnownGroupsEdgeCases:
    """Edge case tests for known-groups validation."""

    def test_handles_no_data_paper_authors(self):
        """Test handling when there are no data paper authors."""
        df = pd.DataFrame({
            "openalex_id": ["A1", "A2", "A3"],
            "s_index": [10, 20, 30],
            "h_index": [15, 25, 35],
            "scientific_data_pubs": [0, 0, 0],
            "essd_pubs": [0, 0, 0],
            "gigascience_pubs": [0, 0, 0],
            "bds_pubs": [0, 0, 0],
            "health_sciences": [1, 0, 0],
            "life_sciences": [0, 1, 0],
            "physical_sciences": [0, 0, 1],
            "social_sciences": [0, 0, 0],
        })

        prepared = prepare_data(df)
        matched_df, report = perform_matching(prepared, seed=42)

        # Should have no matches (no authors to match)
        assert len(matched_df) == 0
        assert report["n_matched"] == 0

    def test_handles_no_controls_available(self):
        """Test handling when there are no controls for matching."""
        df = pd.DataFrame({
            "openalex_id": ["A1", "A2", "A3"],
            "s_index": [10, 20, 30],
            "h_index": [15, 25, 35],
            "scientific_data_pubs": [1, 1, 1],  # All are data paper authors
            "essd_pubs": [0, 0, 0],
            "gigascience_pubs": [0, 0, 0],
            "bds_pubs": [0, 0, 0],
            "health_sciences": [1, 1, 1],
            "life_sciences": [0, 0, 0],
            "physical_sciences": [0, 0, 0],
            "social_sciences": [0, 0, 0],
        })

        prepared = prepare_data(df)
        matched_df, report = perform_matching(prepared, seed=42)

        # Should have no matches (no controls available)
        assert len(matched_df) == 0
        assert report["n_excluded_no_controls"] == 3

    def test_handles_all_same_s_index(self):
        """Test analysis when all S-indices are identical."""
        matched_df = pd.DataFrame({
            "author_id": ["A1", "A2", "A3"],
            "control_id": ["C1", "C2", "C3"],
            "domain": ["Health Sciences", "Life Sciences", "Physical Sciences"],
            "author_s_index": [10, 10, 10],  # All same
            "control_s_index": [10, 10, 10],  # All same
            "author_h_index": [15, 25, 35],
            "control_h_index": [15, 25, 35],
        })

        results = primary_analysis(matched_df)

        # Should handle gracefully with warning
        assert results["prop_pairs_ties"] == 1.0  # All ties
        assert "warning" in results or results["wilcoxon_p_value"] == 1.0
