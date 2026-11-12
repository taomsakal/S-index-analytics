import numpy as np
import pandas as pd

from utilities.run_strict_journal_known_groups import (
    INCLUDED_JOURNAL_COLUMNS,
    EXCLUDED_JOURNAL_COLUMNS,
    _enrich_stability_with_fdr,
    to_builtin,
    apply_strict_journal_definition,
)


def test_strict_journal_definition_preserves_input_and_zeros_broad_journals():
    original = pd.DataFrame(
        {
            "scientific_data_pubs": [1, 0, 0, 0],
            "essd_pubs": [0, 2, 0, 0],
            "gigascience_pubs": [0, 0, 3, 0],
            "bds_pubs": [0, 0, 0, 4],
        }
    )

    strict = apply_strict_journal_definition(original)

    assert original["gigascience_pubs"].tolist() == [0, 0, 3, 0]
    assert original["bds_pubs"].tolist() == [0, 0, 0, 4]
    assert strict[list(EXCLUDED_JOURNAL_COLUMNS)].to_numpy().sum() == 0
    assert strict[list(INCLUDED_JOURNAL_COLUMNS)].equals(
        original[list(INCLUDED_JOURNAL_COLUMNS)]
    )


def test_strict_journal_definition_requires_all_journal_columns():
    incomplete = pd.DataFrame({"scientific_data_pubs": [1]})

    try:
        apply_strict_journal_definition(incomplete)
    except KeyError as error:
        assert "Missing journal columns" in str(error)
    else:
        raise AssertionError("Expected missing journal columns to raise KeyError")


def test_json_conversion_handles_numpy_booleans():
    assert to_builtin(np.bool_(True)) is True
    assert to_builtin({"balanced": np.bool_(False)}) == {"balanced": False}


def test_stability_enrichment_applies_fdr_within_each_draw():
    records = []
    for replicate, overall_p, health_p in [(0, 0.001, 0.04), (1, 0.002, 0.20)]:
        for analysis, p_value in [("Overall", overall_p), ("Health Sciences", health_p)]:
            records.append(
                {
                    "replicate": replicate,
                    "analysis": analysis,
                    "pair_filter": "all_pairs",
                    "wilcoxon_p": p_value,
                    "author_s_positive_rate": 0.4,
                    "control_s_positive_rate": 0.2,
                }
            )
    for domain in ["Life Sciences", "Physical Sciences", "Social Sciences"]:
        for replicate in [0, 1]:
            records.append(
                {
                    "replicate": replicate,
                    "analysis": domain,
                    "pair_filter": "all_pairs",
                    "wilcoxon_p": np.nan,
                    "author_s_positive_rate": 0.0,
                    "control_s_positive_rate": 0.0,
                }
            )

    enriched = _enrich_stability_with_fdr({"replicate_records": records})

    assert enriched["fdr_summaries"]["Overall"]["prop_fdr_below_0_05"] == 1.0
    assert enriched["fdr_summaries"]["Health Sciences"]["prop_fdr_below_0_05"] == 0.5
    assert np.isnan(
        enriched["fdr_summaries"]["Social Sciences"]["prop_fdr_below_0_05"]
    )
