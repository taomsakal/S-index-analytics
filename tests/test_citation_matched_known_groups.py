import numpy as np
import pandas as pd

from utilities.run_citation_matched_known_groups import (
    CITATION_CALIPER_LOG,
    perform_citation_matching,
    s_positive_wilcoxon_check,
)


def _prepared_fixture() -> pd.DataFrame:
    rows = []
    # Authors in two domains; the last Life author has no control within 10%.
    for i, (domain, cites) in enumerate([
        ("Life Sciences", 100), ("Life Sciences", 1000), ("Life Sciences", 50000),
        ("Physical Sciences", 0), ("Physical Sciences", 300),
    ]):
        rows.append({
            "openalex_id": f"A{i:03d}", "primary_domain": domain,
            "is_data_paper_author": True, "cited_by_count": cites,
            "h_index": 5 + i, "s_index": i,
        })
    for i, (domain, cites) in enumerate([
        ("Life Sciences", 95), ("Life Sciences", 105), ("Life Sciences", 1050),
        ("Life Sciences", 20000), ("Physical Sciences", 0), ("Physical Sciences", 310),
        ("Physical Sciences", 100), ("no_domain", 300),
    ]):
        rows.append({
            "openalex_id": f"C{i:03d}", "primary_domain": domain,
            "is_data_paper_author": False, "cited_by_count": cites,
            "h_index": 5, "s_index": 0,
        })
    return pd.DataFrame(rows)


def test_citation_matching_respects_domain_caliper_and_no_reuse():
    prepared = _prepared_fixture()
    matched, report = perform_citation_matching(prepared, seed=1)

    assert len(matched) == 4
    assert report["n_excluded_no_controls"] == 1
    assert report["excluded_authors"][0]["author_id"] == "A002"
    assert matched["control_id"].is_unique

    domains = prepared.set_index("openalex_id")["primary_domain"]
    assert (matched["control_id"].map(domains) == matched["domain"]).all()
    log_diff = np.abs(
        np.log1p(matched["author_cited_by"].astype(float))
        - np.log1p(matched["control_cited_by"].astype(float))
    )
    assert (log_diff <= CITATION_CALIPER_LOG).all()


def test_citation_matching_is_reproducible_for_a_seed():
    prepared = _prepared_fixture()
    first, _ = perform_citation_matching(prepared, seed=7)
    second, _ = perform_citation_matching(prepared, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_s_positive_check_matches_all_pairs_when_ties_are_zero_zero():
    matched = pd.DataFrame({
        "domain": ["Life Sciences"] * 8,
        "author_s_index": [0, 0, 0, 5, 3, 8, 2, 9],
        "control_s_index": [0, 0, 0, 0, 1, 0, 4, 1],
    })
    overall = s_positive_wilcoxon_check(matched)[0]

    assert overall["n_s_positive_pairs"] == 5
    assert overall["nonzero_tied_pairs"] == 0
    assert overall["identical"] is True
