"""
Known-groups validation for the S-Index.

Tests whether researchers who publish data papers have higher S-indices than
matched controls with the same primary domain and exactly the same h-index.

Design: exact 1:1 matching on (primary_domain, h_index) without replacement,
so every pair is identical on both matching variables. Authors with no
exact-match control are excluded from the primary analysis and added back by
nearest-neighbor matching in a sensitivity analysis. When several controls
are eligible for an author, one is drawn at random, so the analysis is
repeated across many random draws to show that the conclusion does not depend
on the draw.

Sections: derive variables, matching, diagnostics, analyses, console report,
figures, and the run_known_groups_validation() entry point.
"""

import hashlib
import os
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats

import config
from viz_config import COLORS, FONT, FIGURE_SIZES, LINES, setup_style, save_figure, add_grid

RANDOM_SEED = config.RANDOM_SEED
MIN_PAIRS_FOR_DOMAIN_TEST = config.MIN_PAIRS_FOR_DOMAIN_TEST

ALL_DOMAINS = [domain.name for domain in sorted(config.ALL_DOMAINS, key=lambda d: d.name)]
DOMAIN_SHORT_LABELS = {domain.name: domain.short_name.capitalize() for domain in config.ALL_DOMAINS}
NO_DOMAIN = "no_domain"

DATA_JOURNAL_COLUMNS = ["scientific_data_pubs", "essd_pubs", "gigascience_pubs", "bds_pubs"]

KNOWN_GROUPS_FIGURES_DIR = config.get_known_groups_figures_dir()
KNOWN_GROUPS_DATA_DIR = config.get_known_groups_data_dir()

# Caliper for the nearest-neighbor sensitivity analysis, which adds authors
# lacking an exact (domain, h-index) control: a control qualifies if
# |h_control - h_author| <= max(NN_CALIPER_MIN, NN_CALIPER_FRAC * h_author).
NN_CALIPER_MIN = 2
NN_CALIPER_FRAC = 0.10

# Number of random control draws used to check that the primary result does
# not depend on which eligible controls happen to be drawn.
DEFAULT_STABILITY_REPLICATES = 500


# =============================================================================
# Step 1: Derive Variables
# =============================================================================

def derive_data_paper_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns for known-groups validation.

    Adds:
    - data_paper_count: sum of publications in data journals
    - is_data_paper_author: True if data_paper_count >= 1

    Expects the DATA_JOURNAL_COLUMNS (lowercase, as normalized by data_loader);
    missing columns are treated as 0.
    """
    df = df.copy()
    data_journal_cols = DATA_JOURNAL_COLUMNS

    existing_cols = [c for c in data_journal_cols if c in df.columns]
    if not existing_cols:
        raise KeyError(f"No data journal columns found. Expected any of: {data_journal_cols}")

    missing_cols = [c for c in data_journal_cols if c not in df.columns]
    if missing_cols:
        print(f"Warning: Missing data journal columns (treating as 0): {missing_cols}")
        for col in missing_cols:
            df[col] = 0

    for col in data_journal_cols:
        df[col] = df[col].fillna(0).astype(int)

    df["data_paper_count"] = df[data_journal_cols].sum(axis=1)
    df["is_data_paper_author"] = df["data_paper_count"] >= 1

    return df


def derive_primary_domain(df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Assign primary_domain based on highest publication count.

    Ties are broken deterministically from the researcher ID and seed, so the
    assignment does not depend on row order. If all counts are zero, assign
    "no_domain". An existing complete and valid primary_domain column is kept.

    Expects columns: life_sciences, social_sciences, physical_sciences, health_sciences
    """
    df = df.copy()

    domain_cols = ["health_sciences", "life_sciences", "physical_sciences", "social_sciences"]
    domain_names = ALL_DOMAINS

    valid_domains = set(domain_names + [NO_DOMAIN])
    if "primary_domain" in df.columns:
        existing = df["primary_domain"]
        if existing.notna().all() and existing.isin(valid_domains).all():
            return df

    existing_cols = [c for c in domain_cols if c in df.columns]
    if not existing_cols:
        raise KeyError(f"No domain columns found. Expected any of: {domain_cols}")

    missing_cols = [c for c in domain_cols if c not in df.columns]
    if missing_cols:
        print(f"Warning: Missing domain columns (treating as 0): {missing_cols}")
        for col in missing_cols:
            df[col] = 0

    for col in domain_cols:
        df[col] = df[col].fillna(0).astype(int)

    def get_primary_domain(row):
        counts = [(domain_names[i], row[domain_cols[i]]) for i in range(len(domain_cols))]
        max_count = max(c[1] for c in counts)
        if max_count == 0:
            return NO_DOMAIN
        max_domains = [c[0] for c in counts if c[1] == max_count]
        if len(max_domains) == 1:
            return max_domains[0]
        researcher_id = row.get("openalex_id")
        if pd.isna(researcher_id) or not str(researcher_id).strip():
            return sorted(max_domains)[0]
        digest = hashlib.sha256(
            f"{seed}:{researcher_id}".encode("utf-8")
        ).digest()
        choice_index = int.from_bytes(digest[:8], "big") % len(max_domains)
        return sorted(max_domains)[choice_index]

    df["primary_domain"] = df.apply(get_primary_domain, axis=1)

    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add the data paper and primary domain columns needed for matching."""
    return derive_primary_domain(derive_data_paper_variables(df))


# =============================================================================
# Step 2: Matching
# =============================================================================

def perform_matching(df: pd.DataFrame, seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, dict]:
    """
    Exact 1:1 matching of data paper authors to controls on
    (primary_domain, h_index), without replacement.

    Within each (domain, h-index) cell, min(n_authors, n_controls) pairs are
    formed. When several controls share an author's exact domain and h-index,
    one is drawn at random (seeded). Authors with no exact-match control are
    excluded and listed in the report; perform_matching_with_nn_sensitivity()
    adds them back via nearest-neighbor matching as a sensitivity analysis.

    Every returned pair is identical on domain and h-index by construction.
    The NUMBER of matched pairs is fixed by cell capacities regardless of
    seed; the seed only affects WHICH tied controls are drawn.

    Excludes researchers with primary_domain == "no_domain".

    Returns:
        matched_pairs: DataFrame with columns [author_id, control_id, domain,
                       author_s_index, control_s_index, author_h_index,
                       control_h_index, author_cited_by, control_cited_by,
                       match_type]
        report: dict with matching statistics, including excluded_authors
    """
    if "openalex_id" not in df.columns:
        raise KeyError("Column 'openalex_id' required for matching")

    rng = np.random.default_rng(seed)

    df_valid = df[df["primary_domain"] != NO_DOMAIN]
    authors = df_valid[df_valid["is_data_paper_author"]]
    controls = df_valid[~df_valid["is_data_paper_author"]]

    has_citations = "cited_by_count" in df.columns

    # Sort within cells by openalex_id so results depend only on the seed,
    # not on the row order of the input
    control_cells = {
        key: grp.sort_values("openalex_id")
        for key, grp in controls.groupby(["primary_domain", "h_index"], sort=True)
    }

    pair_frames = []
    excluded_authors = []

    for key, author_cell in authors.groupby(["primary_domain", "h_index"], sort=True):
        domain, h = key
        author_cell = author_cell.sort_values("openalex_id")
        control_cell = control_cells.get(key)
        n_available = 0 if control_cell is None else len(control_cell)
        k = min(len(author_cell), n_available)

        if k < len(author_cell):
            # Not enough exact controls: randomly select which authors match
            order = rng.permutation(len(author_cell))
            matched_authors = author_cell.iloc[order[:k]]
            for _, row in author_cell.iloc[order[k:]].iterrows():
                excluded_authors.append({
                    "author_id": row["openalex_id"],
                    "domain": domain,
                    "h_index": row["h_index"],
                    "s_index": row["s_index"],
                })
        else:
            matched_authors = author_cell

        if k == 0:
            continue

        control_idx = rng.choice(n_available, size=k, replace=False)
        matched_controls = control_cell.iloc[control_idx]

        pair_frames.append(pd.DataFrame({
            "author_id": matched_authors["openalex_id"].to_numpy(),
            "control_id": matched_controls["openalex_id"].to_numpy(),
            "domain": domain,
            "author_s_index": matched_authors["s_index"].to_numpy(),
            "control_s_index": matched_controls["s_index"].to_numpy(),
            "author_h_index": matched_authors["h_index"].to_numpy(),
            "control_h_index": matched_controls["h_index"].to_numpy(),
            "author_cited_by": (matched_authors["cited_by_count"].to_numpy()
                                if has_citations else np.nan),
            "control_cited_by": (matched_controls["cited_by_count"].to_numpy()
                                 if has_citations else np.nan),
            "match_type": "exact",
        }))

    matched_df = (pd.concat(pair_frames, ignore_index=True)
                  if pair_frames else pd.DataFrame())

    report = {
        "n_data_paper_authors_total": len(authors),
        "n_matched": len(matched_df),
        "n_excluded_no_controls": len(excluded_authors),
        "n_excluded_no_domain": int((df["primary_domain"] == NO_DOMAIN).sum()),
        "n_controls_total": len(controls),
        "domain_counts": authors["primary_domain"].value_counts().to_dict(),
        "excluded_authors": excluded_authors,
    }

    return matched_df, report


def perform_matching_with_nn_sensitivity(df: pd.DataFrame,
                                         seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, dict]:
    """
    Sensitivity analysis: exact matching plus nearest-neighbor matches for
    the authors excluded from the primary analysis.

    Each excluded author is paired with the in-domain control closest in
    h-index (among controls not already used), subject to the caliper
    |dh| <= max(NN_CALIPER_MIN, NN_CALIPER_FRAC * author_h). Authors with
    no control inside the caliper remain unmatched.

    Returns:
        combined_df: exact pairs plus NN pairs (match_type distinguishes them)
        report: exact-matching report extended with NN statistics
    """
    matched_df, report = perform_matching(df, seed=seed)
    excluded = report.get("excluded_authors", [])

    report = dict(report)
    report["n_nn_matched"] = 0
    report["n_nn_unmatched"] = len(excluded)
    report["nn_pair_h_deltas"] = []

    if not excluded:
        return matched_df, report

    used_controls = set(matched_df["control_id"]) if len(matched_df) > 0 else set()
    df_valid = df[df["primary_domain"] != NO_DOMAIN]
    available = df_valid[~df_valid["is_data_paper_author"]]
    available = available[~available["openalex_id"].isin(used_controls)]
    available = available.sort_values("openalex_id").set_index("openalex_id")

    author_lookup = df_valid.set_index("openalex_id")

    nn_records = []
    nn_deltas = []
    n_unmatched = 0

    # Match hardest cases first (highest h-index = sparsest control pools)
    for exc in sorted(excluded, key=lambda e: -e["h_index"]):
        pool = available[available["primary_domain"] == exc["domain"]]
        if len(pool) == 0:
            n_unmatched += 1
            continue
        dh = (pool["h_index"] - exc["h_index"]).abs()
        caliper = max(NN_CALIPER_MIN, NN_CALIPER_FRAC * exc["h_index"])
        dh_in_caliper = dh[dh <= caliper]
        if len(dh_in_caliper) == 0:
            n_unmatched += 1
            continue
        control_id = dh_in_caliper.idxmin()
        control = pool.loc[control_id]
        author = author_lookup.loc[exc["author_id"]]
        nn_records.append({
            "author_id": exc["author_id"],
            "control_id": control_id,
            "domain": exc["domain"],
            "author_s_index": author["s_index"],
            "control_s_index": control["s_index"],
            "author_h_index": author["h_index"],
            "control_h_index": control["h_index"],
            "author_cited_by": author.get("cited_by_count", np.nan),
            "control_cited_by": control.get("cited_by_count", np.nan),
            "match_type": "nn",
        })
        nn_deltas.append(abs(author["h_index"] - control["h_index"]))
        available = available.drop(control_id)

    if nn_records:
        combined_df = pd.concat([matched_df, pd.DataFrame(nn_records)],
                                ignore_index=True)
    else:
        combined_df = matched_df

    report["n_nn_matched"] = len(nn_records)
    report["n_nn_unmatched"] = n_unmatched
    report["nn_pair_h_deltas"] = sorted(int(d) for d in nn_deltas)

    return combined_df, report


def matching_stability_analysis(df_prepared: pd.DataFrame,
                                n_replicates: int = DEFAULT_STABILITY_REPLICATES,
                                base_seed: int = RANDOM_SEED) -> dict:
    """
    Re-run the exact matching across many random control draws.

    Exact matching fixes the pair count and guarantees perfect h-index/domain
    balance in every draw. Returns summary statistics across draws plus
    long-form "replicate_records" (one row per draw, analysis, and pair
    filter) covering all pairs and the descriptive S>0 view.
    """
    replicate_records = []

    def add_record(replicate: int, seed: int, analysis: str,
                   subset: pd.DataFrame, pair_filter: str,
                   total_n: int) -> None:
        author_s = subset["author_s_index"].to_numpy(dtype=float)
        control_s = subset["control_s_index"].to_numpy(dtype=float)
        n_pairs = len(subset)

        p_value = np.nan
        enough_pairs = analysis == "Overall" or n_pairs >= MIN_PAIRS_FOR_DOMAIN_TEST
        if pair_filter == "all_pairs" and n_pairs > 0 and enough_pairs:
            p_value = paired_wilcoxon(author_s, control_s)["wilcoxon_p_value"]

        replicate_records.append({
            "replicate": replicate,
            "seed": seed,
            "analysis": analysis,
            "pair_filter": pair_filter,
            "n_pairs": n_pairs,
            "n_total_pairs": total_n,
            "wilcoxon_p": p_value,
            "author_win_rate": float(np.mean(author_s > control_s)) if n_pairs else np.nan,
            "tie_rate": float(np.mean(author_s == control_s)) if n_pairs else np.nan,
            "control_win_rate": float(np.mean(author_s < control_s)) if n_pairs else np.nan,
            "author_s_positive_rate": float(np.mean(author_s > 0)) if n_pairs else np.nan,
            "control_s_positive_rate": float(np.mean(control_s > 0)) if n_pairs else np.nan,
            "author_mean": float(np.mean(author_s)) if n_pairs else np.nan,
            "control_mean": float(np.mean(control_s)) if n_pairs else np.nan,
        })

    for i in range(n_replicates):
        draw_seed = base_seed + i
        matched_df, _ = perform_matching(df_prepared, seed=draw_seed)
        if len(matched_df) == 0:
            return {"error": "No matched pairs in stability replicate"}

        for analysis in ["Overall"] + ALL_DOMAINS:
            subset = (
                matched_df if analysis == "Overall"
                else matched_df[matched_df["domain"] == analysis]
            )
            total_n = len(subset)
            add_record(i, draw_seed, analysis, subset, "all_pairs", total_n)
            s_positive = subset[
                (subset["author_s_index"] > 0)
                | (subset["control_s_index"] > 0)
            ]
            add_record(i, draw_seed, analysis, s_positive, "s_positive", total_n)

    records_df = pd.DataFrame(replicate_records)

    def summarize(analysis: str) -> dict:
        rows = records_df[
            (records_df["analysis"] == analysis)
            & (records_df["pair_filter"] == "all_pairs")
        ]
        valid_p = rows["wilcoxon_p"].dropna().to_numpy(dtype=float)
        win_arr = rows["author_win_rate"].dropna().to_numpy(dtype=float)
        n_arr = rows["n_pairs"].to_numpy(dtype=int)
        return {
            "n_pairs_min": int(n_arr.min()),
            "n_pairs_median": float(np.median(n_arr)),
            "n_pairs_max": int(n_arr.max()),
            "p_median": float(np.median(valid_p)) if len(valid_p) else np.nan,
            "p_min": float(np.min(valid_p)) if len(valid_p) else np.nan,
            "p_max": float(np.max(valid_p)) if len(valid_p) else np.nan,
            "prop_p_below_0_05": float(np.mean(valid_p < 0.05)) if len(valid_p) else np.nan,
            "prop_p_below_0_001": float(np.mean(valid_p < 0.001)) if len(valid_p) else np.nan,
            "win_rate_min": float(win_arr.min()) if len(win_arr) else np.nan,
            "win_rate_median": float(np.median(win_arr)) if len(win_arr) else np.nan,
            "win_rate_max": float(win_arr.max()) if len(win_arr) else np.nan,
        }

    overall = summarize("Overall")
    return {
        "n_replicates": n_replicates,
        "n_pairs": int(overall["n_pairs_median"]),
        **{k: v for k, v in overall.items() if not k.startswith("n_pairs_")},
        "domain_summaries": {
            domain: summarize(domain) for domain in ALL_DOMAINS
        },
        "replicate_records": replicate_records,
    }


def paired_wilcoxon(author_s: np.ndarray, control_s: np.ndarray) -> dict:
    """
    One-sided Wilcoxon signed-rank test that authors exceed their controls.

    Returns wilcoxon_statistic and wilcoxon_p_value, plus a warning when the
    test is degenerate (all differences zero gives p = 1).
    """
    if np.all(author_s == control_s):
        return {
            "wilcoxon_statistic": np.nan,
            "wilcoxon_p_value": 1.0,
            "warning": "All paired differences are zero",
        }
    try:
        stat, p_value = stats.wilcoxon(author_s, control_s, alternative="greater")
    except ValueError as e:
        return {"wilcoxon_statistic": np.nan, "wilcoxon_p_value": np.nan, "warning": str(e)}
    return {"wilcoxon_statistic": stat, "wilcoxon_p_value": p_value}


def pair_summary(author_s: np.ndarray, control_s: np.ndarray) -> dict:
    """Descriptive statistics and pairwise outcome shares for matched pairs."""
    n = len(author_s)
    return {
        "median_s_authors": np.median(author_s),
        "median_s_controls": np.median(control_s),
        "mean_s_authors": np.mean(author_s),
        "mean_s_controls": np.mean(control_s),
        "prop_s_positive_authors": np.mean(author_s > 0),
        "prop_s_positive_controls": np.mean(control_s > 0),
        "prop_pairs_author_wins": np.sum(author_s > control_s) / n,
        "prop_pairs_ties": np.sum(author_s == control_s) / n,
        "prop_pairs_control_wins": np.sum(author_s < control_s) / n,
    }


def paired_direction_tests(author_s: np.ndarray, control_s: np.ndarray) -> dict:
    """Exact paired tests for directional wins and S>0 prevalence."""
    author_s = np.asarray(author_s)
    control_s = np.asarray(control_s)

    author_wins = int(np.sum(author_s > control_s))
    control_wins = int(np.sum(author_s < control_s))
    non_tied = author_wins + control_wins
    sign_p = (
        float(stats.binomtest(
            author_wins, non_tied, p=0.5, alternative="greater"
        ).pvalue)
        if non_tied else 1.0
    )

    author_positive_control_zero = int(np.sum((author_s > 0) & (control_s == 0)))
    author_zero_control_positive = int(np.sum((author_s == 0) & (control_s > 0)))
    positive_discordant = author_positive_control_zero + author_zero_control_positive
    positive_p = (
        float(stats.binomtest(
            author_positive_control_zero,
            positive_discordant,
            p=0.5,
            alternative="greater",
        ).pvalue)
        if positive_discordant else 1.0
    )

    return {
        "sign_test_p_value": sign_p,
        "n_non_tied_pairs": non_tied,
        "author_positive_control_zero": author_positive_control_zero,
        "author_zero_control_positive": author_zero_control_positive,
        "positive_discordant_pairs": positive_discordant,
        "positive_mcnemar_p_value": positive_p,
    }


def s_positive_analysis(matched_df: pd.DataFrame, domain: str = None) -> dict:
    """
    Describe the pairs where at least one member has S > 0.

    This view is descriptive: the Wilcoxon test already discards zero
    differences, so restricting to these pairs does not change it. The
    direction tests in the result are computed on the full matched set.

    Args:
        matched_df: DataFrame of matched pairs
        domain: Optional domain to filter on (None = overall)
    """
    if len(matched_df) == 0 or "domain" not in matched_df.columns:
        return {"error": "No matched pairs to analyze", "domain": domain or "Overall"}

    if domain:
        matched_df = matched_df[matched_df["domain"] == domain].copy()
        if len(matched_df) == 0:
            return {"error": f"No pairs in {domain}", "domain": domain}

    full_author_s = matched_df["author_s_index"].to_numpy()
    full_control_s = matched_df["control_s_index"].to_numpy()
    direction_tests = paired_direction_tests(full_author_s, full_control_s)

    s_positive_pairs = matched_df[
        (matched_df["author_s_index"] > 0) | (matched_df["control_s_index"] > 0)
    ].copy()

    if len(s_positive_pairs) == 0:
        return {
            "error": "No pairs with S > 0",
            "domain": domain or "Overall",
            "n_pairs_total": len(matched_df),
        }

    author_s = s_positive_pairs["author_s_index"].values
    control_s = s_positive_pairs["control_s_index"].values

    results = {
        "domain": domain or "Overall",
        "n_pairs_total": len(matched_df),
        "n_pairs_s_positive": len(s_positive_pairs),
        "pct_pairs_s_positive": len(s_positive_pairs) / len(matched_df) * 100,
        **pair_summary(author_s, control_s),
        **direction_tests,
    }

    results["inference_note"] = (
        "Descriptive subset only; the primary Wilcoxon test already omits "
        "zero paired differences. Use positive_mcnemar_p_value for the "
        "paired S>0 prevalence comparison."
    )
    if len(s_positive_pairs) < 10:
        results["warning"] = f"Small descriptive subset (n={len(s_positive_pairs)})"

    return results


def all_domains_s_positive_analysis(matched_df: pd.DataFrame) -> list[dict]:
    """Run the S>0 descriptive view overall, then for each domain."""
    return [s_positive_analysis(matched_df, domain=domain) for domain in [None] + ALL_DOMAINS]


# =============================================================================
# Step 3: Matching Diagnostics
# =============================================================================

def matching_diagnostics(matched_df: pd.DataFrame) -> dict:
    """
    Verify match quality and covariate balance for matched pairs.

    With exact matching every pair is identical on domain and h-index by
    construction; this verifies that invariant. Citation balance is a
    genuine diagnostic because citations are never matched on.
    """
    if len(matched_df) == 0:
        return {"error": "No matched pairs to analyze"}

    h_delta = (matched_df["author_h_index"] - matched_df["control_h_index"]).abs()

    diagnostics = {
        "n_pairs": len(matched_df),
        "n_exact_h": int((h_delta == 0).sum()),
        "pct_exact_h": float((h_delta == 0).mean() * 100),
        "h_pair_diff_mean": float(h_delta.mean()),
        "h_pair_diff_max": float(h_delta.max()),
        "mean_h_authors": matched_df["author_h_index"].mean(),
        "mean_h_controls": matched_df["control_h_index"].mean(),
        "median_h_authors": matched_df["author_h_index"].median(),
        "median_h_controls": matched_df["control_h_index"].median(),
        "domain_distribution": matched_df["domain"].value_counts().to_dict(),
    }

    h_diff = abs(diagnostics["mean_h_authors"] - diagnostics["mean_h_controls"])
    diagnostics["h_mean_difference"] = h_diff
    diagnostics["balance_ok"] = h_diff < 2.0

    # Citation balance (not matched on, so this is a genuine balance check)
    if "author_cited_by" in matched_df.columns and matched_df["author_cited_by"].notna().any():
        mean_ca = matched_df["author_cited_by"].mean()
        mean_cc = matched_df["control_cited_by"].mean()
        pooled_sd = np.sqrt(
            (matched_df["author_cited_by"].var() + matched_df["control_cited_by"].var()) / 2
        )
        diagnostics["mean_citations_authors"] = mean_ca
        diagnostics["mean_citations_controls"] = mean_cc
        diagnostics["citation_smd"] = (
            float((mean_ca - mean_cc) / pooled_sd) if pooled_sd > 0 else np.nan
        )
        smd = diagnostics["citation_smd"]
        diagnostics["citation_balance_ok"] = (
            bool(abs(smd) < 0.1) if not np.isnan(smd) else None
        )

    return diagnostics


# =============================================================================
# Step 4: Primary Analysis
# =============================================================================

def primary_analysis(matched_df: pd.DataFrame) -> dict:
    """
    Compare S-index between data paper authors and matched controls using a
    one-sided paired Wilcoxon signed-rank test and exact direction tests.
    """
    if len(matched_df) == 0:
        return {"error": "No matched pairs to analyze"}

    author_s = matched_df["author_s_index"].values
    control_s = matched_df["control_s_index"].values

    results = {
        "n_pairs": len(matched_df),
        **pair_summary(author_s, control_s),
        **paired_direction_tests(author_s, control_s),
        **paired_wilcoxon(author_s, control_s),
    }
    if len(matched_df) < 10 and "warning" not in results:
        results["warning"] = (
            f"Small sample size (n={len(matched_df)}): Wilcoxon test has low "
            "power, interpret with caution"
        )

    # Percent difference in medians
    if results["median_s_controls"] > 0:
        results["percent_difference_median"] = (
            (results["median_s_authors"] - results["median_s_controls"])
            / results["median_s_controls"] * 100
        )
    else:
        results["raw_difference_median"] = results["median_s_authors"] - results["median_s_controls"]

    return results


# =============================================================================
# Step 5: Per-Domain Analyses
# =============================================================================

def domain_matched_analysis(matched_df: pd.DataFrame, domain: str,
                            min_pairs_for_test: int = MIN_PAIRS_FOR_DOMAIN_TEST) -> dict:
    """
    Matched comparison within one domain.

    Args:
        matched_df: DataFrame of matched pairs from perform_matching()
        domain: Domain name to filter on
        min_pairs_for_test: Minimum pairs required for the Wilcoxon test

    Returns:
        dict with comparison statistics or an error message
    """
    if len(matched_df) == 0 or "domain" not in matched_df.columns:
        return {"domain": domain, "error": f"No matched pairs in {domain}"}

    domain_pairs = matched_df[matched_df["domain"] == domain]

    if len(domain_pairs) == 0:
        return {"domain": domain, "error": f"No matched pairs in {domain}"}

    if len(domain_pairs) < 5:
        return {
            "domain": domain,
            "error": f"Insufficient matched pairs: {len(domain_pairs)} (need at least 5)"
        }

    author_s = domain_pairs["author_s_index"].values
    control_s = domain_pairs["control_s_index"].values

    results = {
        "domain": domain,
        "n_pairs": len(domain_pairs),
        **pair_summary(author_s, control_s),
        **paired_direction_tests(author_s, control_s),
    }

    if len(domain_pairs) < min_pairs_for_test:
        results["wilcoxon_statistic"] = np.nan
        results["wilcoxon_p_value"] = np.nan
        results["warning"] = (
            f"Sample too small for reliable inference "
            f"(n={len(domain_pairs)} < {min_pairs_for_test})"
        )
    else:
        results.update(paired_wilcoxon(author_s, control_s))

    return results


def all_domains_matched_analysis(matched_df: pd.DataFrame) -> list[dict]:
    """Run the matched analysis for each domain."""
    return [domain_matched_analysis(matched_df, domain) for domain in ALL_DOMAINS]


def domain_unmatched_analysis(df: pd.DataFrame, domain: str) -> dict:
    """
    Supplementary unmatched comparison within a specific domain.

    Uses Mann-Whitney U test (Wilcoxon rank-sum).

    Args:
        df: Prepared DataFrame with is_data_paper_author and primary_domain columns
        domain: Domain name to filter on

    Returns:
        dict with comparison statistics or error message
    """
    domain_df = df[df["primary_domain"] == domain].copy()

    if len(domain_df) == 0:
        return {"domain": domain, "error": f"No {domain} researchers in data"}

    authors = domain_df[domain_df["is_data_paper_author"]]["s_index"]
    non_authors = domain_df[~domain_df["is_data_paper_author"]]["s_index"]

    if len(authors) < 2 or len(non_authors) < 2:
        return {
            "domain": domain,
            "error": f"Insufficient sample sizes: {len(authors)} authors, {len(non_authors)} non-authors"
        }

    results = {
        "domain": domain,
        "n_authors": len(authors),
        "n_non_authors": len(non_authors),
        "median_s_authors": authors.median(),
        "median_s_non_authors": non_authors.median(),
        "mean_s_authors": authors.mean(),
        "mean_s_non_authors": non_authors.mean(),
    }

    try:
        stat, p_value = stats.mannwhitneyu(authors, non_authors, alternative='greater')
        results["mannwhitney_statistic"] = stat
        results["mannwhitney_p_value"] = p_value
    except ValueError as e:
        results["mannwhitney_statistic"] = np.nan
        results["mannwhitney_p_value"] = np.nan
        results["warning"] = str(e)

    return results


def all_domains_unmatched_analysis(df: pd.DataFrame) -> list[dict]:
    """Run the supplementary unmatched analysis for each domain."""
    return [domain_unmatched_analysis(df, domain) for domain in ALL_DOMAINS]


# =============================================================================
# Step 6: Outputs
# =============================================================================

def print_report(matching_report: dict, diagnostics: dict,
                 primary_results: dict, domain_matched_results: list[dict],
                 domain_unmatched_results: list[dict] = None,
                 s_positive_results: dict = None,
                 stability_results: dict = None,
                 nn_sensitivity_report: dict = None,
                 nn_sensitivity_primary: dict = None):
    """
    Print formatted console output of all results.
    """
    print("=" * 70)
    print("KNOWN-GROUPS VALIDATION: DATA PAPER AUTHORS VS MATCHED CONTROLS")
    print("=" * 70)
    print()

    # Matching summary
    print("-" * 50)
    print("MATCHING SUMMARY (exact match on domain + h-index)")
    print("-" * 50)
    print(f"Data paper authors (excl. no_domain): {matching_report['n_data_paper_authors_total']}")
    print(f"Matched (exact domain + h-index):     {matching_report['n_matched']}")
    print(f"Excluded (no exact-match control):    {matching_report['n_excluded_no_controls']}")
    print(f"Excluded (no_domain):                 {matching_report['n_excluded_no_domain']}")
    excluded = matching_report.get("excluded_authors", [])
    if excluded:
        print("  Excluded authors (domain, h-index):")
        for exc in sorted(excluded, key=lambda e: (e["domain"], e["h_index"])):
            print(f"    {exc['domain']}: h={exc['h_index']:.0f}")
    print()
    print("Data paper authors by domain:")
    for domain, count in sorted(matching_report.get('domain_counts', {}).items()):
        print(f"  {domain}: {count}")
    print()

    # Matching diagnostics
    print("-" * 50)
    print("MATCHING DIAGNOSTICS")
    print("-" * 50)
    if "error" in diagnostics:
        print(f"Error: {diagnostics['error']}")
    else:
        if "pct_exact_h" in diagnostics:
            print(f"Pairs identical on h-index: {diagnostics['n_exact_h']} of "
                  f"{diagnostics['n_pairs']} ({diagnostics['pct_exact_h']:.1f}%)")
        print(f"Mean h-index - Authors:   {diagnostics['mean_h_authors']:.2f}")
        print(f"Mean h-index - Controls:  {diagnostics['mean_h_controls']:.2f}")
        print(f"Median h-index - Authors: {diagnostics['median_h_authors']:.1f}")
        print(f"Median h-index - Controls:{diagnostics['median_h_controls']:.1f}")
        if "citation_smd" in diagnostics:
            print("Citation balance (citations are NOT matched on):")
            print(f"  Mean citations - Authors:  {diagnostics['mean_citations_authors']:.0f}")
            print(f"  Mean citations - Controls: {diagnostics['mean_citations_controls']:.0f}")
            print(f"  Standardized mean difference: {diagnostics['citation_smd']:.3f} "
                  f"(|SMD| < 0.1 indicates balance)")
        print(f"Balance OK: {diagnostics['balance_ok']}")
        if not diagnostics['balance_ok']:
            print(f"  WARNING: Mean h-index differs by {diagnostics['h_mean_difference']:.2f}")
    print()

    # Primary analysis - OVERALL
    print("-" * 50)
    print("PRIMARY ANALYSIS: OVERALL MATCHED COMPARISON")
    print("-" * 50)
    if "error" in primary_results:
        print(f"Error: {primary_results['error']}")
    else:
        print(f"Number of matched pairs: {primary_results['n_pairs']}")
        print()
        print("S-index comparison:")
        print(f"  Median - Authors:  {primary_results['median_s_authors']:.1f}")
        print(f"  Median - Controls: {primary_results['median_s_controls']:.1f}")
        print(f"  Mean - Authors:    {primary_results['mean_s_authors']:.2f}")
        print(f"  Mean - Controls:   {primary_results['mean_s_controls']:.2f}")
        print()
        print(f"  Prop. S > 0 - Authors:  {primary_results['prop_s_positive_authors']:.1%}")
        print(f"  Prop. S > 0 - Controls: {primary_results['prop_s_positive_controls']:.1%}")
        print()
        print("Pairwise outcomes:")
        print(f"  Author S > Control S: {primary_results['prop_pairs_author_wins']:.1%}")
        print(f"  Ties:                 {primary_results['prop_pairs_ties']:.1%}")
        print(f"  Author S < Control S: {primary_results['prop_pairs_control_wins']:.1%}")
        print()
        print("Wilcoxon signed-rank test (one-sided, authors > controls):")
        print(f"  Statistic: {primary_results.get('wilcoxon_statistic', 'N/A')}")
        print(f"  p-value:   {primary_results.get('wilcoxon_p_value', 'N/A')}")
        print("Exact paired directional tests:")
        print(f"  Wins vs losses p-value: {primary_results['sign_test_p_value']:.2e}")
        print(
            "  S>0 discordance (author+/control0 vs author0/control+): "
            f"{primary_results['author_positive_control_zero']} vs "
            f"{primary_results['author_zero_control_positive']}, "
            f"p={primary_results['positive_mcnemar_p_value']:.2e}"
        )
        if "warning" in primary_results:
            print(f"  Warning: {primary_results['warning']}")
        print()
        if "percent_difference_median" in primary_results:
            print(f"Percent difference in medians: {primary_results['percent_difference_median']:.1f}%")
        elif "raw_difference_median" in primary_results:
            print(f"Raw difference in medians: {primary_results['raw_difference_median']:.1f}")
    print()

    # S>0 RESTRICTED ANALYSIS
    if s_positive_results and "error" not in s_positive_results:
        print("-" * 50)
        print("S>0 DESCRIPTIVE VIEW")
        print("-" * 50)
        print("Analyzing only pairs where at least one member has S > 0.")
        print(f"Pairs with S>0: {s_positive_results['n_pairs_s_positive']} of {s_positive_results['n_pairs_total']} ({s_positive_results['pct_pairs_s_positive']:.1f}%)")
        print()
        print(f"  Median S - Authors:  {s_positive_results['median_s_authors']:.1f}")
        print(f"  Median S - Controls: {s_positive_results['median_s_controls']:.1f}")
        print(f"  Mean S - Authors:    {s_positive_results['mean_s_authors']:.2f}")
        print(f"  Mean S - Controls:   {s_positive_results['mean_s_controls']:.2f}")
        print()
        print("Pairwise outcomes (S>0 pairs only):")
        print(f"  Author S > Control S: {s_positive_results['prop_pairs_author_wins']:.1%}")
        print(f"  Ties:                 {s_positive_results['prop_pairs_ties']:.1%}")
        print(f"  Author S < Control S: {s_positive_results['prop_pairs_control_wins']:.1%}")
        print()
        print("Inference: descriptive subset only; zero differences are already")
        print("omitted by the primary Wilcoxon test.")
        if "warning" in s_positive_results:
            print(f"Warning: {s_positive_results['warning']}")
        print()

    # STABILITY ACROSS RANDOM CONTROL DRAWS
    if stability_results and "error" not in stability_results:
        print("-" * 50)
        print("STABILITY ACROSS RANDOM CONTROL DRAWS")
        print("-" * 50)
        print("Ties among eligible controls are broken at random; the analysis")
        print("is repeated across independent draws to show the conclusion does")
        print("not depend on which controls were drawn.")
        print()
        print(f"Replicates: {stability_results['n_replicates']} "
              f"(each with {stability_results['n_pairs']} pairs)")
        print(f"Wilcoxon p: median {stability_results['p_median']:.2e}, "
              f"range [{stability_results['p_min']:.2e}, {stability_results['p_max']:.2e}]")
        print(f"Replicates with p < 0.001: {stability_results['prop_p_below_0_001']:.1%}")
        print(f"Author win rate: {stability_results['win_rate_min']:.1%} - "
              f"{stability_results['win_rate_max']:.1%}")
        print("Per-domain draw stability (all exact pairs):")
        for domain, summary in stability_results.get("domain_summaries", {}).items():
            if np.isnan(summary["p_median"]):
                p_summary = "not tested (insufficient pairs)"
            else:
                p_summary = (
                    f"p median {summary['p_median']:.2e}; "
                    f"p<0.05 in {summary['prop_p_below_0_05']:.1%} of draws"
                )
            print(
                f"  {domain}: wins {summary['win_rate_min']:.1%}-"
                f"{summary['win_rate_max']:.1%}; {p_summary}"
            )
        print()

    # SENSITIVITY: NN MATCHING FOR AUTHORS WITHOUT EXACT MATCHES
    if nn_sensitivity_report is not None and nn_sensitivity_primary is not None:
        print("-" * 50)
        print("SENSITIVITY: INCLUDING AUTHORS WITHOUT EXACT MATCHES")
        print("-" * 50)
        print("Authors lacking an exact (domain, h-index) control are added via")
        print(f"nearest-neighbor matching with caliper |dh| <= "
              f"max({NN_CALIPER_MIN}, {NN_CALIPER_FRAC:.0%} of author h).")
        print()
        print(f"Added NN pairs: {nn_sensitivity_report.get('n_nn_matched', 0)} "
              f"(|dh| values: {nn_sensitivity_report.get('nn_pair_h_deltas', [])})")
        print(f"Still unmatched (outside caliper): "
              f"{nn_sensitivity_report.get('n_nn_unmatched', 0)}")
        if "error" not in nn_sensitivity_primary:
            print(f"Pairs analyzed: {nn_sensitivity_primary['n_pairs']}")
            print(f"  Author wins: {nn_sensitivity_primary['prop_pairs_author_wins']:.1%}, "
                  f"Ties: {nn_sensitivity_primary['prop_pairs_ties']:.1%}, "
                  f"Control wins: {nn_sensitivity_primary['prop_pairs_control_wins']:.1%}")
            p_val = nn_sensitivity_primary.get('wilcoxon_p_value')
            if p_val is not None and not np.isnan(p_val):
                print(f"  Wilcoxon p-value: {p_val:.6f}")
            else:
                print("  Wilcoxon p-value: N/A")
        print()

    # Domain-level MATCHED analyses
    print("-" * 50)
    print("PER-DOMAIN MATCHED ANALYSES")
    print("-" * 50)
    print("Matched comparisons within each domain (Wilcoxon signed-rank test).")
    print()

    for result in domain_matched_results:
        domain = result["domain"]
        print(f"  {domain}:")
        if "error" in result:
            print(f"    {result['error']}")
        else:
            print(f"    Matched pairs: {result['n_pairs']}")
            print(f"    Median S - Authors: {result['median_s_authors']:.1f}, Controls: {result['median_s_controls']:.1f}")
            print(f"    Author wins: {result['prop_pairs_author_wins']:.1%}, Ties: {result['prop_pairs_ties']:.1%}, Control wins: {result['prop_pairs_control_wins']:.1%}")
            p_val = result.get('wilcoxon_p_value')
            if p_val is not None and not np.isnan(p_val):
                print(f"    Wilcoxon p-value: {p_val:.4f}")
            else:
                print("    Wilcoxon p-value: N/A")
            if "warning" in result:
                print(f"    Warning: {result['warning']}")
        print()

    # Domain-level unmatched analyses (supplementary)
    if domain_unmatched_results:
        print("-" * 50)
        print("SUPPLEMENTARY: DOMAIN UNMATCHED ANALYSES")
        print("-" * 50)
        print("NOTE: These are unmatched analyses - shown for reference only.")
        print()

        for result in domain_unmatched_results:
            domain = result["domain"]
            print(f"  {domain}:")
            if "error" in result:
                print(f"    Error: {result['error']}")
            else:
                print(f"    N authors: {result['n_authors']}, N non-authors: {result['n_non_authors']}")
                print(f"    Median S - Authors: {result['median_s_authors']:.1f}, Non-authors: {result['median_s_non_authors']:.1f}")
                p_val = result.get('mannwhitney_p_value')
                if p_val is not None and not np.isnan(p_val):
                    print(f"    Mann-Whitney U p-value: {p_val:.4f}")
                else:
                    print("    Mann-Whitney U p-value: N/A")
                if "warning" in result:
                    print(f"    Warning: {result['warning']}")
            print()

    print("=" * 70)


def compile_results_dict(matching_report: dict, diagnostics: dict,
                         primary_results: dict, domain_matched_results: list[dict],
                         domain_unmatched_results: list[dict] = None) -> dict:
    """
    Compile all results into a single dictionary for programmatic use.
    """
    results = {}

    # Matching stats
    results["n_data_paper_authors_total"] = matching_report.get("n_data_paper_authors_total")
    results["n_matched"] = matching_report.get("n_matched")
    results["n_excluded_no_controls"] = matching_report.get("n_excluded_no_controls")
    results["n_excluded_no_domain"] = matching_report.get("n_excluded_no_domain")

    # Primary results (overall matched)
    if "error" not in primary_results:
        results["median_s_authors"] = primary_results.get("median_s_authors")
        results["median_s_controls"] = primary_results.get("median_s_controls")
        results["mean_s_authors"] = primary_results.get("mean_s_authors")
        results["mean_s_controls"] = primary_results.get("mean_s_controls")
        results["prop_s_positive_authors"] = primary_results.get("prop_s_positive_authors")
        results["prop_s_positive_controls"] = primary_results.get("prop_s_positive_controls")
        results["prop_pairs_author_wins"] = primary_results.get("prop_pairs_author_wins")
        results["prop_pairs_ties"] = primary_results.get("prop_pairs_ties")
        results["prop_pairs_control_wins"] = primary_results.get("prop_pairs_control_wins")
        results["wilcoxon_statistic"] = primary_results.get("wilcoxon_statistic")
        results["wilcoxon_p_value"] = primary_results.get("wilcoxon_p_value")
        results["percent_difference_median"] = primary_results.get("percent_difference_median")

    # Diagnostics
    if "error" not in diagnostics:
        results["mean_h_authors"] = diagnostics.get("mean_h_authors")
        results["mean_h_controls"] = diagnostics.get("mean_h_controls")
        results["median_h_authors"] = diagnostics.get("median_h_authors")
        results["median_h_controls"] = diagnostics.get("median_h_controls")

    # Domain-level MATCHED results
    results["domain_matched"] = {}
    for dr in domain_matched_results:
        domain_key = dr["domain"].lower().replace(" ", "_")
        if "error" not in dr:
            results["domain_matched"][domain_key] = {
                "n_pairs": dr.get("n_pairs"),
                "median_s_authors": dr.get("median_s_authors"),
                "median_s_controls": dr.get("median_s_controls"),
                "prop_pairs_author_wins": dr.get("prop_pairs_author_wins"),
                "wilcoxon_p_value": dr.get("wilcoxon_p_value"),
            }
        else:
            results["domain_matched"][domain_key] = {"error": dr["error"]}

    # Domain-level unmatched results (supplementary)
    if domain_unmatched_results:
        results["domain_unmatched"] = {}
        for dr in domain_unmatched_results:
            domain_key = dr["domain"].lower().replace(" ", "_")
            if "error" not in dr:
                results["domain_unmatched"][domain_key] = {
                    "n_authors": dr.get("n_authors"),
                    "n_non_authors": dr.get("n_non_authors"),
                    "median_s_authors": dr.get("median_s_authors"),
                    "median_s_non_authors": dr.get("median_s_non_authors"),
                    "mannwhitney_p_value": dr.get("mannwhitney_p_value"),
                }
            else:
                results["domain_unmatched"][domain_key] = {"error": dr["error"]}

    return results


def save_matched_pairs(matched_df: pd.DataFrame, output_path: str = None):
    """Save matched pairs to CSV."""
    if output_path is None:
        output_path = f"{KNOWN_GROUPS_DATA_DIR}/matched_pairs.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matched_df.to_csv(output_path, index=False)
    print(f"Matched pairs saved to: {output_path}")


# =============================================================================
# Step 7: Figures
# =============================================================================

def _bootstrap_mean_ci(data: np.ndarray, rng: np.random.Generator,
                       n_boot: int = 1000, empty_value: float = 0.1) -> tuple:
    """Mean with a 95% percentile-bootstrap CI: (mean, (low, high))."""
    if len(data) == 0:
        return empty_value, (empty_value, empty_value)
    boot_means = [np.mean(rng.choice(data, len(data), replace=True)) for _ in range(n_boot)]
    return np.mean(data), (np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5))


def _create_styled_boxplot(ax, data_list: list, labels: list):
    """
    Create a styled boxplot with consistent formatting.

    Args:
        ax: Matplotlib axis
        data_list: List of arrays to plot (e.g., [author_s, control_s])
        labels: List of labels for each box
    """
    color_primary = mcolors.to_rgba(COLORS["box_secondary"], alpha=0.65)
    color_secondary = mcolors.to_rgba(COLORS["box_primary"], alpha=0.65)

    bp = ax.boxplot(
        data_list,
        tick_labels=labels,
        patch_artist=True,
        medianprops=dict(color=COLORS["median"], linewidth=LINES["box_median"]),
        whiskerprops=dict(color="#666666", linewidth=LINES["box_whisker"]),
        capprops=dict(color="#666666", linewidth=LINES["box_whisker"]),
        boxprops=dict(linewidth=LINES["box_whisker"], edgecolor="#555555"),
        flierprops=dict(marker='o', markersize=3, markerfacecolor=COLORS["primary"],
                        markeredgecolor="white", markeredgewidth=0.3, alpha=0.5)
    )

    # Apply colors to boxes
    colors = [color_primary, color_secondary]
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[i % len(colors)])

    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    add_grid(ax, axis='y')

    return bp


def _create_differences_histogram(ax, differences: np.ndarray):
    """
    Create a styled histogram of paired differences.

    Args:
        ax: Matplotlib axis
        differences: Array of difference values
    """
    n_bins = max(5, min(25, int(np.ceil(np.log2(len(differences)) + 1))))

    ax.hist(differences, bins=n_bins, color=COLORS["positive"], alpha=0.75,
            edgecolor="#F0F0F0", linewidth=LINES["hist_edge"])

    ax.axvline(x=0, color=COLORS["threshold"], linestyle="--",
               linewidth=LINES["reference"], label="No difference")
    ax.axvline(x=np.median(differences), color=COLORS["median"], linestyle="-",
               linewidth=LINES["reference"], label=f"Median: {np.median(differences):.1f}")

    ax.set_xlabel("S-index Difference (Author \u2212 Control)")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    add_grid(ax, axis='y')


def plot_s_index_comparison(matched_df: pd.DataFrame, output_path: str = None):
    """
    Side-by-side boxplot and histogram of S-index differences.
    (a) Boxplot comparing authors vs controls
    (b) Histogram of paired differences
    """
    setup_style()

    if output_path is None:
        output_path = f"{KNOWN_GROUPS_FIGURES_DIR}/overall_comparison"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if len(matched_df) == 0:
        print("No matched pairs to plot")
        return

    author_s = matched_df["author_s_index"].values
    control_s = matched_df["control_s_index"].values
    differences = author_s - control_s

    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["double"])

    ax1 = axes[0]
    _create_styled_boxplot(ax1, [author_s, control_s], ["Authors", "Controls"])
    ax1.set_ylabel("S-index")
    ax1.set_title(f"(a) S-index by Group (n={len(matched_df)})")

    ax2 = axes[1]
    _create_differences_histogram(ax2, differences)
    ax2.set_title("(b) Paired Differences")

    plt.tight_layout()

    save_figure(fig, str(output_path).removesuffix('.png'))
    plt.close()


def plot_domain_comparisons(matched_df: pd.DataFrame, output_dir: str = None):
    """
    Create comparison plots for each domain.
    """
    setup_style()

    if output_dir is None:
        output_dir = KNOWN_GROUPS_FIGURES_DIR

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if len(matched_df) == 0:
        print("No matched pairs to plot")
        return

    for domain in ALL_DOMAINS:
        domain_df = matched_df[matched_df["domain"] == domain]

        if len(domain_df) < 5:
            print(f"  Skipping {domain}: only {len(domain_df)} pairs")
            continue

        author_s = domain_df["author_s_index"].values
        control_s = domain_df["control_s_index"].values
        differences = author_s - control_s

        fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["double"])

        ax1 = axes[0]
        _create_styled_boxplot(ax1, [author_s, control_s], ["Authors", "Controls"])
        ax1.set_ylabel("S-index")
        ax1.set_title(f"(a) {domain} (n={len(domain_df)})")

        ax2 = axes[1]
        _create_differences_histogram(ax2, differences)
        ax2.set_title("(b) Paired Differences")

        plt.tight_layout()

        short = DOMAIN_SHORT_LABELS[domain].lower()
        save_figure(fig, str(output_path / f"{short}_comparison"))
        plt.close()


def plot_known_groups_summary(matched_df: pd.DataFrame, output_path: str = None):
    """
    Bar chart of mean S-index for authors vs controls, overall and by domain.

    Each group shows authors and controls for all pairs, then for members
    with S > 0 only. Error bars are 95% bootstrap confidence intervals. The
    y-axis is logarithmic, so bars and intervals are floored at small
    positive values.

    Args:
        matched_df: DataFrame of matched pairs from perform_matching()
        output_path: Path to save the figure (without extension)
    """
    setup_style()

    if output_path is None:
        output_path = f"{KNOWN_GROUPS_FIGURES_DIR}/summary"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if len(matched_df) == 0:
        print("No matched pairs to plot")
        return None

    COLOR_AUTHOR = COLORS["highlight"]
    COLOR_CONTROL = COLORS["primary"]

    categories = ["Overall"] + ALL_DOMAINS
    short_labels = {"Overall": "Overall", **DOMAIN_SHORT_LABELS}

    n_bootstrap = 1000
    rng = np.random.default_rng(RANDOM_SEED)

    # Collect data for all categories
    all_data = []
    for cat in categories:
        if cat == "Overall":
            subset = matched_df
        else:
            subset = matched_df[matched_df["domain"] == cat]

        n = len(subset)
        if n < 3:
            all_data.append({
                'label': short_labels[cat], 'n': n,
                'auth_all': 0.1, 'ctrl_all': 0.1, 'auth_pos': 0.1, 'ctrl_pos': 0.1,
                'auth_all_ci': (0.1, 0.1), 'ctrl_all_ci': (0.1, 0.1),
                'auth_pos_ci': (0.1, 0.1), 'ctrl_pos_ci': (0.1, 0.1),
            })
            continue

        author_s = subset["author_s_index"].values
        control_s = subset["control_s_index"].values
        author_pos = author_s[author_s > 0]
        control_pos = control_s[control_s > 0]

        auth_all_mean, auth_all_ci = _bootstrap_mean_ci(author_s, rng, n_bootstrap)
        ctrl_all_mean, ctrl_all_ci = _bootstrap_mean_ci(control_s, rng, n_bootstrap)
        auth_pos_mean, auth_pos_ci = _bootstrap_mean_ci(author_pos, rng, n_bootstrap)
        ctrl_pos_mean, ctrl_pos_ci = _bootstrap_mean_ci(control_pos, rng, n_bootstrap)

        all_data.append({
            'label': short_labels[cat], 'n': n,
            'auth_all': max(auth_all_mean, 0.1), 'ctrl_all': max(ctrl_all_mean, 0.1),
            'auth_pos': max(auth_pos_mean, 0.1), 'ctrl_pos': max(ctrl_pos_mean, 0.1),
            'auth_all_ci': (max(auth_all_ci[0], 0.05), max(auth_all_ci[1], 0.1)),
            'ctrl_all_ci': (max(ctrl_all_ci[0], 0.05), max(ctrl_all_ci[1], 0.1)),
            'auth_pos_ci': (max(auth_pos_ci[0], 0.05), max(auth_pos_ci[1], 0.1)),
            'ctrl_pos_ci': (max(ctrl_pos_ci[0], 0.05), max(ctrl_pos_ci[1], 0.1)),
        })

    # Fonts are scaled up because this figure is twice the double-column width
    scale = 1.8
    fig, ax = plt.subplots(figsize=(14, 6.0))
    ax.tick_params(axis='both', labelsize=FONT['size_tick'] * scale)

    # Bar layout: 4 bars per group with spacing
    # Order: Auth(all), Ctrl(all), Auth(S>0), Ctrl(S>0)
    n_groups = len(all_data)
    bar_width = 0.20
    pair_gap = 0.10       # Gap between author-control pairs
    group_gap = 0.35      # Gap between domains
    group_width = 4 * bar_width + pair_gap + group_gap

    error_kw_style = {'linewidth': 0.8, 'capthick': 0.8, 'color': '#444444'}

    for i, d in enumerate(all_data):
        x_base = i * group_width

        # Pair 1: All researchers (solid bars)
        # Authors (all)
        yerr_auth_all = [[d['auth_all'] - d['auth_all_ci'][0]],
                         [d['auth_all_ci'][1] - d['auth_all']]]
        ax.bar(x_base, d['auth_all'], bar_width,
               color=COLOR_AUTHOR, alpha=0.88, edgecolor='white', linewidth=0.5,
               yerr=yerr_auth_all, capsize=2.5, error_kw=error_kw_style)

        # Controls (all)
        yerr_ctrl_all = [[d['ctrl_all'] - d['ctrl_all_ci'][0]],
                         [d['ctrl_all_ci'][1] - d['ctrl_all']]]
        ax.bar(x_base + bar_width, d['ctrl_all'], bar_width,
               color=COLOR_CONTROL, alpha=0.88, edgecolor='white', linewidth=0.5,
               yerr=yerr_ctrl_all, capsize=2.5, error_kw=error_kw_style)

        # Pair 2: S>0 only (lighter shade with border)
        x_offset = 2 * bar_width + pair_gap

        # Authors (S>0)
        yerr_auth_pos = [[d['auth_pos'] - d['auth_pos_ci'][0]],
                         [d['auth_pos_ci'][1] - d['auth_pos']]]
        ax.bar(x_base + x_offset, d['auth_pos'], bar_width,
               color=COLOR_AUTHOR, alpha=0.55, edgecolor=COLOR_AUTHOR, linewidth=1.0,
               yerr=yerr_auth_pos, capsize=2.5, error_kw=error_kw_style)

        # Controls (S>0)
        yerr_ctrl_pos = [[d['ctrl_pos'] - d['ctrl_pos_ci'][0]],
                         [d['ctrl_pos_ci'][1] - d['ctrl_pos']]]
        ax.bar(x_base + x_offset + bar_width, d['ctrl_pos'], bar_width,
               color=COLOR_CONTROL, alpha=0.55, edgecolor=COLOR_CONTROL, linewidth=1.0,
               yerr=yerr_ctrl_pos, capsize=2.5, error_kw=error_kw_style)

    ax.set_yscale('log')
    ax.set_ylim(bottom=0.05)

    # Set x-ticks at group centers
    group_centers = [i * group_width + 1.5 * bar_width + pair_gap/2 for i in range(n_groups)]
    ax.set_xticks(group_centers)
    ax.set_xticklabels([f"{d['label']}\n(n={d['n']})" for d in all_data])

    ax.set_ylabel("Mean S-index (log scale)", fontsize=FONT['size_label'] * scale)
    ax.set_title("Known-Groups Validation: Data Paper Authors vs Matched Controls",
                fontsize=FONT["size_title"] * scale, fontweight=FONT["weight_title"], pad=12)

    legend_elements = [
        Patch(facecolor=COLOR_AUTHOR, alpha=0.88, edgecolor='white', label='Authors (all)'),
        Patch(facecolor=COLOR_CONTROL, alpha=0.88, edgecolor='white', label='Controls (all)'),
        Patch(facecolor=COLOR_AUTHOR, alpha=0.55, edgecolor=COLOR_AUTHOR,
              linewidth=1.0, label='Authors (S>0 only)'),
        Patch(facecolor=COLOR_CONTROL, alpha=0.55, edgecolor=COLOR_CONTROL,
              linewidth=1.0, label='Controls (S>0 only)'),
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.01, 0.5),
              fontsize=FONT["size_legend"] * scale, frameon=True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    add_grid(ax, axis='y')

    plt.tight_layout()

    save_figure(fig, str(output_path).removesuffix('.png'))
    plt.close()

    return fig


def generate_known_groups_plots(matched_df: pd.DataFrame):
    """Generate all known-groups figures for one matched set."""
    if len(matched_df) == 0:
        print("  No matched pairs - skipping known-groups plots")
        return
    plot_known_groups_summary(matched_df)
    plot_s_index_comparison(matched_df)
    plot_domain_comparisons(matched_df)


# =============================================================================
# Main Entry Point
# =============================================================================

def run_known_groups_validation(df: pd.DataFrame, seed: int = RANDOM_SEED,
                                 n_stability_replicates: int = DEFAULT_STABILITY_REPLICATES,
                                 write_outputs: bool = True,
                                 generate_plots: bool = True) -> dict:
    """
    Run the complete known-groups validation analysis.

    Performs:
    - Exact 1:1 matching on primary domain + h-index
    - Overall matched comparison (data paper authors vs matched controls)
    - Stability analysis across many random control draws
    - Nearest-neighbor sensitivity analysis for authors without exact matches
    - Per-domain matched comparison (4 domains)

    Args:
        df: DataFrame with required columns (s_index, h_index, openalex_id,
            and data journal/domain columns)
        seed: Random seed for the headline matched set (tie-breaking only;
              the pair count and balance are identical for every seed)
        n_stability_replicates: Number of random re-draws for the stability
              analysis (default: DEFAULT_STABILITY_REPLICATES)
        write_outputs: Save the matched-pair and stability CSVs.
        generate_plots: Generate the known-groups figures.

    Returns:
        dict with all results
    """
    print("\n" + "=" * 70)
    print("STARTING KNOWN-GROUPS VALIDATION ANALYSIS")
    print("=" * 70 + "\n")

    # Step 1: Prepare data
    print("Step 1: Deriving variables...")
    df_prepared = prepare_data(df)

    n_authors = df_prepared["is_data_paper_author"].sum()
    n_controls = (~df_prepared["is_data_paper_author"]).sum()
    print(f"  Data paper authors: {n_authors}")
    print(f"  Non-authors (potential controls): {n_controls}")
    print("  Domain distribution:")
    print(df_prepared["primary_domain"].value_counts().to_string(header=False))
    print()

    # Step 2: Matching
    print("Step 2: Exact matching on domain + h-index...")
    matched_df, matching_report = perform_matching(df_prepared, seed=seed)
    print(f"  Matched {matching_report['n_matched']} pairs "
          f"({matching_report['n_excluded_no_controls']} authors without exact match)")
    print()

    # Step 3: Matching diagnostics
    print("Step 3: Checking matching balance...")
    diagnostics = matching_diagnostics(matched_df)
    print()

    # Step 4: Primary analysis (overall)
    print("Step 4: Running overall matched analysis...")
    primary_results = primary_analysis(matched_df)
    print()

    # Step 4b: S>0 restricted analysis
    print("Step 4b: Running S>0 restricted analysis...")
    s_positive_results = s_positive_analysis(matched_df)
    print()

    # Step 4c: Stability across random control draws
    print(f"Step 4c: Running stability analysis ({n_stability_replicates} random draws)...")
    stability_results = matching_stability_analysis(
        df_prepared, n_replicates=n_stability_replicates, base_seed=seed
    )
    print()

    # Step 4d: NN sensitivity analysis (include authors without exact matches)
    print("Step 4d: Running nearest-neighbor sensitivity analysis...")
    nn_matched_df, nn_report = perform_matching_with_nn_sensitivity(df_prepared, seed=seed)
    nn_primary_results = primary_analysis(nn_matched_df)
    print(f"  Added {nn_report['n_nn_matched']} NN pairs, "
          f"{nn_report['n_nn_unmatched']} still unmatched")
    print()

    # Step 5: Per-domain matched analyses
    print("Step 5: Running per-domain matched analyses...")
    domain_matched_results = all_domains_matched_analysis(matched_df)
    print()

    # Step 5b: Domain-level unmatched analyses (supplementary)
    print("Step 5b: Running supplementary domain unmatched analyses...")
    domain_unmatched_results = all_domains_unmatched_analysis(df_prepared)
    print()

    # Step 6: Outputs
    print("Step 6: Generating outputs...")
    print()

    # Print full report
    print_report(matching_report, diagnostics, primary_results,
                 domain_matched_results, domain_unmatched_results,
                 s_positive_results=s_positive_results,
                 stability_results=stability_results,
                 nn_sensitivity_report=nn_report,
                 nn_sensitivity_primary=nn_primary_results)

    if write_outputs:
        # Save matched pairs CSV
        if len(matched_df) > 0:
            save_matched_pairs(matched_df, f"{KNOWN_GROUPS_DATA_DIR}/matched_pairs.csv")

        # Save NN sensitivity pairs (exact + NN combined)
        if len(nn_matched_df) > len(matched_df):
            save_matched_pairs(
                nn_matched_df,
                f"{KNOWN_GROUPS_DATA_DIR}/matched_pairs_nn_sensitivity.csv",
            )

        # Save long-form overall and per-domain replicate results.
        if "error" not in stability_results:
            stability_path = Path(f"{KNOWN_GROUPS_DATA_DIR}/stability_replicates.csv")
            stability_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(stability_results["replicate_records"]).to_csv(
                stability_path, index=False
            )
            print(f"Stability replicates saved to: {stability_path}")

    # Step 7: Figures
    if generate_plots:
        print("\nStep 7: Creating figures...")
        generate_known_groups_plots(matched_df)

    # Compile and return results
    results = compile_results_dict(matching_report, diagnostics, primary_results,
                                   domain_matched_results, domain_unmatched_results)

    # Add S>0, stability, and NN sensitivity results
    results["s_positive"] = s_positive_results
    results["stability"] = {k: v for k, v in stability_results.items()
                            if not k.startswith("replicate_")}
    results["nn_sensitivity"] = {
        "report": {k: v for k, v in nn_report.items() if k != "excluded_authors"},
        "primary": nn_primary_results,
    }

    print("\n" + "=" * 70)
    print("KNOWN-GROUPS VALIDATION COMPLETE")
    print("=" * 70 + "\n")

    return results
