"""
Core validation analyses for the S-Index.

1. Correlation with traditional metrics
2. Conditional entropy H(S | metric): information in S beyond the metric
3. Known groups: data paper authors vs matched controls
4. Hidden stewards: high S-index but low traditional metric

Also assembles the plain-text validation report.
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

import config
from config import ComparisonMetric, H_INDEX_METRIC, Domain, filter_by_domain
from known_groups_validation import (
    prepare_data as prepare_known_groups_data,
    perform_matching,
    perform_matching_with_nn_sensitivity,
    matching_stability_analysis,
    matching_diagnostics,
    primary_analysis,
    all_domains_s_positive_analysis,
    all_domains_matched_analysis,
    DEFAULT_STABILITY_REPLICATES,
    NN_CALIPER_MIN,
    NN_CALIPER_FRAC,
)


# =============================================================================
# Correlation with Traditional Metrics
# =============================================================================

def compute_correlation(df, col1, col2):
    """
    Compute Pearson and Spearman correlations between two columns.
    Returns dict with r, p-values, and sample size.
    """
    valid = df[[col1, col2]].dropna()
    n = len(valid)

    if n < 3:
        raise ValueError(f"Need at least 3 samples for correlation, got {n}")

    x, y = valid[col1].values, valid[col2].values

    # Check for constant values (would cause division by zero)
    if np.std(x) == 0 or np.std(y) == 0:
        return {
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "n": n,
            "warning": "Constant values detected - correlation undefined"
        }

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_rho, spearman_p = stats.spearmanr(x, y)

    result = {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "n": n
    }

    # Warn if sample size is small (correlations unreliable)
    if n < 10:
        result["warning"] = f"Small sample size (n={n}) - interpret with caution"

    return result


def correlations_with_traditional_metrics(df):
    """
    Correlate S-index with h-index, citation count, and number of datasets.
    Returns dict of correlation results for each metric present in df.
    """
    return {
        col: compute_correlation(df, "s_index", col)
        for col in ("h_index", "cited_by_count", "num_datasets")
        if col in df.columns
    }


def compute_conditional_entropy(
    df: pd.DataFrame,
    n_bins: int = 10,
    s_positive_only: bool = False,
    metric: Optional[ComparisonMetric] = None,
    domain: Optional[Domain] = None
) -> dict:
    """
    Compute conditional entropy H(S|metric) to quantify new information.

    Higher values indicate S-index provides more information beyond the comparison metric.
    Uses quantile-based discretization (equal-frequency bins) for robust entropy
    estimation with skewed distributions like citation metrics.

    Args:
        df: DataFrame with s_index and comparison metric columns
        n_bins: Number of bins for discretization
        s_positive_only: If True, only include researchers with S > 0.
                        This provides more meaningful entropy among those
                        who have any data sharing impact.
        metric: ComparisonMetric to compare against (default: h-index)
        domain: Domain to filter researchers by (default: all domains)

    Returns dict with:
    - H_S_given_metric: conditional entropy (new information in S given metric)
    - H_S: marginal entropy of S (total information in S)
    - ratio: H(S|metric)/H(S) - fraction of S information NOT explained by metric

    Formula: H(S|M) = H(S,M) - H(M)  (joint entropy minus marginal entropy of M)
    """
    if metric is None:
        metric = H_INDEX_METRIC

    # Apply domain filter first
    df_filtered = filter_by_domain(df, domain)

    metric_col = metric.column

    if metric_col not in df_filtered.columns:
        raise KeyError(f"Column '{metric_col}' not found in dataframe")

    valid = df_filtered[["s_index", metric_col]].dropna()

    # Optionally filter to S > 0 for more meaningful entropy
    if s_positive_only:
        valid = valid[valid["s_index"] > 0]

    n_total = len(df_filtered[["s_index", metric_col]].dropna())
    n_s_positive = len(df_filtered[(df_filtered["s_index"] > 0) & df_filtered["s_index"].notna()])

    if len(valid) < 4:
        raise ValueError(f"Need at least 4 valid samples for entropy calculation, got {len(valid)}")

    if valid["s_index"].std() == 0 or valid[metric_col].std() == 0:
        raise ValueError(f"Cannot compute entropy: zero variance in s_index or {metric_col}")

    if len(valid) < n_bins:
        n_bins = max(2, len(valid) // 2)

    # Quantile (equal-frequency) bins suit skewed metrics better than
    # equal-width bins; duplicates='drop' merges bins that share an edge.
    try:
        s_binned = pd.qcut(valid["s_index"], q=n_bins, labels=False, duplicates='drop')
        metric_binned = pd.qcut(valid[metric_col], q=n_bins, labels=False, duplicates='drop')
    except ValueError:
        # Fallback if too many duplicates: use fewer bins
        n_bins = max(2, n_bins // 2)
        s_binned = pd.qcut(valid["s_index"], q=n_bins, labels=False, duplicates='drop')
        metric_binned = pd.qcut(valid[metric_col], q=n_bins, labels=False, duplicates='drop')

    # Joint distribution P(S,M)
    joint = pd.crosstab(s_binned, metric_binned, normalize=True)

    # Marginal distributions
    p_metric = joint.sum(axis=0)
    p_s = joint.sum(axis=1)

    # H(S,M) = joint entropy
    joint_probs = joint.values.flatten()
    joint_probs = joint_probs[joint_probs > 0]  # Remove zeros to avoid log(0)
    joint_entropy = stats.entropy(joint_probs, base=2)

    # H(M) = marginal entropy of comparison metric
    p_metric_vals = p_metric.values[p_metric.values > 0]
    metric_entropy = stats.entropy(p_metric_vals, base=2)

    # H(S) = marginal entropy of S
    p_s_vals = p_s.values[p_s.values > 0]
    s_entropy = stats.entropy(p_s_vals, base=2)

    # H(S|M) = H(S,M) - H(M)
    conditional_entropy = joint_entropy - metric_entropy

    # Clamp to zero: discretization artifacts can produce slightly negative
    # values, which are information-theoretically impossible
    conditional_entropy = max(0.0, conditional_entropy)

    # Ratio: what fraction of S information is NOT explained by metric?
    # If ratio = 1.0, metric tells us nothing about S (independent)
    # If ratio = 0.0, metric perfectly predicts S
    ratio = conditional_entropy / s_entropy if s_entropy > 0 else np.nan

    # Clamp ratio to [0, 1] for same reason (discretization artifacts)
    if not np.isnan(ratio):
        ratio = max(0.0, min(1.0, ratio))

    # Report actual number of unique bins (may be fewer than requested
    # due to duplicates='drop' in pd.qcut)
    actual_s_bins = int(s_binned.nunique())
    actual_metric_bins = int(metric_binned.nunique())

    return {
        "metric": metric.name,
        "domain": domain.name if domain else "All",
        "H_S_given_metric": conditional_entropy,
        "H_S": s_entropy,
        "ratio": ratio,
        "n_bins_used": min(actual_s_bins, actual_metric_bins),
        "n": len(valid),
        "n_total": n_total,
        "n_s_positive": n_s_positive,
        "s_positive_only": s_positive_only,
    }


# =============================================================================
# Known Groups Validation (Matched)
# =============================================================================

def apply_fdr_correction(results: dict) -> None:
    """
    Apply Benjamini-Hochberg FDR correction to the primary Wilcoxon family,
    adding "fdr_adjusted_p" to each tested entry of results["domain_results"].

    The family contains the overall test and each per-domain exact-match test
    that could be computed. The S>0 subset is descriptive and adds no tests.
    """
    tested = [
        r for r in results.get("domain_results", [])
        if "error" not in r
        and r.get("wilcoxon_p_value") is not None
        and not np.isnan(r["wilcoxon_p_value"])
    ]
    if not tested:
        return

    raw_pvals = np.array([r["wilcoxon_p_value"] for r in tested], dtype=float)
    _, fdr_pvals, _, _ = multipletests(raw_pvals, alpha=config.ALPHA, method='fdr_bh')
    for r, adj_p in zip(tested, fdr_pvals):
        r["fdr_adjusted_p"] = float(adj_p)


def run_known_groups_summary(df, n_stability_replicates: int = DEFAULT_STABILITY_REPLICATES):
    """
    Run known-groups analysis for validation report summary.
    Returns a dict with:
      - 'domain_results': list of result dicts for overall + each domain
      - 's_positive': list of S>0 descriptive results for overall + each domain
      - 'matching_report': exact-matching summary (incl. excluded authors)
      - 'diagnostics': balance diagnostics (exact-match %, citation SMD)
      - 'stability': results across random control draws
      - 'nn_sensitivity': NN sensitivity analysis for excluded authors
    """
    df_prepared = prepare_known_groups_data(df)

    # Exact matching on domain + h-index
    matched_df, matching_report = perform_matching(df_prepared)

    if len(matched_df) == 0:
        return {
            "domain_results": [{"analysis": "Overall", "error": "No matched pairs found"}],
            "s_positive": [],
            "matching_report": matching_report,
            "diagnostics": {"error": "No matched pairs"},
            "stability": {"error": "No matched pairs"},
            "nn_sensitivity": {"error": "No matched pairs"},
        }

    # Overall first, then one entry per domain
    overall = primary_analysis(matched_df)
    overall["analysis"] = "Overall"
    domain_results = [overall]
    for dr in all_domains_matched_analysis(matched_df):
        dr["analysis"] = dr.get("domain", "Unknown")
        domain_results.append(dr)

    # S>0 descriptive view, overall and per domain
    s_pos_results = all_domains_s_positive_analysis(matched_df)

    # Balance diagnostics (exact-match invariant + citation SMD)
    diagnostics = matching_diagnostics(matched_df)

    # Stability across random control draws
    stability = matching_stability_analysis(
        df_prepared, n_replicates=n_stability_replicates
    )
    stability = {k: v for k, v in stability.items() if not k.startswith("replicate_")}

    # NN sensitivity: include authors without exact matches
    nn_matched_df, nn_report = perform_matching_with_nn_sensitivity(df_prepared)
    nn_results = {
        "n_nn_matched": nn_report.get("n_nn_matched", 0),
        "n_nn_unmatched": nn_report.get("n_nn_unmatched", 0),
        "nn_pair_h_deltas": nn_report.get("nn_pair_h_deltas", []),
        "primary": primary_analysis(nn_matched_df),
    }

    all_results = {
        "domain_results": domain_results,
        "s_positive": s_pos_results,
        "matching_report": {k: v for k, v in matching_report.items()
                            if k != "excluded_authors"},
        "excluded_authors": matching_report.get("excluded_authors", []),
        "diagnostics": diagnostics,
        "stability": stability,
        "nn_sensitivity": nn_results,
    }

    # Apply Benjamini-Hochberg FDR correction across the primary test family.
    apply_fdr_correction(all_results)

    return all_results


# =============================================================================
# Hidden Stewards
# =============================================================================

def find_hidden_stewards(
    df,
    metric: Optional[ComparisonMetric] = None,
    domain: Optional[Domain] = None
):
    """
    Find hidden stewards: researchers with a high S-index but a low
    comparison metric.

    A hidden steward has:
    - S-index >= the HIDDEN_STEWARD_S_PERCENTILE-th percentile of researchers
      with S > 0 ("high" among researchers whose data has been reused), and
    - metric <= the HIDDEN_STEWARD_H_PERCENTILE-th percentile of all
      researchers, zeros included ("low" in the full population).

    Args:
        df: DataFrame with s_index and comparison metric columns
        metric: ComparisonMetric to compare against (default: h-index)
        domain: Domain to filter researchers by (default: all domains);
                thresholds are computed within the domain

    Returns:
        Tuple of (stewards_df, thresholds_dict), stewards sorted by S-index.
    """
    if metric is None:
        metric = H_INDEX_METRIC

    df_filtered = filter_by_domain(df, domain)
    metric_col = metric.column

    required_cols = ["s_index", metric_col]
    optional_cols = [c for c in ("name", "openalex_id") if c in df_filtered.columns]
    valid = df_filtered[required_cols + optional_cols].dropna(subset=required_cols).copy()
    s_positive = valid[valid["s_index"] > 0]

    if len(s_positive) > 0:
        s_threshold = np.percentile(s_positive["s_index"], config.HIDDEN_STEWARD_S_PERCENTILE)
    else:
        s_threshold = 0

    if len(valid) > 0:
        metric_threshold = np.percentile(valid[metric_col], config.HIDDEN_STEWARD_H_PERCENTILE)
    else:
        metric_threshold = 0

    stewards = valid[(valid["s_index"] >= s_threshold) & (valid[metric_col] <= metric_threshold)]

    thresholds = {
        "metric": metric.name,
        "domain": domain.name if domain else "All",
        "n_total": len(valid),
        "n_s_positive": len(s_positive),
        "s_percentile": config.HIDDEN_STEWARD_S_PERCENTILE,
        "s_threshold": s_threshold,
        "metric_percentile": config.HIDDEN_STEWARD_H_PERCENTILE,
        "metric_threshold": metric_threshold,
    }
    return stewards.sort_values("s_index", ascending=False), thresholds


# =============================================================================
# Summary Report - Section Helpers
# =============================================================================

def _report_pipeline_stats(pipeline_stats: dict) -> list:
    """Generate pipeline statistics section."""
    lines = []
    lines.append("-" * 40)
    lines.append("DATA PIPELINE SUMMARY")
    lines.append("-" * 40)
    if "n_raw" in pipeline_stats:
        lines.append(f"Raw data records: {pipeline_stats['n_raw']:,}")
    if "n_filtered" in pipeline_stats:
        lines.append(f"After filtering: {pipeline_stats['n_filtered']:,}")
    if "ess" in pipeline_stats:
        ess = pipeline_stats['ess']
        n_filt = pipeline_stats.get('n_filtered', ess)
        pct = (ess / n_filt * 100) if n_filt > 0 else 0
        lines.append(f"Effective Sample Size (ESS): {ess:,} ({pct:.1f}% of filtered)")
    if "n_resampled" in pipeline_stats:
        lines.append(f"After resampling: {pipeline_stats['n_resampled']:,}")
    if "n_cleaned" in pipeline_stats:
        lines.append(f"After cleaning: {pipeline_stats['n_cleaned']:,}")
    lines.append("")
    return lines


def _report_sample_characteristics(df) -> tuple:
    """Generate sample characteristics section. Returns (lines, pct_s_positive)."""
    lines = []
    n_total = len(df)
    # Use the intersection of valid S-index AND h-index as the base population
    valid_both = df[["s_index", "h_index"]].dropna()
    n_valid = len(valid_both)
    # Compute S and h counts from the SAME valid population
    n_s_positive = int((valid_both["s_index"] > 0).sum())
    n_s_zero = n_valid - n_s_positive
    pct_s_positive = n_s_positive / n_valid * 100 if n_valid > 0 else 0

    n_h_zero = int((valid_both["h_index"] == 0).sum()) if "h_index" in df.columns else 0

    lines.append("-" * 40)
    lines.append("SAMPLE CHARACTERISTICS")
    lines.append("-" * 40)
    lines.append(f"Total researchers in analysis: {n_total:,}")
    lines.append(f"With valid S-index and h-index: {n_valid:,}")
    lines.append("")
    lines.append("S-index distribution:")
    lines.append(f"  S > 0 (active data sharers): {n_s_positive:,} ({pct_s_positive:.1f}%)")
    lines.append(f"  S = 0 (no data sharing impact): {n_s_zero:,} ({100-pct_s_positive:.1f}%)")

    if n_s_positive > 0:
        s_positive_values = valid_both[valid_both["s_index"] > 0]["s_index"]
        lines.append(f"  Among S > 0: median={s_positive_values.median():.1f}, mean={s_positive_values.mean():.1f}")

    if "h_index" in df.columns and n_valid > 0:
        lines.append("")
        lines.append("h-index distribution:")
        lines.append(f"  h = 0: {n_h_zero:,} ({n_h_zero/n_valid*100:.1f}%)")
        lines.append(f"  h > 0: {n_valid - n_h_zero:,} ({(n_valid-n_h_zero)/n_valid*100:.1f}%)")
        h_values = valid_both["h_index"]
        lines.append(f"  All researchers: median={h_values.median():.0f}, mean={h_values.mean():.1f}")

    if "cited_by_count" in df.columns:
        valid_cit = df[["s_index", "h_index", "cited_by_count"]].dropna()
        n_cit_valid = len(valid_cit)
        n_cit_zero = int((valid_cit["cited_by_count"] == 0).sum())
        lines.append("")
        lines.append("Citation count distribution:")
        lines.append(f"  citations = 0: {n_cit_zero:,} ({n_cit_zero/n_cit_valid*100:.1f}%)" if n_cit_valid > 0 else f"  citations = 0: {n_cit_zero:,}")
        lines.append(f"  citations > 0: {n_cit_valid - n_cit_zero:,} ({(n_cit_valid-n_cit_zero)/n_cit_valid*100:.1f}%)" if n_cit_valid > 0 else f"  citations > 0: {n_cit_valid - n_cit_zero:,}")
        cit_values = valid_cit["cited_by_count"]
        lines.append(f"  All researchers: median={cit_values.median():.0f}, mean={cit_values.mean():.1f}")

    lines.append("")
    return lines, pct_s_positive


def _report_correlations(df) -> list:
    """Generate correlation section of report."""
    lines = []
    lines.append("-" * 40)
    lines.append("1. CORRELATION WITH TRADITIONAL METRICS")
    lines.append("-" * 40)

    # Overall correlations
    lines.append("\n=== ALL DOMAINS ===")
    corr_results = correlations_with_traditional_metrics(df)
    for metric_name, r in corr_results.items():
        lines.append(f"\nS-index vs {metric_name}:")
        if np.isnan(r['pearson_r']):
            lines.append(f"  {r.get('warning', 'Correlation undefined')}")
        else:
            lines.append(f"  Pearson r = {r['pearson_r']:.3f} (p = {r['pearson_p']:.4f})")
            lines.append(f"  Spearman rho = {r['spearman_rho']:.3f} (p = {r['spearman_p']:.4f})")
        lines.append(f"  N = {r['n']}")
        if "warning" in r and not np.isnan(r['pearson_r']):
            lines.append(f"  WARNING: {r['warning']}")

    # Per-domain correlations
    lines.append("\n=== BY DOMAIN ===")
    for domain in config.ALL_DOMAINS:
        df_domain = filter_by_domain(df, domain)
        n_domain = len(df_domain)
        n_s_pos = len(df_domain[df_domain["s_index"] > 0])
        lines.append(f"\n{domain.name} (N={n_domain:,}, S>0={n_s_pos:,}):")

        corr_domain = correlations_with_traditional_metrics(df_domain)
        for metric_name, r in corr_domain.items():
            if np.isnan(r['pearson_r']):
                lines.append(f"  vs {metric_name}: N/A")
            else:
                lines.append(f"  vs {metric_name}: r={r['pearson_r']:.3f}, rho={r['spearman_rho']:.3f}")

    lines.append("")
    return lines


def _report_conditional_entropy(df, pct_s_positive: float) -> list:
    """Generate conditional entropy section of report."""
    lines = []
    lines.append("-" * 40)
    lines.append("2. CONDITIONAL ENTROPY")
    lines.append("-" * 40)
    lines.append("\nH(S|metric)/H(S) ratio: fraction of S information NOT explained by metric.")
    lines.append("Higher ratio = S-index provides more new information beyond the metric.")

    # Overall results
    lines.append("\n=== ALL DOMAINS ===")
    for metric in config.COMPARISON_METRICS:
        if metric.column not in df.columns:
            lines.append(f"\n{metric.name}: Column '{metric.column}' not found, skipping.")
            continue

        lines.append(f"\n--- S-index vs {metric.name} ---")

        try:
            entropy_all = compute_conditional_entropy(df, s_positive_only=False, metric=metric)
            lines.append(f"\nAll researchers (n={entropy_all['n']}):")
            lines.append(f"  H(S|{metric.short_name}) = {entropy_all['H_S_given_metric']:.3f} bits")
            lines.append(f"  H(S) = {entropy_all['H_S']:.3f} bits")
            lines.append(f"  Ratio H(S|{metric.short_name})/H(S) = {entropy_all['ratio']:.1%}")
            lines.append(f"  NOTE: Low H(S) reflects that {100-pct_s_positive:.0f}% have S=0")
        except (ValueError, KeyError) as e:
            lines.append(f"\nAll researchers: {e}")

        try:
            entropy_pos = compute_conditional_entropy(df, s_positive_only=True, metric=metric)
            lines.append(f"\nResearchers with S > 0 only (n={entropy_pos['n']}):")
            lines.append(f"  H(S|{metric.short_name}) = {entropy_pos['H_S_given_metric']:.3f} bits")
            lines.append(f"  H(S) = {entropy_pos['H_S']:.3f} bits")
            lines.append(f"  Ratio H(S|{metric.short_name})/H(S) = {entropy_pos['ratio']:.1%}")
            lines.append("  (This measures information content among active data sharers)")
        except (ValueError, KeyError) as e:
            lines.append(f"\nResearchers with S > 0 only: {e}")

    # Per-domain summary table
    lines.append("\n=== BY DOMAIN (S > 0 only) ===")
    lines.append("\nDomain                   | vs h-index | vs citations")
    lines.append("-" * 55)

    for domain in config.ALL_DOMAINS:
        row = f"{domain.name:<24} |"
        for metric in config.COMPARISON_METRICS:
            if metric.column not in df.columns:
                row += "    N/A    |"
                continue
            try:
                entropy = compute_conditional_entropy(
                    df, s_positive_only=True, metric=metric, domain=domain
                )
                row += f"   {entropy['ratio']:.1%}   |"
            except Exception:
                row += "    N/A    |"
        lines.append(row)

    lines.append("")
    return lines


def significance_stars(p_val: float) -> str:
    """Conventional significance stars for a p-value."""
    return "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""


def _format_p_value(p_val, n_pairs=None, min_n=config.MIN_PAIRS_FOR_DOMAIN_TEST) -> str:
    """Format a p-value with significance stars, or say why it is absent."""
    if n_pairs is not None and n_pairs < min_n:
        return f"Not computed (n={n_pairs} < {min_n})"
    if p_val is None or np.isnan(p_val):
        return "N/A"
    return f"{p_val:.4f} {significance_stars(p_val)}".strip()


def _format_p_with_fdr(result_dict: dict, n_pairs=None) -> str:
    """Format the raw Wilcoxon p-value together with its FDR-adjusted value.

    Significance stars are based on the FDR-adjusted value when available.
    """
    raw_p = result_dict.get("wilcoxon_p_value")
    fdr_p = result_dict.get("fdr_adjusted_p")
    if raw_p is None or np.isnan(raw_p) or fdr_p is None or np.isnan(fdr_p):
        return _format_p_value(raw_p, n_pairs=n_pairs)
    return f"{raw_p:.4f} (FDR: {fdr_p:.4f}) {significance_stars(fdr_p)}".strip()


def _report_known_groups(df) -> list:
    """Generate known groups validation section of report."""
    lines = []
    lines.append("-" * 40)
    lines.append("3. KNOWN GROUPS VALIDATION (MATCHED)")
    lines.append("-" * 40)
    lines.append("\nComparing data paper authors vs matched controls.")
    lines.append("Matching: exact 1:1 on (domain, h-index), without replacement.")
    lines.append("Every pair is identical on both matching variables by construction.")
    lines.append("Test: Wilcoxon signed-rank (paired, one-sided: authors > controls)")
    lines.append("Overall and domain Wilcoxon p-values use Benjamini-Hochberg FDR correction.")

    data_journal_cols = ["scientific_data_pubs", "essd_pubs", "gigascience_pubs", "bds_pubs"]
    domain_cols = ["health_sciences", "life_sciences", "physical_sciences", "social_sciences"]
    has_data_paper_cols = any(c in df.columns for c in data_journal_cols)
    has_domain_cols = any(c in df.columns for c in domain_cols)

    if not has_data_paper_cols:
        lines.append("\nSkipped: No data paper columns found in data.")
        return lines
    if not has_domain_cols:
        lines.append("\nSkipped: No domain columns found in data.")
        return lines
    if "openalex_id" not in df.columns:
        lines.append("\nSkipped: openalex_id column required for matching.")
        return lines

    try:
        kg_all = run_known_groups_summary(df)
        kg_results = kg_all.get("domain_results", [])
        s_pos = kg_all.get("s_positive", {})
        kg_report = kg_all.get("matching_report", {})
        kg_excluded = kg_all.get("excluded_authors", [])
        kg_diag = kg_all.get("diagnostics", {})
        kg_stability = kg_all.get("stability", {})
        kg_nn = kg_all.get("nn_sensitivity", {})

        # Main results
        for r in kg_results:
            name = r["analysis"]
            lines.append(f"\n{name.upper()}:")
            if "error" in r:
                lines.append(f"  {r['error']}")
                continue
            n_pairs = r['n_pairs']
            lines.append(f"  Matched pairs: {n_pairs}")
            lines.append(f"  Median S - Authors: {r['median_s_authors']:.1f}, Controls: {r['median_s_controls']:.1f}")
            s_pos_auth = r.get('prop_s_positive_authors', 0)
            s_pos_ctrl = r.get('prop_s_positive_controls', 0)
            if s_pos_auth > 0 or s_pos_ctrl > 0:
                lines.append(f"  S > 0 - Authors: {s_pos_auth:.0%}, Controls: {s_pos_ctrl:.0%}")
            lines.append(f"  Author wins: {r['prop_pairs_author_wins']:.0%}, Ties: {r['prop_pairs_ties']:.0%}, Control wins: {r['prop_pairs_control_wins']:.0%}")
            lines.append(f"  Wilcoxon p-value: {_format_p_with_fdr(r, n_pairs)}")
            lines.append(
                f"  Exact sign test (wins vs losses): p={r.get('sign_test_p_value', float('nan')):.2e}"
            )
            lines.append(
                "  Exact paired S>0 test (author+/control0 vs author0/control+): "
                f"{r.get('author_positive_control_zero', 0)} vs "
                f"{r.get('author_zero_control_positive', 0)}, "
                f"p={r.get('positive_mcnemar_p_value', float('nan')):.2e}"
            )

        # Interpretation note
        overall = kg_results[0] if kg_results else {}
        if overall.get('median_s_authors', 0) == 0 and overall.get('median_s_controls', 0) == 0:
            lines.append("\nNOTE: Both medians = 0 indicates sparse S-index distribution.")
            lines.append("The Wilcoxon test ranks nonzero paired S-index differences; the")
            lines.append("separate exact paired S>0 test evaluates positive prevalence.")

        # S>0 restricted analysis
        lines.append("\n" + "-" * 40)
        lines.append("3b. S>0 DESCRIPTIVE VIEW (exact matching)")
        lines.append("-" * 40)
        lines.append("Analyzing only pairs where at least one member has S > 0.")

        for sp in s_pos:
            domain_name = sp.get('domain', 'Overall')
            lines.append(f"\n{domain_name.upper()}:")
            if "error" in sp:
                lines.append(f"  {sp['error']}")
                continue
            n_s_pos = sp.get('n_pairs_s_positive', 0)
            n_total = sp.get('n_pairs_total', 0)
            pct = sp.get('pct_pairs_s_positive', 0)
            lines.append(f"  Pairs with S>0: {n_s_pos} of {n_total} ({pct:.1f}%)")
            lines.append(f"  Median S - Authors: {sp.get('median_s_authors', 0):.1f}, Controls: {sp.get('median_s_controls', 0):.1f}")
            lines.append(f"  Author wins: {sp.get('prop_pairs_author_wins', 0):.0%}, Ties: {sp.get('prop_pairs_ties', 0):.0%}, Control wins: {sp.get('prop_pairs_control_wins', 0):.0%}")
            lines.append(
                "  Paired S>0 prevalence p-value (full matched set): "
                f"{sp.get('positive_mcnemar_p_value', float('nan')):.2e}"
            )
            if "warning" in sp:
                lines.append(f"  Warning: {sp['warning']}")
        lines.append("  No second Wilcoxon test is run: removing 0-vs-0 pairs does not change its ranked differences.")

        # Matching quality, stability, and sensitivity
        lines.append("\n" + "-" * 40)
        lines.append("3c. MATCHING QUALITY, STABILITY, AND SENSITIVITY")
        lines.append("-" * 40)

        if kg_report:
            lines.append(f"\nAuthors matched exactly: {kg_report.get('n_matched', 0)} of "
                         f"{kg_report.get('n_data_paper_authors_total', 0)}")
            if kg_excluded:
                excl_desc = ", ".join(
                    f"{e['domain']} h={e['h_index']:.0f}"
                    for e in sorted(kg_excluded, key=lambda e: (e['domain'], e['h_index']))
                )
                lines.append(f"Excluded (no exact-match control): {len(kg_excluded)} ({excl_desc})")

        if kg_diag and "error" not in kg_diag:
            lines.append(f"Pairs identical on h-index: {kg_diag.get('pct_exact_h', 0):.1f}%")
            if "citation_smd" in kg_diag:
                lines.append(f"Citation balance (not matched on): SMD = {kg_diag['citation_smd']:.3f} "
                             f"(|SMD| < 0.1 indicates balance)")

        if kg_stability and "error" not in kg_stability:
            lines.append(f"\nStability across {kg_stability.get('n_replicates', 0)} random control draws:")
            lines.append(f"  Wilcoxon p: median {kg_stability.get('p_median', float('nan')):.2e}, "
                         f"range [{kg_stability.get('p_min', float('nan')):.2e}, "
                         f"{kg_stability.get('p_max', float('nan')):.2e}]")
            lines.append(f"  Replicates with p < 0.001: {kg_stability.get('prop_p_below_0_001', 0):.0%}")
            lines.append(f"  Author win rate: {kg_stability.get('win_rate_min', 0):.1%} - "
                         f"{kg_stability.get('win_rate_max', 0):.1%}")
            lines.append("  Per-domain draw stability:")
            for domain, summary in kg_stability.get("domain_summaries", {}).items():
                if np.isnan(summary.get("p_median", np.nan)):
                    p_text = "not tested (insufficient pairs)"
                else:
                    p_text = (
                        f"median p={summary['p_median']:.2e}; "
                        f"p<0.05 in {summary['prop_p_below_0_05']:.0%} of draws"
                    )
                lines.append(
                    f"    {domain}: wins {summary['win_rate_min']:.1%}-"
                    f"{summary['win_rate_max']:.1%}; {p_text}"
                )

        if kg_nn and "error" not in kg_nn:
            lines.append("\nSensitivity: nearest-neighbor matching for excluded authors")
            lines.append(f"  (caliper |dh| <= max({NN_CALIPER_MIN}, {NN_CALIPER_FRAC:.0%} of author h))")
            lines.append(f"  Added NN pairs: {kg_nn.get('n_nn_matched', 0)} "
                         f"(|dh|: {kg_nn.get('nn_pair_h_deltas', [])}), "
                         f"still unmatched: {kg_nn.get('n_nn_unmatched', 0)}")
            nn_primary = kg_nn.get('primary', {})
            if nn_primary and "error" not in nn_primary:
                p_val = nn_primary.get('wilcoxon_p_value')
                p_str = f"{p_val:.2e}" if p_val is not None and not np.isnan(p_val) else "N/A"
                lines.append(f"  With NN pairs included: n={nn_primary.get('n_pairs', 0)}, "
                             f"Author wins={nn_primary.get('prop_pairs_author_wins', 0):.0%}, p={p_str}")

    except Exception as e:
        lines.append(f"\nError running matched validation: {e}")

    lines.append("")
    return lines


def _report_hidden_stewards(df) -> list:
    """Generate hidden stewards section of report."""
    lines = []
    lines.append("-" * 40)
    lines.append("4. HIDDEN STEWARDS")
    lines.append("-" * 40)
    lines.append("\nHidden stewards: researchers with HIGH S-index but LOW traditional metrics.")
    lines.append("")
    lines.append("Threshold computation methodology:")
    lines.append("  - S-index threshold: computed on S > 0 researchers only")
    lines.append("    (What counts as 'high' among active data sharers)")
    lines.append("  - Traditional metric threshold: computed on ALL researchers")
    lines.append("    (What counts as 'low' in the full population, including zeros)")

    for metric in config.COMPARISON_METRICS:
        if metric.column not in df.columns:
            lines.append(f"\n{metric.name}: Column '{metric.column}' not found, skipping.")
            continue

        lines.append(f"\n--- S-index vs {metric.name} ---")

        try:
            stewards_pct, thresholds_pct = find_hidden_stewards(df, metric=metric)
            n_s_pos = thresholds_pct['n_s_positive']
            lines.append("\nPercentile method:")
            lines.append(f"  S threshold ({thresholds_pct['s_percentile']}th %ile of S>0, n={n_s_pos}): >= {thresholds_pct['s_threshold']:.1f}")
            lines.append(f"  {metric.name} threshold ({thresholds_pct['metric_percentile']}th %ile of all, n={thresholds_pct['n_total']}): <= {thresholds_pct['metric_threshold']:.1f}")
            lines.append(f"  Hidden stewards identified: {len(stewards_pct)}")
            if len(stewards_pct) > 0 and "name" in stewards_pct.columns:
                lines.append("  Top examples:")
                for _, row in stewards_pct.head(5).iterrows():
                    lines.append(f"    - {row['name']}: S={row['s_index']:.0f}, {metric.short_name}={row[metric.column]:.0f}")
        except Exception as e:
            lines.append(f"\nPercentile method: Error - {e}")

    # Domain breakdown summary
    lines.append("\n=== HIDDEN STEWARDS BY DOMAIN ===")
    lines.append("\nDomain                   | vs h-index | vs citations")
    lines.append("-" * 55)

    for domain in config.ALL_DOMAINS:
        row = f"{domain.name:<24} |"
        for metric in config.COMPARISON_METRICS:
            if metric.column not in df.columns:
                row += "    N/A    |"
                continue
            try:
                stewards, _ = find_hidden_stewards(df, metric=metric, domain=domain)
                df_domain = filter_by_domain(df, domain)
                n_total = len(df_domain)
                pct = 100 * len(stewards) / n_total if n_total > 0 else 0
                row += f"  {len(stewards):>3} ({pct:>4.1f}%)|"
            except Exception:
                row += "    N/A    |"
        lines.append(row)

    lines.append("")
    return lines


# =============================================================================
# Summary Report
# =============================================================================

def generate_validation_report(df, pipeline_stats: Optional[dict] = None):
    """
    Generate a text summary of all validation analyses.
    Compares S-index against both h-index and citation count.

    Args:
        df: Cleaned DataFrame with researcher data
        pipeline_stats: Optional dict with pipeline statistics:
            - n_raw: Number of raw data records
            - n_filtered: Number after filtering (before resampling)
            - n_resampled: Number after resampling (ESS)
            - n_cleaned: Number after cleaning/deduplication
            - ess: Effective sample size
    """
    lines = []
    lines.append("=" * 60)
    lines.append("S-INDEX VALIDATION REPORT")
    lines.append("=" * 60)
    lines.append("")

    # Pipeline statistics (if provided)
    if pipeline_stats:
        lines.extend(_report_pipeline_stats(pipeline_stats))

    # Sample characteristics
    sample_lines, pct_s_positive = _report_sample_characteristics(df)
    lines.extend(sample_lines)

    # 1. Correlation with traditional metrics
    lines.extend(_report_correlations(df))

    # 2. Conditional entropy
    lines.extend(_report_conditional_entropy(df, pct_s_positive))

    # 3. Known groups validation
    lines.extend(_report_known_groups(df))

    # 4. Hidden stewards
    lines.extend(_report_hidden_stewards(df))

    lines.append("=" * 60)

    return "\n".join(lines)
