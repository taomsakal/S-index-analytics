"""
Summary tables for the S-Index validation results.

Writes four CSV tables to output/tables/:
1. Correlations (S-index vs traditional metrics, by domain)
2. Conditional entropy
3. Known groups (data paper authors vs matched controls)
4. Hidden stewards
"""

import os

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

import config
from config import COMPARISON_METRICS, ALL_DOMAINS, get_all_domain_configs, filter_by_domain
from known_groups_validation import (
    RANDOM_SEED, paired_direction_tests, paired_wilcoxon, perform_matching,
    prepare_data, s_positive_analysis,
)
from validation import (
    compute_correlation, compute_conditional_entropy, find_hidden_stewards,
    significance_stars,
)


TABLES_DIR = config.get_tables_dir()


def ensure_tables_dir():
    """Create tables directory if needed."""
    os.makedirs(TABLES_DIR, exist_ok=True)
    return TABLES_DIR


# =============================================================================
# 1. CORRELATION TABLE
# =============================================================================

def generate_correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate correlation table: S-index vs traditional metrics across domains.

    Rows: Domains (All, Life, Health, Physical, Social)
    Columns: Metric pairs (S vs h-index, S vs citations, etc.)
    """
    rows = []

    domain_configs = get_all_domain_configs()

    for domain, domain_label in domain_configs:
        df_domain = filter_by_domain(df, domain)
        n_total = len(df_domain)
        n_s_positive = len(df_domain[df_domain["s_index"] > 0])

        row = {
            "Domain": domain_label,
            "N_total": n_total,
            "N_S>0": n_s_positive,
            "Pct_S>0": round(100 * n_s_positive / n_total, 1) if n_total > 0 else 0,
        }

        # Correlations with each metric
        for metric in COMPARISON_METRICS:
            if metric.column not in df_domain.columns:
                row[f"r_S_vs_{metric.short_name}"] = np.nan
                row[f"p_S_vs_{metric.short_name}"] = np.nan
                continue

            try:
                corr_result = compute_correlation(df_domain, "s_index", metric.column)
                row[f"r_S_vs_{metric.short_name}"] = round(corr_result["pearson_r"], 3)
                row[f"p_S_vs_{metric.short_name}"] = corr_result["pearson_p"]
                row[f"rho_S_vs_{metric.short_name}"] = round(corr_result["spearman_rho"], 3)
                row[f"p_rho_S_vs_{metric.short_name}"] = corr_result["spearman_p"]

                # Also compute for S > 0 only
                df_s_pos = df_domain[df_domain["s_index"] > 0]
                if len(df_s_pos) >= 3:
                    corr_s_pos = compute_correlation(df_s_pos, "s_index", metric.column)
                    row[f"r_S_vs_{metric.short_name}_S>0"] = round(corr_s_pos["pearson_r"], 3)
                    row[f"rho_S_vs_{metric.short_name}_S>0"] = round(corr_s_pos["spearman_rho"], 3)
                else:
                    row[f"r_S_vs_{metric.short_name}_S>0"] = np.nan
                    row[f"rho_S_vs_{metric.short_name}_S>0"] = np.nan
            except Exception:
                row[f"r_S_vs_{metric.short_name}"] = np.nan
                row[f"p_S_vs_{metric.short_name}"] = np.nan
                row[f"r_S_vs_{metric.short_name}_S>0"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def save_correlation_table(df: pd.DataFrame) -> str:
    """Generate and save correlation table."""
    ensure_tables_dir()
    table = generate_correlation_table(df)
    path = os.path.join(TABLES_DIR, "table_correlations.csv")
    table.to_csv(path, index=False)
    return path


# =============================================================================
# 2. CONDITIONAL ENTROPY TABLE
# =============================================================================

def generate_entropy_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate conditional entropy table.

    Shows H(S|metric) for each domain and metric combination.
    Includes both all researchers and S > 0 only cases.
    """
    rows = []

    domain_configs = get_all_domain_configs()

    for domain, domain_label in domain_configs:
        for metric in COMPARISON_METRICS:
            if metric.column not in df.columns:
                continue

            for s_positive_only in [False, True]:
                try:
                    result = compute_conditional_entropy(
                        df, s_positive_only=s_positive_only,
                        metric=metric, domain=domain
                    )

                    rows.append({
                        "Domain": domain_label,
                        "Comparison_Metric": metric.name,
                        "S_Filter": "S > 0" if s_positive_only else "All",
                        "N": result["n"],
                        "H_S": round(result["H_S"], 3),
                        "H_S_given_metric": round(result["H_S_given_metric"], 3),
                        "Ratio": round(result["ratio"], 3),
                        "Interpretation": _interpret_entropy_ratio(result["ratio"]),
                    })
                except Exception as e:
                    rows.append({
                        "Domain": domain_label,
                        "Comparison_Metric": metric.name,
                        "S_Filter": "S > 0" if s_positive_only else "All",
                        "N": np.nan,
                        "H_S": np.nan,
                        "H_S_given_metric": np.nan,
                        "Ratio": np.nan,
                        "Interpretation": f"Error: {str(e)[:50]}",
                    })

    return pd.DataFrame(rows)


def _interpret_entropy_ratio(ratio: float) -> str:
    """Interpret the entropy ratio value."""
    if np.isnan(ratio):
        return "N/A"
    if ratio >= 0.9:
        return "S nearly independent of metric"
    elif ratio >= 0.7:
        return "S mostly independent"
    elif ratio >= 0.5:
        return "Moderate new information"
    elif ratio >= 0.3:
        return "Some new information"
    else:
        return "Metric explains most of S"


def save_entropy_table(df: pd.DataFrame) -> str:
    """Generate and save entropy table."""
    ensure_tables_dir()
    table = generate_entropy_table(df)
    path = os.path.join(TABLES_DIR, "table_conditional_entropy.csv")
    table.to_csv(path, index=False)
    return path


# =============================================================================
# 3. KNOWN GROUPS VALIDATION TABLE
# =============================================================================

def generate_known_groups_table(df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate the known-groups summary table from one exact-matched set.

    Rows: overall and each domain, for all pairs and for the descriptive
    S > 0 view. Benjamini-Hochberg FDR is applied across the tested
    all-pairs rows.
    """
    try:
        matched_df, _ = perform_matching(prepare_data(df), seed=seed)
    except Exception as e:
        return pd.DataFrame([{"Error": str(e)}])

    rows = []
    if len(matched_df) > 0:
        rows.append(extract_kg_row(matched_df, "All Domains", "exact h-index", "All pairs"))
        s_pos_result = s_positive_analysis(matched_df)
        if "error" not in s_pos_result:
            rows.append(_extract_kg_row_from_result(
                s_pos_result, "All Domains", "exact h-index", "S > 0 pairs"
            ))

        for domain in ALL_DOMAINS:
            domain_matched = matched_df[matched_df["domain"] == domain.name]
            if len(domain_matched) < 3:
                continue
            rows.append(extract_kg_row(domain_matched, domain.name, "exact h-index", "All pairs"))
            s_pos_result = s_positive_analysis(matched_df, domain=domain.name)
            if "error" not in s_pos_result and s_pos_result.get("n_pairs_s_positive", 0) >= 3:
                rows.append(_extract_kg_row_from_result(
                    s_pos_result, domain.name, "exact h-index", "S > 0 pairs"
                ))

    table = pd.DataFrame(rows)
    if not table.empty:
        table = add_fdr_columns(table)
        table.loc[table["Pair_Filter"] == "S > 0 pairs", "Sig"] = "Descriptive"
    return table


def add_fdr_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Add Wilcoxon_FDR_p (Benjamini-Hochberg across tested all-pairs rows)
    and base the Sig stars of those rows on the adjusted value."""
    table = table.copy()
    table["Wilcoxon_FDR_p"] = np.nan
    tested = (table["Pair_Filter"] == "All pairs") & table["Wilcoxon_p"].notna()
    if tested.any():
        adjusted = multipletests(
            table.loc[tested, "Wilcoxon_p"].to_numpy(dtype=float),
            method="fdr_bh",
        )[1]
        table.loc[tested, "Wilcoxon_FDR_p"] = adjusted
        table.loc[tested, "Sig"] = [significance_stars(p) for p in adjusted]
    return table


def extract_kg_row(matched_df: pd.DataFrame, domain: str,
                   matching_var: str, pair_filter: str) -> dict:
    """Build one known-groups table row from a set of matched pairs."""
    n_pairs = len(matched_df)
    author_s = matched_df["author_s_index"].values
    control_s = matched_df["control_s_index"].values

    author_wins = (author_s > control_s).sum()
    ties = (author_s == control_s).sum()
    control_wins = (author_s < control_s).sum()
    direction = paired_direction_tests(author_s, control_s)

    # One-sided paired Wilcoxon, as in known_groups_validation.primary_analysis
    p_value = np.nan
    if n_pairs >= config.MIN_PAIRS_FOR_DOMAIN_TEST and not np.all(author_s == control_s):
        p_value = paired_wilcoxon(author_s, control_s)["wilcoxon_p_value"]
    sig = "N/A" if np.isnan(p_value) else significance_stars(p_value)

    return {
        "Domain": domain,
        "Matching_Variable": matching_var,
        "Pair_Filter": pair_filter,
        "N_pairs": n_pairs,
        "Author_wins": author_wins,
        "Ties": ties,
        "Control_wins": control_wins,
        "Pct_Author_wins": round(100 * author_wins / n_pairs, 1) if n_pairs > 0 else 0,
        "Pct_Control_wins": round(100 * control_wins / n_pairs, 1) if n_pairs > 0 else 0,
        "Pct_S>0_Authors": round((author_s > 0).mean() * 100, 1),
        "Pct_S>0_Controls": round((control_s > 0).mean() * 100, 1),
        "Median_S_Author": round(np.median(author_s), 1),
        "Median_S_Control": round(np.median(control_s), 1),
        "Mean_S_Author": round(author_s.mean(), 2),
        "Mean_S_Control": round(control_s.mean(), 2),
        "Wilcoxon_p": p_value,
        "Sign_test_p": direction["sign_test_p_value"],
        "Paired_S_positive_p": direction["positive_mcnemar_p_value"],
        "Sig": sig,
    }


def _extract_kg_row_from_result(result: dict, domain: str,
                                 matching_var: str, pair_filter: str) -> dict:
    """Build a descriptive table row from an s_positive_analysis() result."""
    n_pairs = result.get("n_pairs_s_positive", result.get("n_pairs", np.nan))
    prop_author = result.get("prop_pairs_author_wins", 0)
    prop_ties = result.get("prop_pairs_ties", 0)
    prop_control = result.get("prop_pairs_control_wins", 0)

    prop_s_pos_auth = result.get("prop_s_positive_authors", 0)
    prop_s_pos_ctrl = result.get("prop_s_positive_controls", 0)

    if not np.isnan(n_pairs) and n_pairs > 0:
        author_wins = round(prop_author * n_pairs)
        ties = round(prop_ties * n_pairs)
        control_wins = round(prop_control * n_pairs)
        pct_author = round(100 * prop_author, 1)
        pct_control = round(100 * prop_control, 1)
    else:
        author_wins = np.nan
        ties = np.nan
        control_wins = np.nan
        pct_author = np.nan
        pct_control = np.nan

    return {
        "Domain": domain,
        "Matching_Variable": matching_var,
        "Pair_Filter": pair_filter,
        "N_pairs": n_pairs,
        "Author_wins": author_wins,
        "Ties": ties,
        "Control_wins": control_wins,
        "Pct_Author_wins": pct_author,
        "Pct_Control_wins": pct_control,
        "Pct_S>0_Authors": round(prop_s_pos_auth * 100, 1),
        "Pct_S>0_Controls": round(prop_s_pos_ctrl * 100, 1),
        "Median_S_Author": round(result.get("median_s_authors", np.nan), 1),
        "Median_S_Control": round(result.get("median_s_controls", np.nan), 1),
        "Mean_S_Author": round(result.get("mean_s_authors", np.nan), 2),
        "Mean_S_Control": round(result.get("mean_s_controls", np.nan), 2),
        "Wilcoxon_p": np.nan,
        "Sign_test_p": result.get("sign_test_p_value", np.nan),
        "Paired_S_positive_p": result.get("positive_mcnemar_p_value", np.nan),
        "Sig": "Descriptive",
    }


def save_known_groups_table(df: pd.DataFrame) -> str:
    """Generate and save known groups table."""
    ensure_tables_dir()
    table = generate_known_groups_table(df)
    path = os.path.join(TABLES_DIR, "table_known_groups.csv")
    table.to_csv(path, index=False)
    return path


# =============================================================================
# 4. HIDDEN STEWARDS TABLE
# =============================================================================

def generate_hidden_stewards_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate hidden stewards summary table.

    Shows steward counts and thresholds by domain and comparison metric.
    """
    rows = []

    domain_configs = get_all_domain_configs()

    for domain, domain_label in domain_configs:
        for metric in COMPARISON_METRICS:
            if metric.column not in df.columns:
                continue

            try:
                stewards_pct, thresholds_pct = find_hidden_stewards(
                    df, metric=metric, domain=domain
                )

                df_domain = filter_by_domain(df, domain)
                n_total = len(df_domain)
                n_s_positive = len(df_domain[df_domain["s_index"] > 0])

                rows.append({
                    "Domain": domain_label,
                    "Comparison_Metric": metric.name,
                    "N_total": n_total,
                    "N_S>0": n_s_positive,
                    "N_stewards_percentile": len(stewards_pct),
                    "Pct_stewards_of_total": round(100 * len(stewards_pct) / n_total, 2) if n_total > 0 else 0,
                    "Pct_stewards_of_S>0": round(100 * len(stewards_pct) / n_s_positive, 2) if n_s_positive > 0 else 0,
                    "S_threshold": thresholds_pct.get("s_threshold", np.nan),
                    "Metric_threshold": thresholds_pct.get("metric_threshold", np.nan),
                })
            except Exception as e:
                rows.append({
                    "Domain": domain_label,
                    "Comparison_Metric": metric.name,
                    "N_total": np.nan,
                    "N_S>0": np.nan,
                    "N_stewards_percentile": np.nan,
                    "Pct_stewards_of_total": np.nan,
                    "Pct_stewards_of_S>0": np.nan,
                    "S_threshold": np.nan,
                    "Metric_threshold": np.nan,
                    "Error": f"{str(e)[:30]}",
                })

    return pd.DataFrame(rows)


def save_hidden_stewards_table(df: pd.DataFrame) -> str:
    """Generate and save hidden stewards table."""
    ensure_tables_dir()
    table = generate_hidden_stewards_table(df)
    path = os.path.join(TABLES_DIR, "table_hidden_stewards.csv")
    table.to_csv(path, index=False)
    return path


# =============================================================================
# ALL TABLES
# =============================================================================

def generate_all_supplementary_tables(df: pd.DataFrame) -> dict:
    """
    Generate all supplementary tables.

    Returns dict with paths to all generated tables.
    """
    print("\nGenerating supplementary tables...")

    paths = {}

    print("  1. Correlation table...")
    paths["correlations"] = save_correlation_table(df)
    print(f"     Saved: {paths['correlations']}")

    print("  2. Conditional entropy table...")
    paths["entropy"] = save_entropy_table(df)
    print(f"     Saved: {paths['entropy']}")

    print("  3. Known groups validation table...")
    paths["known_groups"] = save_known_groups_table(df)
    print(f"     Saved: {paths['known_groups']}")

    print("  4. Hidden stewards table...")
    paths["hidden_stewards"] = save_hidden_stewards_table(df)
    print(f"     Saved: {paths['hidden_stewards']}")

    print(f"\nAll tables saved to: {TABLES_DIR}/")

    return paths


if __name__ == "__main__":
    from data_loader import load_and_prepare
    generate_all_supplementary_tables(load_and_prepare(config.CLEANED_DATA_FILE))
