"""Known-groups sensitivity analysis using only the two strict data journals.

The known group is defined by publications in Scientific Data or Earth System
Science Data only. GigaScience and Biodiversity Data Journal are set to zero
on an in-memory copy of the cleaned data, so cleaned_data.csv and the primary
four-journal outputs are not modified.

Run `python main.py` first; it writes cleaned_data.csv.

Usage (from the repository root):
    python utilities/run_strict_journal_known_groups.py

Outputs:

* output/known_groups_strict_journals_report.txt
* output/tables/table_known_groups_strict_journals.csv
* output/data/known_groups_strict_journals/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import known_groups_validation as kg  # noqa: E402
import supplementary_tables  # noqa: E402
from validation import apply_fdr_correction  # noqa: E402


INCLUDED_JOURNAL_COLUMNS = ("scientific_data_pubs", "essd_pubs")
EXCLUDED_JOURNAL_COLUMNS = ("gigascience_pubs", "bds_pubs")
ANALYSIS_LABEL = "Scientific Data + Earth System Science Data only"


def apply_strict_journal_definition(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy in which only the two strict journals can qualify authors."""
    required = set(INCLUDED_JOURNAL_COLUMNS + EXCLUDED_JOURNAL_COLUMNS)
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Missing journal columns required for strict analysis: {missing}")

    strict_df = df.copy()
    for column in EXCLUDED_JOURNAL_COLUMNS:
        strict_df[column] = 0
    return strict_df


def to_builtin(value: Any) -> Any:
    """Recursively convert NumPy/Pandas values to JSON-serializable objects."""
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_builtin(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def build_inference_summary(primary: dict, per_domain: list[dict]) -> dict:
    """Combine overall and per-domain results and add BH-FDR adjusted p-values."""
    overall = dict(primary)
    overall["analysis"] = "Overall"
    domain_results = [overall]
    for result in per_domain:
        item = dict(result)
        item["analysis"] = item.get("domain", "Unknown")
        domain_results.append(item)

    inference = {"domain_results": domain_results}
    apply_fdr_correction(inference)
    return inference


def _enrich_stability_with_fdr(stability: dict) -> dict:
    """Add per-draw BH-FDR values and field-level stability summaries."""
    stability = dict(stability)
    records = pd.DataFrame(stability["replicate_records"])
    records["wilcoxon_fdr_p"] = np.nan

    primary = records[records["pair_filter"] == "all_pairs"]
    for _, replicate_rows in primary.groupby("replicate"):
        valid = replicate_rows["wilcoxon_p"].notna()
        valid_index = replicate_rows.index[valid]
        if len(valid_index) == 0:
            continue
        records.loc[valid_index, "wilcoxon_fdr_p"] = multipletests(
            records.loc[valid_index, "wilcoxon_p"].to_numpy(dtype=float),
            method="fdr_bh",
        )[1]

    fdr_summaries = {}
    primary = records[records["pair_filter"] == "all_pairs"]
    for analysis in ["Overall"] + kg.ALL_DOMAINS:
        rows = primary[primary["analysis"] == analysis]
        valid_fdr = rows["wilcoxon_fdr_p"].dropna()
        fdr_summaries[analysis] = {
            "prop_fdr_below_0_05": (
                float((valid_fdr < 0.05).mean()) if len(valid_fdr) else np.nan
            ),
            "prop_fdr_below_0_001": (
                float((valid_fdr < 0.001).mean()) if len(valid_fdr) else np.nan
            ),
            "author_s_positive_rate_median": float(
                rows["author_s_positive_rate"].median()
            ),
            "author_s_positive_rate_min": float(rows["author_s_positive_rate"].min()),
            "author_s_positive_rate_max": float(rows["author_s_positive_rate"].max()),
            "control_s_positive_rate_median": float(
                rows["control_s_positive_rate"].median()
            ),
            "control_s_positive_rate_min": float(
                rows["control_s_positive_rate"].min()
            ),
            "control_s_positive_rate_max": float(
                rows["control_s_positive_rate"].max()
            ),
        }

    stability["replicate_records"] = records.to_dict(orient="records")
    stability["fdr_summaries"] = fdr_summaries
    return stability


def _add_stability_columns(table: pd.DataFrame, stability: dict) -> pd.DataFrame:
    """Add per-row summaries of the stability draws to the known-groups table."""
    table = table.copy()
    stability_columns = [
        "Stability_p_lt_0.05_pct",
        "Stability_p_lt_0.001_pct",
        "Stability_author_win_median_pct",
        "Stability_author_win_min_pct",
        "Stability_author_win_max_pct",
        "Stability_FDR_p_lt_0.05_pct",
        "Stability_author_S_positive_median_pct",
        "Stability_control_S_positive_median_pct",
    ]
    for column in stability_columns:
        table[column] = np.nan

    summaries = {"All Domains": stability}
    summaries.update(stability.get("domain_summaries", {}))
    for domain, summary in summaries.items():
        mask = (table["Domain"] == domain) & (table["Pair_Filter"] == "All pairs")
        if not mask.any():
            continue
        table.loc[mask, "Stability_p_lt_0.05_pct"] = 100 * summary.get(
            "prop_p_below_0_05", np.nan
        )
        table.loc[mask, "Stability_p_lt_0.001_pct"] = 100 * summary.get(
            "prop_p_below_0_001", np.nan
        )
        table.loc[mask, "Stability_author_win_median_pct"] = 100 * summary.get(
            "win_rate_median", np.nan
        )
        table.loc[mask, "Stability_author_win_min_pct"] = 100 * summary.get(
            "win_rate_min", np.nan
        )
        table.loc[mask, "Stability_author_win_max_pct"] = 100 * summary.get(
            "win_rate_max", np.nan
        )
        fdr_summary = stability.get("fdr_summaries", {}).get(
            "Overall" if domain == "All Domains" else domain, {}
        )
        table.loc[mask, "Stability_FDR_p_lt_0.05_pct"] = 100 * fdr_summary.get(
            "prop_fdr_below_0_05", np.nan
        )
        table.loc[mask, "Stability_author_S_positive_median_pct"] = 100 * (
            fdr_summary.get("author_s_positive_rate_median", np.nan)
        )
        table.loc[mask, "Stability_control_S_positive_median_pct"] = 100 * (
            fdr_summary.get("control_s_positive_rate_median", np.nan)
        )
    return table


def format_p(value: Any) -> str:
    """Format a p-value for the text reports ("not tested" when absent)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "not tested"
    return f"{float(value):.6g}"


def _build_report(
    n_rows: int,
    prepared: pd.DataFrame,
    matching_report: dict,
    diagnostics: dict,
    inference: dict,
    stability: dict,
    nn_report: dict,
    nn_primary: dict,
    seed: int,
    replicates: int,
) -> str:
    lines = [
        "STRICT-JOURNAL KNOWN-GROUPS VALIDATION",
        "=" * 42,
        "",
        f"Definition: {ANALYSIS_LABEL}",
        "Excluded as qualifying journals: GigaScience; Biodiversity Data Journal",
        f"Input rows: {n_rows:,}",
        f"Headline matching seed: {seed}",
        f"Eligible-control draws: {replicates}",
        "Matching: exact 1:1 without replacement on (primary_domain, h_index)",
        "Inference: one-sided paired Wilcoxon; BH FDR across Overall and computable fields",
        "",
        "MATCHING",
        "--------",
        f"Eligible strict-journal authors: {int(prepared['is_data_paper_author'].sum())}",
        f"Exact matched pairs: {matching_report['n_matched']}",
        f"Authors without exact controls: {matching_report['n_excluded_no_controls']}",
        f"Researchers excluded for no classified domain: {matching_report['n_excluded_no_domain']}",
        f"Exact h-index match rate: {diagnostics.get('pct_exact_h', np.nan):.1f}%",
        f"Citation SMD: {diagnostics.get('citation_smd', np.nan):.4f}",
        "",
        "PRIMARY AND FIELD RESULTS",
        "-------------------------",
    ]

    for result in inference["domain_results"]:
        analysis = result.get("analysis", result.get("domain", "Unknown"))
        if "error" in result:
            lines.append(f"{analysis}: {result['error']}")
            continue
        n_pairs = result.get("n_pairs", 0)
        raw_p = result.get("wilcoxon_p_value")
        fdr_p = result.get("fdr_adjusted_p")
        warning = result.get("warning")
        line = (
            f"{analysis}: n={n_pairs}; S>0 authors "
            f"{100 * result.get('prop_s_positive_authors', np.nan):.1f}% vs controls "
            f"{100 * result.get('prop_s_positive_controls', np.nan):.1f}%; "
            f"author wins {100 * result.get('prop_pairs_author_wins', np.nan):.1f}% vs "
            f"control wins {100 * result.get('prop_pairs_control_wins', np.nan):.1f}%; "
            f"Wilcoxon p={format_p(raw_p)}; FDR p={format_p(fdr_p)}"
        )
        if warning:
            line += f"; {warning}"
        lines.append(line)

    lines.extend(
        [
            "",
            f"{replicates}-DRAW STABILITY",
            "-" * (len(str(replicates)) + 15),
            (
                f"Overall: p<0.05 in {100 * stability['prop_p_below_0_05']:.1f}% of draws; "
                f"p<0.001 in {100 * stability['prop_p_below_0_001']:.1f}%; "
                f"FDR p<0.05 in "
                f"{100 * stability['fdr_summaries']['Overall']['prop_fdr_below_0_05']:.1f}%; "
                f"p range {format_p(stability['p_min'])} to {format_p(stability['p_max'])}; "
                f"median author-win rate {100 * stability['win_rate_median']:.1f}%"
            ),
        ]
    )
    for domain, summary in stability.get("domain_summaries", {}).items():
        fdr_summary = stability["fdr_summaries"][domain]
        tested = not np.isnan(summary.get("prop_p_below_0_05", np.nan))
        if not tested:
            lines.append(f"{domain}: not tested in stability draws (field sample below threshold)")
            continue
        lines.append(
            f"{domain}: p<0.05 in {100 * summary['prop_p_below_0_05']:.1f}% of draws; "
            f"p<0.001 in {100 * summary['prop_p_below_0_001']:.1f}%; "
            f"FDR p<0.05 in {100 * fdr_summary['prop_fdr_below_0_05']:.1f}%; "
            f"median S>0 {100 * fdr_summary['author_s_positive_rate_median']:.1f}% authors "
            f"vs {100 * fdr_summary['control_s_positive_rate_median']:.1f}% controls; "
            f"median author-win rate {100 * summary['win_rate_median']:.1f}%"
        )

    lines.extend(
        [
            "",
            "NEAREST-NEIGHBOR SENSITIVITY",
            "----------------------------",
            f"Additional nearest-neighbor pairs: {nn_report.get('n_nn_matched', 0)}",
            f"Authors still unmatched: {nn_report.get('n_nn_unmatched', 0)}",
            f"Total sensitivity pairs: {nn_primary.get('n_pairs', 0)}",
            (
                f"Sensitivity result: S>0 authors "
                f"{100 * nn_primary.get('prop_s_positive_authors', np.nan):.1f}% vs controls "
                f"{100 * nn_primary.get('prop_s_positive_controls', np.nan):.1f}%; "
                f"Wilcoxon p={format_p(nn_primary.get('wilcoxon_p_value'))}"
            ),
            "",
            "INTERPRETATION BOUNDARY",
            "-----------------------",
            "This is a journal-based known-groups test. It supports convergent validity at the",
            "aggregate level; it does not prove that every coauthor personally deposited data",
            "or that journal publication caused a higher S-index.",
            "",
        ]
    )
    return "\n".join(lines)


def run_strict_journal_analysis(
    df: pd.DataFrame,
    *,
    seed: int = kg.RANDOM_SEED,
    replicates: int = kg.DEFAULT_STABILITY_REPLICATES,
) -> dict:
    """Run the complete strict-journal analysis without writing output files."""
    strict_df = apply_strict_journal_definition(df)
    prepared = kg.prepare_data(strict_df)

    expected_flag = strict_df[list(INCLUDED_JOURNAL_COLUMNS)].fillna(0).sum(axis=1) > 0
    if not prepared["is_data_paper_author"].equals(expected_flag):
        raise AssertionError("Strict-journal author flag does not match the two included columns")

    matched, matching_report = kg.perform_matching(prepared, seed=seed)
    diagnostics = kg.matching_diagnostics(matched)
    primary = kg.primary_analysis(matched)
    per_domain = kg.all_domains_matched_analysis(matched)
    s_positive = kg.all_domains_s_positive_analysis(matched)
    stability = kg.matching_stability_analysis(
        prepared, n_replicates=replicates, base_seed=seed
    )
    stability = _enrich_stability_with_fdr(stability)
    nn_matched, nn_report = kg.perform_matching_with_nn_sensitivity(prepared, seed=seed)
    nn_primary = kg.primary_analysis(nn_matched)
    inference = build_inference_summary(primary, per_domain)

    table = supplementary_tables.generate_known_groups_table(strict_df, seed=seed)
    table = _add_stability_columns(table, stability)

    return {
        "strict_df": strict_df,
        "prepared": prepared,
        "matched": matched,
        "matching_report": matching_report,
        "diagnostics": diagnostics,
        "primary": primary,
        "per_domain": per_domain,
        "s_positive": s_positive,
        "stability": stability,
        "nn_matched": nn_matched,
        "nn_report": nn_report,
        "nn_primary": nn_primary,
        "inference": inference,
        "table": table,
    }


def write_outputs(results: dict, output_root: Path, *, seed: int, replicates: int) -> dict:
    """Write the strict-journal outputs to their own paths under output_root."""
    output_root.mkdir(parents=True, exist_ok=True)
    tables_dir = output_root / "tables"
    data_dir = output_root / "data" / "known_groups_strict_journals"
    tables_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    table_path = tables_dir / "table_known_groups_strict_journals.csv"
    matched_path = data_dir / "matched_pairs.csv"
    nn_path = data_dir / "matched_pairs_nn_sensitivity.csv"
    stability_path = data_dir / "stability_replicates.csv"
    excluded_path = data_dir / "excluded_authors.csv"
    summary_path = data_dir / "analysis_summary.json"
    report_path = output_root / "known_groups_strict_journals_report.txt"

    results["table"].to_csv(table_path, index=False)
    results["matched"].to_csv(matched_path, index=False)
    results["nn_matched"].to_csv(nn_path, index=False)
    pd.DataFrame(results["stability"]["replicate_records"]).to_csv(
        stability_path, index=False
    )
    pd.DataFrame(results["matching_report"].get("excluded_authors", [])).to_csv(
        excluded_path, index=False
    )

    summary = {
        "analysis_label": ANALYSIS_LABEL,
        "included_journal_columns": INCLUDED_JOURNAL_COLUMNS,
        "excluded_journal_columns": EXCLUDED_JOURNAL_COLUMNS,
        "seed": seed,
        "replicates": replicates,
        "matching_report": results["matching_report"],
        "diagnostics": results["diagnostics"],
        "inference": results["inference"],
        "s_positive": results["s_positive"],
        "stability": {
            key: value
            for key, value in results["stability"].items()
            if key != "replicate_records"
        },
        "nn_report": {
            key: value
            for key, value in results["nn_report"].items()
            if key != "excluded_authors"
        },
        "nn_primary": results["nn_primary"],
    }
    summary_path.write_text(
        json.dumps(to_builtin(summary), indent=2, sort_keys=True), encoding="utf-8"
    )

    report = _build_report(
        n_rows=len(results["strict_df"]),
        prepared=results["prepared"],
        matching_report=results["matching_report"],
        diagnostics=results["diagnostics"],
        inference=results["inference"],
        stability=results["stability"],
        nn_report=results["nn_report"],
        nn_primary=results["nn_primary"],
        seed=seed,
        replicates=replicates,
    )
    report_path.write_text(report, encoding="utf-8")

    return {
        "report": report_path,
        "table": table_path,
        "matched_pairs": matched_path,
        "nn_pairs": nn_path,
        "stability": stability_path,
        "excluded_authors": excluded_path,
        "summary": summary_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ANALYSIS_DIR / "cleaned_data.csv",
        help="Cleaned researcher-level CSV",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ANALYSIS_DIR / "output",
        help="Output root directory",
    )
    parser.add_argument("--seed", type=int, default=kg.RANDOM_SEED)
    parser.add_argument(
        "--replicates", type=int, default=kg.DEFAULT_STABILITY_REPLICATES
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    results = run_strict_journal_analysis(
        df, seed=args.seed, replicates=args.replicates
    )
    paths = write_outputs(
        results, args.output_root, seed=args.seed, replicates=args.replicates
    )

    print(f"Strict-journal eligible authors: {int(results['prepared'].is_data_paper_author.sum())}")
    print(f"Exact matched pairs: {len(results['matched'])}")
    print(f"Overall Wilcoxon p: {results['primary']['wilcoxon_p_value']:.6g}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
