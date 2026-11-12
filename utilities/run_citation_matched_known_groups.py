"""Run the known-groups comparison with controls matched on citation count.

The primary known-groups design matches each data paper author to a control
with the same primary domain and exactly the same h-index. This robustness
analysis replaces h-index with total citation count. Citation counts are close
to continuous, so exact matching is not possible; instead each author is paired
with a control drawn at random from unused controls in the same domain whose
citation count is within 10% of the author's (|log(1 + c_author) -
log(1 + c_control)| <= log(1.1)). As in the primary design, matching is 1:1
without replacement and the draw is repeated across 500 seeds, so the result
does not depend on which eligible control happens to be drawn.

The report also checks the S > 0 subset of pairs. The Wilcoxon signed-rank
test discards zero differences, and every tied pair is a zero-zero pair, so
the test on S > 0 pairs reproduces the all-pairs test. This is why the
known-groups tables label the S > 0 rows as descriptive.

Run `python main.py` first; it writes cleaned_data.csv.

Usage (from the repository root):
    python utilities/run_citation_matched_known_groups.py

Outputs (the primary h-index-matched outputs are not modified):

* output/known_groups_citation_matched_report.txt
* output/tables/table_known_groups_citation_matched.csv
* output/data/known_groups_citation_matched/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import config  # noqa: E402
import known_groups_validation as kg  # noqa: E402
import supplementary_tables  # noqa: E402
from utilities.run_strict_journal_known_groups import (  # noqa: E402
    build_inference_summary,
    format_p,
    to_builtin,
)


# Controls qualify when their citation count is within 10% of the author's
# (on the log(1 + citations) scale, so zero-citation profiles are defined).
CITATION_CALIPER_LOG = float(np.log(1.1))
MATCHING_LABEL = "citation count (within 10%)"
ANALYSES = ["Overall"] + kg.ALL_DOMAINS


def perform_citation_matching(
    df_prepared: pd.DataFrame,
    seed: int = kg.RANDOM_SEED,
    caliper: float = CITATION_CALIPER_LOG,
) -> tuple[pd.DataFrame, dict]:
    """1:1 matching without replacement on primary domain and citation count.

    Each author is paired with a control drawn at random from the unused
    controls in the same domain whose log(1 + citations) lies within the
    caliper. Authors with the fewest eligible controls are matched first; ties
    in that order are broken at random. Authors with no eligible control are
    left unmatched and listed in the report.
    """
    rng = np.random.default_rng(seed)

    df_valid = df_prepared[df_prepared["primary_domain"] != "no_domain"]
    authors = df_valid[df_valid["is_data_paper_author"]].sort_values("openalex_id")
    controls = df_valid[~df_valid["is_data_paper_author"]].sort_values("openalex_id")

    pair_frames = []
    excluded_authors = []

    for domain, author_cell in authors.groupby("primary_domain", sort=True):
        control_cell = controls[controls["primary_domain"] == domain]
        control_log = np.log1p(control_cell["cited_by_count"].to_numpy(dtype=float))
        author_log = np.log1p(author_cell["cited_by_count"].to_numpy(dtype=float))
        used = np.zeros(len(control_cell), dtype=bool)

        n_eligible = np.array(
            [np.sum(np.abs(control_log - value) <= caliper) for value in author_log]
        )
        order = np.lexsort((rng.random(len(author_cell)), n_eligible))

        author_idx, control_idx = [], []
        for i in order:
            eligible = np.flatnonzero(
                (np.abs(control_log - author_log[i]) <= caliper) & ~used
            )
            if len(eligible) == 0:
                row = author_cell.iloc[i]
                excluded_authors.append({
                    "author_id": row["openalex_id"],
                    "domain": domain,
                    "h_index": row["h_index"],
                    "cited_by_count": row["cited_by_count"],
                    "s_index": row["s_index"],
                })
                continue
            j = rng.choice(eligible)
            used[j] = True
            author_idx.append(i)
            control_idx.append(j)

        if not author_idx:
            continue
        matched_authors = author_cell.iloc[author_idx]
        matched_controls = control_cell.iloc[control_idx]
        pair_frames.append(pd.DataFrame({
            "author_id": matched_authors["openalex_id"].to_numpy(),
            "control_id": matched_controls["openalex_id"].to_numpy(),
            "domain": domain,
            "author_s_index": matched_authors["s_index"].to_numpy(),
            "control_s_index": matched_controls["s_index"].to_numpy(),
            "author_h_index": matched_authors["h_index"].to_numpy(),
            "control_h_index": matched_controls["h_index"].to_numpy(),
            "author_cited_by": matched_authors["cited_by_count"].to_numpy(),
            "control_cited_by": matched_controls["cited_by_count"].to_numpy(),
            "match_type": "citation_caliper",
        }))

    matched_df = (
        pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    )
    report = {
        "n_data_paper_authors_total": len(authors),
        "n_matched": len(matched_df),
        "n_excluded_no_controls": len(excluded_authors),
        "n_excluded_no_domain": int((df_prepared["primary_domain"] == "no_domain").sum()),
        "n_controls_total": len(controls),
        "domain_counts": authors["primary_domain"].value_counts().to_dict(),
        "excluded_authors": excluded_authors,
        "caliper_log": caliper,
    }
    return matched_df, report


def _smd(a: pd.Series, b: pd.Series) -> float:
    """Standardized mean difference with a pooled standard deviation."""
    pooled_sd = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / pooled_sd) if pooled_sd > 0 else np.nan


def citation_matching_diagnostics(matched_df: pd.DataFrame) -> dict:
    """Balance on the matching variable and on h-index (not matched on)."""
    log_a = np.log1p(matched_df["author_cited_by"].astype(float))
    log_c = np.log1p(matched_df["control_cited_by"].astype(float))
    return {
        "n_pairs": len(matched_df),
        "citation_smd": _smd(matched_df["author_cited_by"], matched_df["control_cited_by"]),
        "log_citation_smd": _smd(log_a, log_c),
        "max_abs_log_citation_diff": float((log_a - log_c).abs().max()),
        "median_citations_authors": float(matched_df["author_cited_by"].median()),
        "median_citations_controls": float(matched_df["control_cited_by"].median()),
        "h_index_smd": _smd(matched_df["author_h_index"], matched_df["control_h_index"]),
        "median_h_authors": float(matched_df["author_h_index"].median()),
        "median_h_controls": float(matched_df["control_h_index"].median()),
        "domain_distribution": matched_df["domain"].value_counts().to_dict(),
    }


def _wilcoxon_p(author_s: np.ndarray, control_s: np.ndarray, n_min: int) -> float:
    """One-sided paired Wilcoxon p-value, or NaN below n_min pairs."""
    if len(author_s) < n_min:
        return np.nan
    return float(kg.paired_wilcoxon(author_s, control_s)["wilcoxon_p_value"])


def citation_stability_analysis(
    df_prepared: pd.DataFrame,
    n_replicates: int = kg.DEFAULT_STABILITY_REPLICATES,
    base_seed: int = kg.RANDOM_SEED,
) -> dict:
    """Repeat citation matching across seeds; BH-FDR within each draw."""
    records = []
    for i in range(n_replicates):
        seed = base_seed + i
        matched_df, _ = perform_citation_matching(df_prepared, seed=seed)
        draw = []
        for analysis in ANALYSES:
            subset = (
                matched_df if analysis == "Overall"
                else matched_df[matched_df["domain"] == analysis]
            )
            author_s = subset["author_s_index"].to_numpy(dtype=float)
            control_s = subset["control_s_index"].to_numpy(dtype=float)
            n_min = 1 if analysis == "Overall" else config.MIN_PAIRS_FOR_DOMAIN_TEST
            n_pairs = len(subset)
            draw.append({
                "replicate": i,
                "seed": seed,
                "analysis": analysis,
                "n_pairs": n_pairs,
                "wilcoxon_p": _wilcoxon_p(author_s, control_s, n_min),
                "author_wins": int(np.sum(author_s > control_s)),
                "control_wins": int(np.sum(author_s < control_s)),
                "author_s_positive_rate": float(np.mean(author_s > 0)) if n_pairs else np.nan,
                "control_s_positive_rate": float(np.mean(control_s > 0)) if n_pairs else np.nan,
            })
        p_values = np.array([row["wilcoxon_p"] for row in draw], dtype=float)
        valid = ~np.isnan(p_values)
        adjusted = np.full(len(p_values), np.nan)
        if valid.any():
            adjusted[valid] = multipletests(p_values[valid], method="fdr_bh")[1]
        for row, fdr_p in zip(draw, adjusted):
            row["wilcoxon_fdr_p"] = fdr_p
        records.extend(draw)

    records_df = pd.DataFrame(records)
    summaries = {}
    for analysis in ANALYSES:
        rows = records_df[records_df["analysis"] == analysis]
        p = rows["wilcoxon_p"].dropna()
        fdr = rows["wilcoxon_fdr_p"].dropna()
        summaries[analysis] = {
            "n_pairs_median": float(rows["n_pairs"].median()),
            "tested": bool(len(p)),
            "p_median": float(p.median()) if len(p) else np.nan,
            "p_min": float(p.min()) if len(p) else np.nan,
            "p_max": float(p.max()) if len(p) else np.nan,
            "prop_p_below_0_05": float((p < 0.05).mean()) if len(p) else np.nan,
            "prop_p_below_0_001": float((p < 0.001).mean()) if len(p) else np.nan,
            "prop_fdr_below_0_05": float((fdr < 0.05).mean()) if len(fdr) else np.nan,
            "author_wins_median": float(rows["author_wins"].median()),
            "author_wins_min": int(rows["author_wins"].min()),
            "author_wins_max": int(rows["author_wins"].max()),
            "control_wins_median": float(rows["control_wins"].median()),
            "control_wins_min": int(rows["control_wins"].min()),
            "control_wins_max": int(rows["control_wins"].max()),
            "author_s_positive_rate_median": float(rows["author_s_positive_rate"].median()),
            "control_s_positive_rate_median": float(rows["control_s_positive_rate"].median()),
        }
    return {
        "n_replicates": n_replicates,
        "summaries": summaries,
        "replicate_records": records,
    }


def s_positive_wilcoxon_check(matched_df: pd.DataFrame) -> list[dict]:
    """Compare the Wilcoxon p on all pairs with the p on S > 0 pairs only.

    Both use the asymptotic method so the comparison is not confounded by
    scipy switching to an exact distribution for small samples without zeros.
    """
    rows = []
    for analysis in ANALYSES:
        subset = (
            matched_df if analysis == "Overall"
            else matched_df[matched_df["domain"] == analysis]
        )
        positive = subset[(subset["author_s_index"] > 0) | (subset["control_s_index"] > 0)]
        nonzero_ties = int(np.sum(
            (positive["author_s_index"] == positive["control_s_index"]).to_numpy()
        ))

        def p_of(frame: pd.DataFrame) -> float:
            a = frame["author_s_index"].to_numpy(dtype=float)
            c = frame["control_s_index"].to_numpy(dtype=float)
            if len(frame) == 0 or np.all(a == c):
                return np.nan
            return float(stats.wilcoxon(
                a, c, alternative="greater", method="asymptotic"
            ).pvalue)

        p_all = p_of(subset)
        p_pos = p_of(positive)
        rows.append({
            "analysis": analysis,
            "n_all_pairs": len(subset),
            "n_s_positive_pairs": len(positive),
            "nonzero_tied_pairs": nonzero_ties,
            "wilcoxon_p_all_pairs": p_all,
            "wilcoxon_p_s_positive_pairs": p_pos,
            "identical": bool(np.isclose(p_all, p_pos, rtol=1e-12, atol=0))
            if not (np.isnan(p_all) or np.isnan(p_pos)) else None,
        })
    return rows


def build_table(matched_df: pd.DataFrame, stability: dict) -> pd.DataFrame:
    """Known-groups table for the citation-matched pairs plus stability summaries."""
    rows = [supplementary_tables.extract_kg_row(
        matched_df, "All Domains", MATCHING_LABEL, "All pairs"
    )]
    for domain in kg.ALL_DOMAINS:
        subset = matched_df[matched_df["domain"] == domain]
        if len(subset) >= 3:
            rows.append(supplementary_tables.extract_kg_row(
                subset, domain, MATCHING_LABEL, "All pairs"
            ))
    table = supplementary_tables.add_fdr_columns(pd.DataFrame(rows))

    for column in [
        "Stability_p_median", "Stability_p_min", "Stability_p_max",
        "Stability_p_lt_0.05_pct", "Stability_p_lt_0.001_pct",
        "Stability_FDR_p_lt_0.05_pct",
        "Stability_author_wins_median", "Stability_control_wins_median",
    ]:
        table[column] = np.nan
    for analysis, summary in stability["summaries"].items():
        domain = "All Domains" if analysis == "Overall" else analysis
        mask = table["Domain"] == domain
        table.loc[mask, "Stability_p_median"] = summary["p_median"]
        table.loc[mask, "Stability_p_min"] = summary["p_min"]
        table.loc[mask, "Stability_p_max"] = summary["p_max"]
        table.loc[mask, "Stability_p_lt_0.05_pct"] = 100 * summary["prop_p_below_0_05"]
        table.loc[mask, "Stability_p_lt_0.001_pct"] = 100 * summary["prop_p_below_0_001"]
        table.loc[mask, "Stability_FDR_p_lt_0.05_pct"] = 100 * summary["prop_fdr_below_0_05"]
        table.loc[mask, "Stability_author_wins_median"] = summary["author_wins_median"]
        table.loc[mask, "Stability_control_wins_median"] = summary["control_wins_median"]
    return table


def _build_report(results: dict, seed: int, replicates: int) -> str:
    matching_report = results["matching_report"]
    diagnostics = results["diagnostics"]
    lines = [
        "CITATION-MATCHED KNOWN-GROUPS VALIDATION",
        "=" * 40,
        "",
        f"Input rows: {len(results['prepared']):,}",
        f"Headline matching seed: {seed}",
        f"Eligible-control draws: {replicates}",
        "Matching: 1:1 without replacement on primary_domain; control drawn at random",
        "  from in-domain controls with citation count within 10% of the author's",
        f"  (|log1p difference| <= {CITATION_CALIPER_LOG:.4f})",
        "Inference: one-sided paired Wilcoxon; BH FDR across Overall and computable fields",
        "",
        "MATCHING",
        "--------",
        f"Eligible authors: {matching_report['n_data_paper_authors_total']}",
        f"Matched pairs: {matching_report['n_matched']}",
        f"Authors without an eligible control: {matching_report['n_excluded_no_controls']}",
        f"Researchers excluded for no classified domain: {matching_report['n_excluded_no_domain']}",
        f"Citation SMD (raw counts): {diagnostics['citation_smd']:.4f}",
        f"Citation SMD (log1p): {diagnostics['log_citation_smd']:.4f}",
        f"Max |log1p citation difference| in a pair: {diagnostics['max_abs_log_citation_diff']:.4f}",
        f"h-index SMD (not matched on): {diagnostics['h_index_smd']:.4f}",
        "",
        "PRIMARY AND FIELD RESULTS (headline seed)",
        "-----------------------------------------",
    ]
    for result in results["inference"]["domain_results"]:
        analysis = result.get("analysis", result.get("domain", "Unknown"))
        if "error" in result:
            lines.append(f"{analysis}: {result['error']}")
            continue
        n_pairs = result.get("n_pairs", 0)
        line = (
            f"{analysis}: n={n_pairs}; author wins "
            f"{round(n_pairs * result.get('prop_pairs_author_wins', np.nan))} vs control wins "
            f"{round(n_pairs * result.get('prop_pairs_control_wins', np.nan))}; S>0 authors "
            f"{100 * result.get('prop_s_positive_authors', np.nan):.1f}% vs controls "
            f"{100 * result.get('prop_s_positive_controls', np.nan):.1f}%; "
            f"Wilcoxon p={format_p(result.get('wilcoxon_p_value'))}; "
            f"FDR p={format_p(result.get('fdr_adjusted_p'))}"
        )
        if result.get("warning"):
            line += f"; {result['warning']}"
        lines.append(line)

    lines.extend(["", f"{replicates}-DRAW STABILITY", "-" * (len(str(replicates)) + 16)])
    for analysis, summary in results["stability"]["summaries"].items():
        if not summary["tested"]:
            lines.append(
                f"{analysis}: n={summary['n_pairs_median']:.0f}; not tested "
                "(field sample below threshold)"
            )
            continue
        lines.append(
            f"{analysis}: n={summary['n_pairs_median']:.0f}; p<0.05 in "
            f"{100 * summary['prop_p_below_0_05']:.1f}% of draws; p<0.001 in "
            f"{100 * summary['prop_p_below_0_001']:.1f}%; FDR p<0.05 in "
            f"{100 * summary['prop_fdr_below_0_05']:.1f}%; median p "
            f"{format_p(summary['p_median'])} (range {format_p(summary['p_min'])} to "
            f"{format_p(summary['p_max'])}); author wins {summary['author_wins_min']}-"
            f"{summary['author_wins_max']}, control wins {summary['control_wins_min']}-"
            f"{summary['control_wins_max']}"
        )

    lines.extend([
        "",
        "S > 0 SUBSET CHECK",
        "------------------",
        "Wilcoxon on all pairs vs on pairs where either member has S > 0 (asymptotic",
        "method for both). Zero-zero pairs have zero difference and are discarded by the",
        "test, so the two p-values agree whenever no nonzero pair is tied.",
    ])
    for label, rows in [
        ("Exact h-index matching", results["s_positive_check_h_index"]),
        ("Citation matching", results["s_positive_check_citation"]),
    ]:
        lines.append(f"{label} (seed {seed}):")
        for row in rows:
            lines.append(
                f"  {row['analysis']}: {row['n_all_pairs']} pairs, "
                f"{row['n_s_positive_pairs']} with S>0, nonzero ties "
                f"{row['nonzero_tied_pairs']}; p all={format_p(row['wilcoxon_p_all_pairs'])}, "
                f"p S>0={format_p(row['wilcoxon_p_s_positive_pairs'])}, "
                f"identical={row['identical']}"
            )
    lines.extend([
        "",
        "INTERPRETATION BOUNDARY",
        "-----------------------",
        "This is a robustness check on the matching variable. Like the primary analysis,",
        "it supports convergent validity at the aggregate level only.",
        "",
    ])
    return "\n".join(lines)


def run_citation_matched_analysis(
    df: pd.DataFrame,
    *,
    seed: int = kg.RANDOM_SEED,
    replicates: int = kg.DEFAULT_STABILITY_REPLICATES,
) -> dict:
    """Run the complete citation-matched analysis without writing files."""
    prepared = kg.prepare_data(df)
    matched, matching_report = perform_citation_matching(prepared, seed=seed)
    diagnostics = citation_matching_diagnostics(matched)
    primary = kg.primary_analysis(matched)
    per_domain = kg.all_domains_matched_analysis(matched)
    inference = build_inference_summary(primary, per_domain)
    stability = citation_stability_analysis(
        prepared, n_replicates=replicates, base_seed=seed
    )
    h_matched, _ = kg.perform_matching(prepared, seed=seed)
    return {
        "prepared": prepared,
        "matched": matched,
        "matching_report": matching_report,
        "diagnostics": diagnostics,
        "primary": primary,
        "per_domain": per_domain,
        "inference": inference,
        "stability": stability,
        "table": build_table(matched, stability),
        "s_positive_check_h_index": s_positive_wilcoxon_check(h_matched),
        "s_positive_check_citation": s_positive_wilcoxon_check(matched),
    }


def write_outputs(results: dict, output_root: Path, *, seed: int, replicates: int) -> dict:
    """Write the citation-matched outputs to their own paths under output_root."""
    tables_dir = output_root / "tables"
    data_dir = output_root / "data" / "known_groups_citation_matched"
    tables_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "report": output_root / "known_groups_citation_matched_report.txt",
        "table": tables_dir / "table_known_groups_citation_matched.csv",
        "matched_pairs": data_dir / "matched_pairs.csv",
        "stability": data_dir / "stability_replicates.csv",
        "excluded_authors": data_dir / "excluded_authors.csv",
        "summary": data_dir / "analysis_summary.json",
    }
    results["table"].to_csv(paths["table"], index=False)
    results["matched"].to_csv(paths["matched_pairs"], index=False)
    pd.DataFrame(results["stability"]["replicate_records"]).to_csv(
        paths["stability"], index=False
    )
    pd.DataFrame(results["matching_report"]["excluded_authors"]).to_csv(
        paths["excluded_authors"], index=False
    )
    summary = {
        "matching_label": MATCHING_LABEL,
        "caliper_log": CITATION_CALIPER_LOG,
        "seed": seed,
        "replicates": replicates,
        "matching_report": results["matching_report"],
        "diagnostics": results["diagnostics"],
        "inference": results["inference"],
        "stability": results["stability"]["summaries"],
        "s_positive_check_h_index": results["s_positive_check_h_index"],
        "s_positive_check_citation": results["s_positive_check_citation"],
    }
    paths["summary"].write_text(
        json.dumps(to_builtin(summary), indent=2, sort_keys=True), encoding="utf-8"
    )
    paths["report"].write_text(
        _build_report(results, seed=seed, replicates=replicates), encoding="utf-8"
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ANALYSIS_DIR / "cleaned_data.csv",
        help="Cleaned researcher-level CSV",
    )
    parser.add_argument(
        "--output-root", type=Path, default=ANALYSIS_DIR / "output",
        help="Output root directory",
    )
    parser.add_argument("--seed", type=int, default=kg.RANDOM_SEED)
    parser.add_argument("--replicates", type=int, default=kg.DEFAULT_STABILITY_REPLICATES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    results = run_citation_matched_analysis(df, seed=args.seed, replicates=args.replicates)
    paths = write_outputs(results, args.output_root, seed=args.seed, replicates=args.replicates)
    print(f"Matched pairs: {len(results['matched'])}")
    print(f"Overall Wilcoxon p: {results['primary']['wilcoxon_p_value']:.6g}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
