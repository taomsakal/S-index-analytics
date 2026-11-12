"""
S-Index validation pipeline.

    data/ -> resampled_data.csv -> cleaned_data.csv -> output/

Analyses:
1. Correlation with traditional metrics
2. Conditional entropy H(S | metric)
3. Known groups (data paper authors vs matched controls)
4. Hidden stewards (high S-index, low traditional metric)

Each analysis is run against both h-index and citation count, for all
researchers and for each of the four scientific domains.

Usage (from the repository root):
    python main.py
"""

import os
import json

import config
from config import COMPARISON_METRICS, get_all_domain_configs, get_data_dir
from data_loader import (
    load_and_prepare, load_raw_data, postprocess_resampled_data, save_cleaned_data,
    filter_for_resampling,
)
from resample import resample_dataframe_to_file, calculate_effective_sample_size
from validation import generate_validation_report, find_hidden_stewards, compute_conditional_entropy
from visualization import generate_all_domain_plots
from known_groups_validation import run_known_groups_validation, derive_primary_domain
from supplementary_tables import generate_all_supplementary_tables


def export_domain_artifacts(df):
    """Write hidden-steward CSVs and conditional-entropy JSON for each domain."""
    for domain, domain_label in get_all_domain_configs():
        print(f"\n  --- {domain_label} ---")
        data_dir = get_data_dir(domain)
        os.makedirs(data_dir, exist_ok=True)

        entropy_results = {}
        for metric in COMPARISON_METRICS:
            if metric.column not in df.columns:
                print(f"    Skipping {metric.name}: column not found")
                entropy_results[metric.short_name] = {"error": f"Column '{metric.column}' not found"}
                continue

            try:
                stewards, _ = find_hidden_stewards(df, metric=metric, domain=domain)
                stewards_path = os.path.join(data_dir, f"stewards_vs_{metric.short_name}.csv")
                stewards.to_csv(stewards_path, index=False)
                print(f"    Hidden stewards (vs {metric.name}): {len(stewards)} found")
            except Exception as e:
                print(f"    Error computing hidden stewards vs {metric.name}: {e}")

            metric_results = {}
            for key, s_positive_only in [("all_researchers", False), ("s_positive_only", True)]:
                try:
                    metric_results[key] = compute_conditional_entropy(
                        df, s_positive_only=s_positive_only, metric=metric, domain=domain
                    )
                except Exception as e:
                    metric_results[key] = {"error": str(e)}
            entropy_results[metric.short_name] = metric_results

        entropy_path = os.path.join(data_dir, "conditional_entropy.json")
        with open(entropy_path, "w", encoding="utf-8") as f:
            json.dump(entropy_results, f, indent=2)
        print(f"    Conditional entropy saved to: {entropy_path}")


def main():
    print("=" * 60)
    print("S-INDEX VALIDATION PIPELINE")
    print("=" * 60)

    pipeline_stats = {}

    # Step 1: Resample raw data to correct for prolific-author bias
    print("\n[Step 1] Resampling raw data...")
    raw_df = load_raw_data(config.RAW_DATA_SOURCE)
    print(f"  Source: {config.RAW_DATA_SOURCE}")
    print(f"  Raw data size: {len(raw_df)}")
    pipeline_stats["n_raw"] = len(raw_df)

    raw_df = filter_for_resampling(raw_df, verbose=True)
    pipeline_stats["n_filtered"] = len(raw_df)

    ess = calculate_effective_sample_size(raw_df)
    print(f"  Effective Sample Size (ESS): {ess} ({ess/len(raw_df)*100:.1f}% of filtered)")
    pipeline_stats["ess"] = ess

    resample_dataframe_to_file(
        df=raw_df,
        output_path=config.RESAMPLED_DATA_FILE,
        seed=config.RANDOM_SEED,
        n_samples=ess,
    )

    # Step 2: Clean and deduplicate
    print("\n[Step 2] Cleaning and deduplicating data...")
    df = load_and_prepare(config.RESAMPLED_DATA_FILE)
    print(f"  Loaded {len(df)} researchers from resampled data")
    pipeline_stats["n_resampled"] = len(df)

    df = postprocess_resampled_data(df, verbose=True)
    pipeline_stats["n_cleaned"] = len(df)

    # Each researcher gets one primary domain, used by all per-domain analyses
    df = derive_primary_domain(df, seed=config.RANDOM_SEED)
    print("  Primary domain assignment:")
    for dom_name, count in df["primary_domain"].value_counts().items():
        print(f"    {dom_name}: {count:,}")

    save_cleaned_data(df, config.CLEANED_DATA_FILE, verbose=True)

    # Step 3: Validation report
    print("\n[Step 3] Generating validation report...")
    report = generate_validation_report(df, pipeline_stats=pipeline_stats)
    print("\n" + report)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(config.OUTPUT_DIR, "validation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    # Step 4: Per-domain hidden stewards and conditional entropy
    print("\n[Step 4] Exporting analysis artifacts for all domains...")
    export_domain_artifacts(df)

    # Step 5: Figures
    print("\n[Step 5] Generating visualizations for all domains...")
    generate_all_domain_plots(df, save=True, show=False)

    # Step 6: Known-groups validation (matched pairs, stability draws, figures)
    print("\n[Step 6] Running detailed known-groups validation...")
    print("=" * 60)
    run_known_groups_validation(df)

    # Step 7: Summary tables
    print("\n[Step 7] Generating supplementary tables...")
    generate_all_supplementary_tables(df)

    print("\n" + "=" * 60)
    print("VALIDATION PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nOutputs written to {config.OUTPUT_DIR}/ (see README.md for the layout).")
    print(f"Cleaned data: {config.CLEANED_DATA_FILE}")


if __name__ == "__main__":
    main()
