"""
Figures for the S-Index validation.

Each plot function takes the cleaned researcher DataFrame, optionally filters
it to one domain, and saves a PNG under output/figures/{domain}/. Styling
lives in viz_config.py.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

import config
from config import (
    ComparisonMetric, H_INDEX_METRIC, COMPARISON_METRICS,
    Domain, filter_by_domain, get_all_domain_configs, get_figures_dir, get_domain_label
)
from known_groups_validation import DATA_JOURNAL_COLUMNS
from viz_config import (COLORS, FONT, FIGURE_SIZES, HEATMAP_CMAP, LINES, MARKERS,
                        setup_style, save_figure, format_log_ticks, add_grid)


def ensure_output_dir(domain: Optional[Domain] = None) -> str:
    """Create and return output/figures/{domain}/ ('all' when domain is None)."""
    output_dir = get_figures_dir(domain)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _steward_plot_data(df: pd.DataFrame, metric_col: str) -> tuple:
    """
    Rows with a valid S-index and metric, plus the hidden steward thresholds
    (same definition as validation.find_hidden_stewards).

    Returns:
        Tuple of (valid_df, s_threshold, metric_threshold)
    """
    valid = df.dropna(subset=["s_index", metric_col])
    s_positive = valid[valid["s_index"] > 0]
    s_threshold = (
        np.percentile(s_positive["s_index"], config.HIDDEN_STEWARD_S_PERCENTILE)
        if len(s_positive) > 0 else 0
    )
    metric_threshold = (
        np.percentile(valid[metric_col], config.HIDDEN_STEWARD_H_PERCENTILE)
        if len(valid) > 0 else 0
    )
    return valid, s_threshold, metric_threshold


def _add_lowess_trend(ax, x_data: np.ndarray, y_data: np.ndarray,
                      log_x: bool = False, frac: float = 0.3):
    """
    Add LOWESS trend line to axis.

    Args:
        ax: Matplotlib axis
        x_data: X values (in original scale)
        y_data: Y values (in original scale, assumed positive for log)
        log_x: Whether x-axis uses log scale
        frac: LOWESS smoothing fraction
    """
    if log_x:
        x_transformed = np.log10(x_data)
    else:
        x_transformed = x_data
    y_log = np.log10(y_data)

    sort_idx = np.argsort(x_transformed)
    x_sorted = x_transformed[sort_idx]
    y_sorted = y_log[sort_idx]

    lowess_result = sm_lowess(y_sorted, x_sorted, frac=frac, return_sorted=True)
    x_lowess, y_lowess = lowess_result[:, 0], lowess_result[:, 1]

    if log_x:
        x_plot = 10 ** x_lowess
    else:
        x_plot = x_lowess
    y_plot = 10 ** y_lowess

    ax.plot(x_plot, y_plot, color=COLORS["mean"], linestyle="-",
            linewidth=LINES["trend"], alpha=0.85, label="Mean Trend", zorder=6)


def _add_lowess_median_trend(ax, x_data: np.ndarray, y_data: np.ndarray,
                             log_x: bool = False, n_bins: int = 20):
    """
    Add median trend line to axis using binned medians with LOWESS smoothing.

    Args:
        ax: Matplotlib axis
        x_data: X values (in original scale)
        y_data: Y values (in original scale, assumed positive for log)
        log_x: Whether x-axis uses log scale
        n_bins: Number of bins for computing medians
    """
    if log_x:
        x_transformed = np.log10(x_data)
    else:
        x_transformed = x_data
    y_log = np.log10(y_data)

    # Bin data and compute medians per bin
    bin_edges = np.linspace(x_transformed.min(), x_transformed.max(), n_bins + 1)
    bin_centers = []
    bin_medians = []
    for i in range(len(bin_edges) - 1):
        mask = (x_transformed >= bin_edges[i]) & (x_transformed < bin_edges[i + 1])
        if i == len(bin_edges) - 2:  # include right edge for last bin
            mask = mask | (x_transformed == bin_edges[i + 1])
        if mask.sum() >= 3:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_medians.append(np.median(y_log[mask]))

    if len(bin_centers) < 4:
        return

    bin_centers = np.array(bin_centers)
    bin_medians = np.array(bin_medians)

    # Smooth the binned medians with LOWESS
    lowess_result = sm_lowess(bin_medians, bin_centers, frac=0.5, return_sorted=True)
    x_lowess, y_lowess = lowess_result[:, 0], lowess_result[:, 1]

    if log_x:
        x_plot = 10 ** x_lowess
    else:
        x_plot = x_lowess
    y_plot = 10 ** y_lowess

    ax.plot(x_plot, y_plot, color=COLORS["median"], linestyle="--",
            linewidth=LINES["trend"], alpha=0.85, label="Median Trend", zorder=6)


def _plot_correlation_heatmap(df, method: str, cbar_label: str, filename: str,
                              save: bool, domain: Optional[Domain]):
    """Heatmap of pairwise correlations between S-index and traditional metrics."""
    setup_style()

    df_filtered = filter_by_domain(df, domain)
    domain_label = get_domain_label(domain)

    label_map = {
        "s_index": "S-index",
        "h_index": "h-index",
        "cited_by_count": "Citations",
        "num_datasets": "Datasets",
    }
    available_cols = [c for c in label_map if c in df_filtered.columns]

    corr_display = df_filtered[available_cols].corr(method=method)
    display_labels = [label_map[c] for c in available_cols]
    corr_display.index = display_labels
    corr_display.columns = display_labels

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["square"])
    ax.grid(False)

    sns.heatmap(
        corr_display,
        annot=True,
        fmt=".2f",
        cmap=HEATMAP_CMAP,
        vmin=0,
        vmax=1,
        square=True,
        ax=ax,
        annot_kws={"size": FONT["size_annotation"] + 1, "weight": "bold"},
        cbar_kws={"label": cbar_label, "shrink": 0.8},
        linewidths=1.0,
        linecolor="white"
    )

    title = "Correlation Matrix"
    if domain_label:
        title += f" ({domain_label})"
    ax.set_title(title, pad=8)
    ax.tick_params(axis='both', labelsize=FONT["size_tick"])

    plt.tight_layout(pad=1.2)

    if save:
        output_dir = ensure_output_dir(domain=domain)
        save_figure(fig, os.path.join(output_dir, filename))

    return fig


def plot_correlation_matrix(df, save=True, domain: Optional[Domain] = None):
    """Heatmap of Pearson correlations between S-index and traditional metrics."""
    return _plot_correlation_heatmap(
        df, "pearson", "Pearson r", "correlation_matrix", save, domain
    )


def plot_spearman_correlation_matrix(df, save=True, domain: Optional[Domain] = None):
    """
    Heatmap of Spearman rank correlations between S-index and traditional metrics.

    Spearman suits these metrics better than Pearson: they are heavily
    right-skewed with many zeros, and their relationships are monotonic but
    not necessarily linear.
    """
    return _plot_correlation_heatmap(
        df, "spearman", "Spearman \u03c1", "spearman_correlation_matrix", save, domain
    )


def plot_s_index_distribution(df, save=True, domain: Optional[Domain] = None):
    """
    Histogram of S-index distribution (log scale, S > 0 only).

    Args:
        df: DataFrame with s_index column
        save: Whether to save the figure
        domain: Domain to filter by (None = all researchers)
    """
    setup_style()

    df_filtered = filter_by_domain(df, domain)
    domain_label = get_domain_label(domain)

    # Filter to S > 0
    s_col = pd.to_numeric(df_filtered["s_index"], errors="coerce")
    all_valid = s_col.dropna()
    nonzero = all_valid[all_valid > 0]

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["double"])

    if len(nonzero) > 0:
        s_min, s_max = float(nonzero.min()), float(nonzero.max())

        # Hybrid bins: integer-aligned 1-10, then log-spaced above
        int_edges = np.arange(1, 11, dtype=float)
        log_edges = np.logspace(np.log10(10), np.log10(s_max), 25)[1:]
        log_bins = np.concatenate([int_edges, log_edges])

        ax.hist(
            nonzero,
            bins=log_bins,
            density=True,
            color=COLORS["highlight"],
            alpha=0.75,
            edgecolor="#F0F0F0",
            linewidth=LINES["hist_edge"],
        )

        ax.set_xscale("log")
        format_log_ticks(ax, axis='x')
        ax.set_xlim(left=s_min * 0.8, right=s_max * 1.2)

        ax.set_xlabel("S-index")
        ax.set_ylabel("Density")
        title = "S-index Distribution"
        if domain_label:
            title += f" ({domain_label})"
        ax.set_title(title)

        # Median and mean lines with consistent colors
        median_val = nonzero.median()
        mean_val = nonzero.mean()

        ax.axvline(median_val, color=COLORS["median"], linestyle="--",
                   linewidth=LINES["reference"], alpha=0.7, label=f"Median: {median_val:.1f}")
        ax.axvline(mean_val, color=COLORS["mean"], linestyle=":",
                   linewidth=LINES["reference"], label=f"Mean: {mean_val:.1f}")

        add_grid(ax, axis='y')
        ax.legend(loc="upper right")

    plt.tight_layout()

    if save:
        output_dir = ensure_output_dir(domain=domain)
        save_figure(fig, os.path.join(output_dir, "s_index_distribution"))

    return fig


def plot_s_index_distribution_clipped(df, save=True, domain: Optional[Domain] = None):
    """
    Histogram of S-index distribution clipped at x=1000 for detail view.

    Shows the bulk of the distribution without extreme outliers.
    Only non-zero values are included.

    Args:
        df: DataFrame with s_index column
        save: Whether to save the figure
        domain: Domain to filter by (None = all researchers)
    """
    setup_style()

    df_filtered = filter_by_domain(df, domain)
    domain_label = get_domain_label(domain)

    # Filter to 0 < S <= 1000
    s_col = pd.to_numeric(df_filtered["s_index"], errors="coerce")
    all_valid = s_col.dropna()
    nonzero = all_valid[all_valid > 0]
    clipped = nonzero[nonzero <= 1000]

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["double"])

    if len(clipped) > 0:
        # Hybrid bins: integer-aligned 1-10, then log-spaced up to 1000
        int_edges = np.arange(1, 11, dtype=float)
        log_edges = np.logspace(np.log10(10), np.log10(1000), 20)[1:]
        log_bins = np.concatenate([int_edges, log_edges])

        ax.hist(
            clipped,
            bins=log_bins,
            density=True,
            color=COLORS["highlight"],
            alpha=0.75,
            edgecolor="#F0F0F0",
            linewidth=LINES["hist_edge"],
        )

        ax.set_xscale("log")
        format_log_ticks(ax, axis='x')
        ax.set_xlim(left=0.8, right=1200)

        ax.set_xlabel("S-index")
        ax.set_ylabel("Density")
        title = "S-index Distribution (S \u2264 1,000)"
        if domain_label:
            title += f" ({domain_label})"
        ax.set_title(title)

        # Median and mean lines
        median_val = clipped.median()
        mean_val = clipped.mean()

        ax.axvline(median_val, color=COLORS["median"], linestyle="--",
                   linewidth=LINES["reference"], alpha=0.7, label=f"Median: {median_val:.1f}")
        ax.axvline(mean_val, color=COLORS["mean"], linestyle=":",
                   linewidth=LINES["reference"], label=f"Mean: {mean_val:.1f}")

        add_grid(ax, axis='y')
        ax.legend(loc="upper right")

    plt.tight_layout()

    if save:
        output_dir = ensure_output_dir(domain=domain)
        save_figure(fig, os.path.join(output_dir, "s_index_distribution_clipped"))

    return fig


def plot_metric_distributions_overlay(df, save=True, domain: Optional[Domain] = None):
    """
    Overlapping density bar chart comparing S-index, h-index, and citation count.

    All three metrics are plotted as density histograms on a single log-log axis
    with shared bins so their shapes are directly comparable.
    Only non-zero values are included (zeros are undefined on log scale).

    Args:
        df: DataFrame with s_index, h_index, and cited_by_count columns
        save: Whether to save the figure
        domain: Domain to filter by (None = all researchers)
    """
    setup_style()

    df_filtered = filter_by_domain(df, domain)
    domain_label = get_domain_label(domain)

    # Define metrics (S-index orange, h-index blue, citations green)
    metrics_info = [
        ("s_index",       "S-index",        COLORS["highlight"]),
        ("h_index",       "h-index",        COLORS["primary"]),
        ("cited_by_count", "Citation count", COLORS["positive"]),
    ]

    # Keep only columns that exist
    metrics_info = [(col, label, color) for col, label, color in metrics_info
                    if col in df_filtered.columns]
    if len(metrics_info) < 2:
        return None

    # Compute global x-range across all metrics for shared bins
    global_min, global_max = np.inf, -np.inf
    nonzero_data = {}
    for col, label, color in metrics_info:
        series = pd.to_numeric(df_filtered[col], errors="coerce").dropna()
        nonzero = series[series > 0]
        nonzero_data[col] = nonzero
        if len(nonzero) > 0:
            global_min = min(global_min, float(nonzero.min()))
            global_max = max(global_max, float(nonzero.max()))

    if global_min >= global_max:
        return None

    # Hybrid bins: integer-aligned edges from 1-10 (where S-index and h-index
    # are integer-valued) then log-spaced above 10 to avoid empty-bin gaps
    int_edges = np.arange(1, 11, dtype=float)  # 1, 2, 3, ..., 10
    log_edges = np.logspace(np.log10(10), np.log10(global_max), 40)[1:]  # skip 10 (already in int_edges)
    shared_bins = np.concatenate([int_edges, log_edges])

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["double"])

    # Plot widest distribution first (back) to narrowest last (front)
    plot_order = list(reversed(metrics_info))

    for col, label, color in plot_order:
        nonzero = nonzero_data[col]
        if len(nonzero) == 0:
            continue

        ax.hist(nonzero, bins=shared_bins, density=True,
                color=color, alpha=0.50,
                edgecolor=color, linewidth=0.4,
                label=f"{label} (n={len(nonzero):,})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    format_log_ticks(ax, axis='both')

    ax.set_xlabel("Value (log scale)")
    ax.set_ylabel("Density")
    title = "Metric Distributions"
    if domain_label:
        title += f" ({domain_label})"
    ax.set_title(title)

    add_grid(ax, axis='y')
    # Reverse legend order so it matches the visual order (top = front)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper right")

    plt.tight_layout()

    if save:
        output_dir = ensure_output_dir(domain=domain)
        save_figure(fig, os.path.join(output_dir, "metric_distributions_overlay"))

    return fig


def plot_metric_distributions_overlay_linear(df, save=True, domain: Optional[Domain] = None):
    """
    Overlapping density bar chart comparing S-index, h-index, and citation count.

    Same as plot_metric_distributions_overlay but with a linear y-axis
    to show the peak densities more intuitively.
    Only non-zero values are included (zeros are undefined on log scale).

    Args:
        df: DataFrame with s_index, h_index, and cited_by_count columns
        save: Whether to save the figure
        domain: Domain to filter by (None = all researchers)
    """
    setup_style()

    df_filtered = filter_by_domain(df, domain)
    domain_label = get_domain_label(domain)

    # Define metrics (S-index orange, h-index blue, citations green)
    metrics_info = [
        ("s_index",       "S-index",        COLORS["highlight"]),
        ("h_index",       "h-index",        COLORS["primary"]),
        ("cited_by_count", "Citation count", COLORS["positive"]),
    ]

    # Keep only columns that exist
    metrics_info = [(col, label, color) for col, label, color in metrics_info
                    if col in df_filtered.columns]
    if len(metrics_info) < 2:
        return None

    # Compute global x-range across all metrics for shared bins
    global_min, global_max = np.inf, -np.inf
    nonzero_data = {}
    for col, label, color in metrics_info:
        series = pd.to_numeric(df_filtered[col], errors="coerce").dropna()
        nonzero = series[series > 0]
        nonzero_data[col] = nonzero
        if len(nonzero) > 0:
            global_min = min(global_min, float(nonzero.min()))
            global_max = max(global_max, float(nonzero.max()))

    if global_min >= global_max:
        return None

    # Hybrid bins: integer-aligned edges from 1-10, then log-spaced above 10
    int_edges = np.arange(1, 11, dtype=float)
    log_edges = np.logspace(np.log10(10), np.log10(global_max), 40)[1:]
    shared_bins = np.concatenate([int_edges, log_edges])

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["double"])

    # Plot widest distribution first (back) to narrowest last (front)
    plot_order = list(reversed(metrics_info))

    for col, label, color in plot_order:
        nonzero = nonzero_data[col]
        if len(nonzero) == 0:
            continue

        ax.hist(nonzero, bins=shared_bins, density=True,
                color=color, alpha=0.50,
                edgecolor=color, linewidth=0.4,
                label=f"{label} (n={len(nonzero):,})")

    ax.set_xscale("log")
    format_log_ticks(ax, axis='x')
    ax.set_xlim(left=0.8, right=100)

    ax.set_xlabel("Value (log scale)")
    ax.set_ylabel("Density")
    title = "Metric Distributions (Linear Density)"
    if domain_label:
        title += f" ({domain_label})"
    ax.set_title(title)

    add_grid(ax, axis='y')
    # Reverse legend order so it matches the visual order (top = front)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper right")

    plt.tight_layout()

    if save:
        output_dir = ensure_output_dir(domain=domain)
        save_figure(fig, os.path.join(output_dir, "metric_distributions_overlay_linear"))

    return fig


def plot_quadrant_analysis(df, save=True, metric: Optional[ComparisonMetric] = None, domain: Optional[Domain] = None):
    """
    S-index vs metric with hidden steward zone highlighted.

    Shows researchers with S > 0. For h-index, includes h=0 points.
    For log-scale metrics (citations), excludes zeros.

    Args:
        df: DataFrame with s_index and metric columns
        save: Whether to save the figure
        metric: Comparison metric (default: h-index)
        domain: Domain to filter by (None = all researchers)
    """
    if metric is None:
        metric = H_INDEX_METRIC

    setup_style()

    df_filtered = filter_by_domain(df, domain)
    domain_label = get_domain_label(domain)

    metric_col = metric.column
    valid_all, s_threshold, metric_threshold = _steward_plot_data(df_filtered, metric_col)

    # For plotting: only show S > 0 researchers
    # For log-scale metrics (citations), also exclude metric = 0
    valid = valid_all[valid_all["s_index"] > 0].copy()
    if metric.use_log_scale:
        valid = valid[valid[metric_col] > 0]

    is_steward = (valid["s_index"] >= s_threshold) & (valid[metric_col] <= metric_threshold)

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_wide"])

    # Regular researchers
    regular = valid[~is_steward]
    ax.scatter(regular[metric_col], regular["s_index"],
               color=COLORS["regular_researcher"],
               **MARKERS["regular"], label="Researchers")

    # Hidden stewards
    steward_data = valid[is_steward]
    if len(steward_data) > 0:
        ax.scatter(steward_data[metric_col], steward_data["s_index"],
                   color=COLORS["hidden_steward"],
                   **MARKERS["highlight"], label="Hidden Stewards", zorder=5)

    # Threshold lines (with labels for legend)
    ax.axhline(s_threshold, color=COLORS["threshold"], linestyle="--",
               linewidth=LINES["threshold"], alpha=0.7,
               label=f"S-index threshold (P{config.HIDDEN_STEWARD_S_PERCENTILE})")
    ax.axvline(metric_threshold, color=COLORS["mean"], linestyle="--",
               linewidth=LINES["reference"], alpha=0.5,
               label=f"{metric.name} threshold (P{config.HIDDEN_STEWARD_H_PERCENTILE})")

    # Shade hidden steward zone
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.fill_between([xlim[0], metric_threshold], s_threshold, ylim[1],
                    alpha=0.15, color=COLORS["highlight_zone"])

    ax.set_yscale('log')
    format_log_ticks(ax, axis='y')

    if metric.use_log_scale:
        ax.set_xscale('log')
        format_log_ticks(ax, axis='x')

    ax.set_xlabel(metric.name)
    ax.set_ylabel("S-index")
    title = f"S-index vs {metric.name}"
    if domain_label:
        title += f" ({domain_label})"
    ax.set_title(title)

    add_grid(ax, axis='both')
    ax.legend(loc="upper right", fontsize=FONT["size_annotation"])

    plt.tight_layout()

    if save:
        output_dir = ensure_output_dir(domain=domain)
        save_figure(fig, os.path.join(output_dir, f"s_vs_{metric.short_name}_quadrant"))

    return fig


def plot_quadrant_analysis_loglog(df, save=True, domain: Optional[Domain] = None):
    """
    Log-log quadrant analysis: S-index vs h-index with both axes on log scale.

    This complements the standard quadrant analysis by revealing the power-law
    structure of the S-index vs h-index relationship. Excludes h=0 and S=0
    researchers (zeros are undefined on log scale).

    Args:
        df: DataFrame with s_index and h_index columns
        save: Whether to save the figure
        domain: Domain to filter by (None = all researchers)
    """
    metric = H_INDEX_METRIC

    setup_style()

    df_filtered = filter_by_domain(df, domain)
    domain_label = get_domain_label(domain)

    metric_col = metric.column
    # Thresholds come from the full data (zeros included), as in the analysis
    valid_all, s_threshold, metric_threshold = _steward_plot_data(df_filtered, metric_col)

    # For log-log: exclude both S=0 and h=0 (undefined on log scale)
    valid = valid_all[(valid_all["s_index"] > 0) & (valid_all[metric_col] > 0)].copy()

    if len(valid) == 0:
        return None

    is_steward = (valid["s_index"] >= s_threshold) & (valid[metric_col] <= metric_threshold)

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_wide"])

    # Regular researchers
    regular = valid[~is_steward]
    ax.scatter(regular[metric_col], regular["s_index"],
               color=COLORS["regular_researcher"],
               **MARKERS["regular"], label="Researchers")

    # Hidden stewards
    steward_data = valid[is_steward]
    if len(steward_data) > 0:
        ax.scatter(steward_data[metric_col], steward_data["s_index"],
                   color=COLORS["hidden_steward"],
                   **MARKERS["highlight"], label="Hidden Stewards", zorder=5)

    # Threshold lines
    ax.axhline(s_threshold, color=COLORS["threshold"], linestyle="--",
               linewidth=LINES["threshold"], alpha=0.7,
               label=f"S threshold (P{config.HIDDEN_STEWARD_S_PERCENTILE})")
    if metric_threshold > 0:
        ax.axvline(metric_threshold, color=COLORS["mean"], linestyle="--",
                   linewidth=LINES["reference"], alpha=0.5,
                   label=f"h threshold (P{config.HIDDEN_STEWARD_H_PERCENTILE})")

    # Shade hidden steward zone
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if metric_threshold > 0:
        ax.fill_between([xlim[0], metric_threshold], s_threshold, ylim[1],
                        alpha=0.15, color=COLORS["highlight_zone"])

    ax.set_xscale('log')
    ax.set_yscale('log')
    format_log_ticks(ax, axis='both')

    ax.set_xlabel(f"{metric.name} (h > 0 only)")
    ax.set_ylabel("S-index (S > 0 only)")
    title = f"S-index vs {metric.name}"
    if domain_label:
        title += f" ({domain_label})"
    ax.set_title(title)

    add_grid(ax, axis='both')
    ax.legend(loc="upper right", fontsize=FONT["size_annotation"])

    plt.tight_layout()

    if save:
        output_dir = ensure_output_dir(domain=domain)
        save_figure(fig, os.path.join(output_dir, f"s_vs_{metric.short_name}_quadrant_loglog"))

    return fig


def plot_s_vs_metric_scatter(df, save=True, metric: Optional[ComparisonMetric] = None, domain: Optional[Domain] = None):
    """
    Scatter plot of S-index vs metric with LOWESS trend line.

    Shows researchers with S > 0. For h-index, includes h=0 points.
    For log-scale metrics (citations), excludes zeros.

    Args:
        df: DataFrame with s_index and metric columns
        save: Whether to save the figure
        metric: Comparison metric (default: h-index)
        domain: Domain to filter by (None = all researchers)
    """
    if metric is None:
        metric = H_INDEX_METRIC

    setup_style()

    df_filtered = filter_by_domain(df, domain)
    domain_label = get_domain_label(domain)

    metric_col = metric.column
    valid_all, s_threshold, metric_threshold = _steward_plot_data(df_filtered, metric_col)

    # For plotting: only show S > 0 researchers
    # For log-scale metrics (citations), also exclude metric = 0
    valid = valid_all[valid_all["s_index"] > 0].copy()
    if metric.use_log_scale:
        valid = valid[valid[metric_col] > 0]

    is_steward = (valid["s_index"] >= s_threshold) & (valid[metric_col] <= metric_threshold)

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_wide"])

    # Regular researchers
    if (~is_steward).sum() > 0:
        ax.scatter(valid.loc[~is_steward, metric_col], valid.loc[~is_steward, "s_index"],
                   color=COLORS["regular_researcher"],
                   **MARKERS["regular"], label="Researchers")

    # Hidden stewards
    if is_steward.sum() > 0:
        ax.scatter(valid.loc[is_steward, metric_col], valid.loc[is_steward, "s_index"],
                   color=COLORS["hidden_steward"],
                   **MARKERS["highlight"], label="Hidden Stewards", zorder=5)

    ax.set_yscale('log')
    format_log_ticks(ax, axis='y')

    if metric.use_log_scale:
        ax.set_xscale('log')
        format_log_ticks(ax, axis='x')
        # Reference line (S = metric)
        min_val = max(valid[metric_col].min(), valid["s_index"].min(), 1)
        max_val = min(valid[metric_col].max(), valid["s_index"].max())
        ref_range = np.logspace(np.log10(min_val), np.log10(max_val), 100)
        ax.plot(ref_range, ref_range, color=COLORS["reference_line"], linestyle='--',
                linewidth=LINES["reference"], alpha=0.5, label='S = citations')
    else:
        # Reference line (S = h)
        metric_range = np.linspace(max(1, valid[metric_col].min()), valid[metric_col].max(), 100)
        ax.plot(metric_range, metric_range, color=COLORS["reference_line"], linestyle='--',
                linewidth=LINES["reference"], alpha=0.5, label='S = h')

    # LOWESS trend lines (mean + median) using helpers
    _add_lowess_trend(ax, valid[metric_col].values, valid["s_index"].values, log_x=metric.use_log_scale)
    _add_lowess_median_trend(ax, valid[metric_col].values, valid["s_index"].values, log_x=metric.use_log_scale)

    # Clip h-index x-axis at 100 to focus on main data range
    if not metric.use_log_scale:
        ax.set_xlim(right=100)

    ax.set_xlabel(metric.name)
    ax.set_ylabel("S-index")
    title = f"S-index vs {metric.name}"
    if domain_label:
        title += f" ({domain_label})"
    ax.set_title(title)

    add_grid(ax, axis='both')
    ax.legend(loc="upper left", fontsize=FONT["size_annotation"])

    plt.tight_layout()

    if save:
        output_dir = ensure_output_dir(domain=domain)
        save_figure(fig, os.path.join(output_dir, f"s_vs_{metric.short_name}_scatter"))

    return fig


def plot_s_vs_num_datasets_scatter(df, save=True, domain: Optional[Domain] = None):
    """
    Scatter plot of S-index (y) vs num_datasets (x), both on log scales, with
    LOWESS trend lines. Shows researchers with S > 0 and num_datasets > 0.

    Args:
        df: DataFrame with s_index and num_datasets columns
        save: Whether to save the figure
        domain: Domain to filter by (None = all researchers)
    """
    setup_style()

    df_filtered = filter_by_domain(df, domain)
    domain_label = get_domain_label(domain)

    required_cols = ["s_index", "num_datasets"]
    if not all(c in df_filtered.columns for c in required_cols):
        return None
    valid = df_filtered[required_cols].dropna()
    valid = valid[(valid["s_index"] > 0) & (valid["num_datasets"] > 0)]
    if len(valid) == 0:
        return None
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_tall"])
    ax.scatter(valid["num_datasets"], valid["s_index"],
               color=COLORS["regular_researcher"], **MARKERS["regular"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    format_log_ticks(ax, axis="both")
    ax.set_xlabel("Number of datasets")
    ax.set_ylabel("S-index")
    title = "S-index vs Number of Datasets"
    if domain_label:
        title += f" ({domain_label})"
    ax.set_title(title)
    add_grid(ax, axis='both')
    # LOWESS trend lines (mean + median)
    _add_lowess_trend(ax, valid["num_datasets"].values.astype(float),
                      valid["s_index"].values.astype(float), log_x=True)
    _add_lowess_median_trend(ax, valid["num_datasets"].values.astype(float),
                             valid["s_index"].values.astype(float), log_x=True)
    ax.legend(loc="upper left", fontsize=FONT["size_annotation"])
    plt.tight_layout()
    if save:
        output_dir = ensure_output_dir(domain=domain)
        save_figure(fig, os.path.join(output_dir, "s_vs_num_datasets"))
    return fig


def _num_data_papers_series(df: pd.DataFrame):
    """Number of data papers per researcher (sum of the data journal columns)."""
    existing = [c for c in DATA_JOURNAL_COLUMNS if c in df.columns]
    if not existing:
        return None
    return df[existing].fillna(0).astype(float).sum(axis=1)


def plot_s_vs_num_data_papers_boxplot(df, save=True, domain: Optional[Domain] = None):
    """
    Boxplot of S-index distribution by number of data papers.
    Shows distribution, median, and outliers for each data paper count category.

    Args:
        df: DataFrame with s_index and data paper columns
        save: Whether to save the figure
        domain: Domain to filter by (None = all researchers)
    """
    setup_style()

    df_filtered = filter_by_domain(df, domain)
    domain_label = get_domain_label(domain)

    if "s_index" not in df_filtered.columns:
        return None
    num_papers = _num_data_papers_series(df_filtered)
    if num_papers is None:
        return None
    valid = df_filtered[["s_index"]].copy()
    valid["num_data_papers"] = num_papers
    valid = valid.dropna()
    valid = valid[valid["s_index"] > 0]
    if len(valid) == 0:
        return None

    # Create bins: 0, 1, 2-3, 4+
    def bin_papers(n):
        if n == 0:
            return "0"
        elif n == 1:
            return "1"
        elif n <= 3:
            return "2-3"
        else:
            return "4+"

    valid["papers_bin"] = valid["num_data_papers"].apply(bin_papers)
    bin_order = ["0", "1", "2-3", "4+"]

    # Prepare data for boxplot
    data_by_bin = [valid[valid["papers_bin"] == b]["s_index"].values for b in bin_order]
    # Filter out empty bins
    non_empty = [(b, d) for b, d in zip(bin_order, data_by_bin) if len(d) > 0]
    if len(non_empty) == 0:
        return None
    labels = [b for b, d in non_empty]
    data = [d for b, d in non_empty]

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_tall"])

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor=COLORS["mean"],
                                   markeredgecolor="white", markeredgewidth=0.5,
                                   markersize=5),
                    whiskerprops=dict(color="#666666", linewidth=LINES["box_whisker"]),
                    capprops=dict(color="#666666", linewidth=LINES["box_whisker"]),
                    flierprops=dict(marker='o', markersize=3,
                                    markerfacecolor=COLORS["primary"],
                                    markeredgecolor="white", markeredgewidth=0.3,
                                    alpha=0.5))

    for patch in bp['boxes']:
        patch.set_facecolor(COLORS["primary"])
        patch.set_alpha(0.65)
        patch.set_edgecolor("#555555")
        patch.set_linewidth(LINES["box_whisker"])
    for median in bp['medians']:
        median.set_color(COLORS["median"])
        median.set_linewidth(LINES["box_median"])

    ax.set_yscale("log")
    format_log_ticks(ax, axis="y")
    ax.set_xlabel("Number of data papers")
    ax.set_ylabel("S-index (among S > 0)")
    title = "S-index by Number of Data Papers"
    if domain_label:
        title += f" ({domain_label})"
    ax.set_title(title)
    add_grid(ax, axis='y')

    # Add sample sizes as annotations below boxes
    y_min = ax.get_ylim()[0]
    for i, (label, d) in enumerate(zip(labels, data)):
        ax.annotate(f"n={len(d)}", (i + 1, y_min * 1.1),
                    ha='center', fontsize=FONT["size_annotation"], color='#666666')

    legend_elements = [
        Line2D([0], [0], color=COLORS["median"], linewidth=LINES["box_median"], label='Median'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=COLORS["mean"],
               markersize=5, label='Mean'),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    if save:
        output_dir = ensure_output_dir(domain=domain)
        save_figure(fig, os.path.join(output_dir, "s_vs_data_papers"))
    return fig


def generate_all_plots(df, save=True, show=False, domain: Optional[Domain] = None):
    """
    Generate all validation plots for a specific domain or all researchers.

    Args:
        df: DataFrame with all researcher data
        save: Whether to save figures
        show: Whether to display figures
        domain: Domain to filter by (None = all researchers)

    Returns:
        List of the generated figures
    """
    domain_label = get_domain_label(domain)
    display_label = domain_label if domain_label else "All Researchers"
    print(f"Generating visualizations for {display_label}...")

    figs = []

    # General plots
    print(f"\n  General plots ({display_label}):")
    figs.append(plot_correlation_matrix(df, save=save, domain=domain))
    figs.append(plot_spearman_correlation_matrix(df, save=save, domain=domain))
    figs.append(plot_s_index_distribution(df, save=save, domain=domain))
    figs.append(plot_s_index_distribution_clipped(df, save=save, domain=domain))
    overlay_fig = plot_metric_distributions_overlay(df, save=save, domain=domain)
    if overlay_fig is not None:
        figs.append(overlay_fig)
    overlay_linear_fig = plot_metric_distributions_overlay_linear(df, save=save, domain=domain)
    if overlay_linear_fig is not None:
        figs.append(overlay_linear_fig)
    s_vs_datasets = plot_s_vs_num_datasets_scatter(df, save=save, domain=domain)
    if s_vs_datasets is not None:
        figs.append(s_vs_datasets)
    s_vs_data_papers = plot_s_vs_num_data_papers_boxplot(df, save=save, domain=domain)
    if s_vs_data_papers is not None:
        figs.append(s_vs_data_papers)

    # Metric-specific plots
    for metric in COMPARISON_METRICS:
        if metric.column not in df.columns:
            print(f"\n  Skipping {metric.name}: column '{metric.column}' not found")
            continue

        print(f"\n  Plots for S-index vs {metric.name} ({display_label}):")
        figs.append(plot_quadrant_analysis(df, save=save, metric=metric, domain=domain))
        figs.append(plot_s_vs_metric_scatter(df, save=save, metric=metric, domain=domain))

    # Log-log quadrant analysis for h-index
    if "h_index" in df.columns:
        loglog_fig = plot_quadrant_analysis_loglog(df, save=save, domain=domain)
        if loglog_fig is not None:
            figs.append(loglog_fig)

    if show:
        plt.show()
    else:
        plt.close("all")

    print(f"\nPlots for {display_label} generated.")
    return figs


def generate_all_domain_plots(df, save=True, show=False):
    """
    Generate all validation plots for all domains (including overall).

    Args:
        df: DataFrame with all researcher data
        save: Whether to save figures
        show: Whether to display figures
    """
    all_figs = []

    for domain, label in get_all_domain_configs():
        print(f"\n{'='*60}")
        print(f"Generating plots for: {label}")
        print(f"{'='*60}")
        figs = generate_all_plots(df, save=save, show=False, domain=domain)
        all_figs.extend(figs)

    if show:
        plt.show()
    else:
        plt.close("all")

    print(f"\nAll domain plots generated ({len(all_figs)} total figures).")
    return all_figs
