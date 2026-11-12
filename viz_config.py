"""
Figure styling: colors, fonts, sizes, and matplotlib defaults.

Figures are sized for journal column widths and use a colorblind-safe palette.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap

# =============================================================================
# Color Palette (Refined Colorblind-Safe Scheme)
# =============================================================================
# A curated palette balancing visual richness with accessibility.
# Tested against deuteranopia, protanopia, and tritanopia simulations.

COLORS = {
    # Data categories
    "primary": "#4C72B0",           # Slate blue - regular researchers, primary data
    "highlight": "#F5811F",         # Bright orange - hidden stewards, emphasized data
    "secondary": "#937860",         # Warm taupe - matched controls, secondary data
    "positive": "#55A868",          # Sage green - positive differences
    "negative": "#C44E52",          # Muted red - thresholds, warnings

    # Semantic assignments (for consistency across all plots)
    "regular_researcher": "#4C72B0",
    "hidden_steward": "#F5811F",
    "data_paper_author": "#4C72B0",
    "matched_control": "#937860",

    # Statistical markers - CONSISTENT across all plots
    "median": "#E03530",            # Bright red - ALWAYS for medians
    "mean": "#7A7A7A",              # Warm grey - ALWAYS for means
    "threshold": "#C44E52",         # Muted red - thresholds/cutoffs
    "reference_line": "#8C8C8C",    # Grey - y=x, baselines
    "trend_line": "#C44E52",        # Muted red - LOWESS/trend lines

    # Zones/fills (use with low alpha)
    "highlight_zone": "#F5811F",    # Bright orange zone
    "threshold_zone": "#C44E52",    # Red zone

    # Box plot specific
    "box_primary": "#4C72B0",
    "box_secondary": "#937860",
    "box_median": "#E03530",

    # Histogram
    "hist_primary": "#4C72B0",
    "hist_secondary": "#55A868",
}

# Correlation heatmap colormap: white (0) to green (1)
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "white_to_green", ["#FFFFFF", "#55A868"]
)

# =============================================================================
# Typography (Nature Guidelines)
# =============================================================================

FONT = {
    "family": "Arial",              # Sans-serif, Nature-compatible
    "size_title": 10,               # Title (will be ~7pt at 70% reduction)
    "size_label": 9,                # Axis labels
    "size_tick": 8,                 # Tick labels
    "size_legend": 7.5,             # Legend text (slightly smaller for elegance)
    "size_annotation": 7,           # Data labels, annotations
    "weight_title": "bold",         # Bold titles
    "weight_label": "normal",       # Normal axis labels
}

# =============================================================================
# Figure Sizes (Designed for Journal Column Widths)
# =============================================================================

# Single column: ~3.5 inches (89mm)
# Double column: ~7.0 inches (178mm)
# Maximum height: ~9.0 inches (230mm)

FIGURE_SIZES = {
    "single": (3.5, 2.8),           # Standard single-column
    "single_tall": (3.5, 4.0),      # Taller single-column
    "double": (7.0, 3.5),           # Wide/dual-panel
    "double_tall": (7.0, 5.0),      # Large multi-panel
    "single_wide": (5.25, 4.0),     # 50% wider single-column (e.g., S vs h-index)
    "square": (4.0, 3.5),           # Correlation matrix (wider for colorbar)
}

# =============================================================================
# Line Weights (Optimized for Small-Size Visibility)
# =============================================================================

LINES = {
    "axis": 0.7,                    # Axis spines (slightly lighter)
    "grid": 0.4,                    # Grid lines (subtle)
    "data": 1.2,                    # Data lines
    "reference": 0.9,               # Reference lines (dashed)
    "threshold": 1.3,               # Threshold lines (emphasized but not heavy)
    "trend": 1.5,                   # LOWESS/trend lines
    "box_median": 1.8,              # Boxplot median (slightly thicker for visibility)
    "box_whisker": 0.7,             # Boxplot whiskers
    "hist_edge": 0.3,               # Histogram bar edges (subtler)
}

# =============================================================================
# Markers (Sized for Small-Figure Visibility)
# =============================================================================

MARKERS = {
    "regular": {
        "s": 18,                    # Size (area in points^2)
        "alpha": 0.65,
        "linewidth": 0.3,
        "edgecolor": "white",       # Subtle white halo for depth
    },
    "highlight": {
        "s": 35,                    # Larger for emphasis
        "alpha": 0.92,
        "linewidth": 0.5,
        "edgecolor": "white",
    },
    "dense": {
        "s": 10,                    # For very dense plots
        "alpha": 0.55,
        "linewidth": 0.2,
        "edgecolor": "white",
    },
}

# =============================================================================
# Output Settings
# =============================================================================

OUTPUT = {
    "dpi": 300,                     # Print quality
    "format": "png",                # PNG only
    "bbox_inches": "tight",
    "pad_inches": 0.08,             # Slightly more padding for breathing room
}

# =============================================================================
# Legend Style
# =============================================================================

LEGEND = {
    "frameon": True,
    "framealpha": 0.85,
    "edgecolor": "#E0E0E0",         # Very subtle border
    "fancybox": True,               # Rounded corners
    "borderpad": 0.6,
    "handlelength": 1.5,
    "handletextpad": 0.6,
}

# =============================================================================
# Grid Style (subtle reading aid)
# =============================================================================

GRID = {
    "alpha": 0.20,
    "linestyle": "-",
    "linewidth": 0.35,
    "color": "#B0B0B0",
}

# =============================================================================
# Style Setup Function
# =============================================================================

def setup_style():
    """Reset matplotlib to its defaults, then apply the project style."""
    plt.rcdefaults()

    # Font settings
    plt.rcParams.update({
        "font.family": FONT["family"],
        "font.size": FONT["size_tick"],
        "axes.titlesize": FONT["size_title"],
        "axes.titleweight": FONT["weight_title"],
        "axes.labelsize": FONT["size_label"],
        "axes.labelweight": FONT["weight_label"],
        "xtick.labelsize": FONT["size_tick"],
        "ytick.labelsize": FONT["size_tick"],
        "legend.fontsize": FONT["size_legend"],
    })

    # Line and spine settings
    plt.rcParams.update({
        "axes.linewidth": LINES["axis"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,          # No grid by default (enabled per-plot)
        "grid.linewidth": GRID["linewidth"],
        "grid.alpha": GRID["alpha"],
        "grid.color": GRID["color"],
        "grid.linestyle": GRID["linestyle"],
        "lines.linewidth": LINES["data"],
    })

    # Legend settings
    plt.rcParams.update({
        "legend.frameon": LEGEND["frameon"],
        "legend.framealpha": LEGEND["framealpha"],
        "legend.edgecolor": LEGEND["edgecolor"],
        "legend.fancybox": LEGEND["fancybox"],
        "legend.borderpad": LEGEND["borderpad"],
        "legend.handlelength": LEGEND["handlelength"],
        "legend.handletextpad": LEGEND["handletextpad"],
    })

    # Tick styling
    plt.rcParams.update({
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })

    # Embed TrueType fonts in PDF/PS output
    plt.rcParams.update({
        "pdf.fonttype": 42,          # TrueType fonts
        "ps.fonttype": 42,
    })

    # Figure settings
    plt.rcParams.update({
        "figure.dpi": OUTPUT["dpi"],
        "figure.facecolor": "white",
        "savefig.dpi": OUTPUT["dpi"],
        "savefig.bbox": OUTPUT["bbox_inches"],
        "savefig.pad_inches": OUTPUT["pad_inches"],
        "savefig.facecolor": "white",
    })


def format_log_ticks(ax, axis='both'):
    """
    Format log scale ticks to show integers (1, 10, 100) instead of powers (10^0, 10^1).

    Args:
        ax: Matplotlib axis
        axis: 'x', 'y', or 'both'
    """
    formatter = ticker.FuncFormatter(lambda x, p: f'{int(x):,}' if x >= 1 else f'{x:.1g}')

    if axis in ('x', 'both'):
        ax.xaxis.set_major_formatter(formatter)
    if axis in ('y', 'both'):
        ax.yaxis.set_major_formatter(formatter)


def add_grid(ax, axis='y'):
    """
    Add subtle grid lines as a reading aid.

    Args:
        ax: Matplotlib axis
        axis: 'x', 'y', or 'both'
    """
    ax.grid(True, axis=axis, alpha=GRID["alpha"], linestyle=GRID["linestyle"],
            linewidth=GRID["linewidth"], color=GRID["color"])
    ax.set_axisbelow(True)


def save_figure(fig, filepath_base: str):
    """
    Save figure in PNG format.

    Args:
        fig: Matplotlib figure
        filepath_base: Path without extension (e.g., "output/my_plot")
    """
    png_path = f"{filepath_base}.png"

    fig.savefig(png_path, format="png", dpi=OUTPUT["dpi"],
                bbox_inches=OUTPUT["bbox_inches"], pad_inches=OUTPUT["pad_inches"])

    print(f"  Saved: {png_path}")
