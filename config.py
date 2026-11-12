"""
Configuration constants for S-Index validation analyses.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd

# =============================================================================
# Comparison Metrics
# =============================================================================
# We compare S-index against both h-index and citation count to see if
# conclusions are robust across different traditional metrics.

@dataclass(frozen=True)
class ComparisonMetric:
    """Configuration for a comparison metric (h-index or citation count)."""
    name: str           # Display name (e.g., "h-index")
    column: str         # DataFrame column name (e.g., "h_index")
    short_name: str     # Short name for file paths (e.g., "h")
    use_log_scale: bool # Whether to use log scale for this metric in plots

    def __str__(self):
        return self.name

H_INDEX_METRIC = ComparisonMetric(
    name="h-index",
    column="h_index",
    short_name="h",
    use_log_scale=False,  # h-index typically linear (0-100 range)
)

CITATION_COUNT_METRIC = ComparisonMetric(
    name="citation count",
    column="cited_by_count",
    short_name="citations",
    use_log_scale=True,   # Citations span many orders of magnitude
)

COMPARISON_METRICS = [H_INDEX_METRIC, CITATION_COUNT_METRIC]

# Correlation thresholds for divergent validity
# Moderate correlation (0.3-0.6) indicates related but distinct constructs
CORRELATION_MODERATE_LOW = 0.3
CORRELATION_MODERATE_HIGH = 0.6

# Hidden steward detection (percentile-based cutoffs)
HIDDEN_STEWARD_S_PERCENTILE = 75  # Top quartile S-index
HIDDEN_STEWARD_H_PERCENTILE = 50  # Bottom half of the comparison metric

# Statistical significance threshold
ALPHA = 0.05

# Seed for resampling, domain tie-breaking, and control draws
RANDOM_SEED = 42

# Pipeline files (paths are relative to the repository root)
RAW_DATA_SOURCE = "data"  # Input: raw sampled data (file or directory of CSVs)
RESAMPLED_DATA_FILE = "resampled_data.csv"  # Intermediate: bias-corrected data
CLEANED_DATA_FILE = "cleaned_data.csv"  # Final: cleaned and deduplicated data

# Minimum matched pairs for a per-domain Wilcoxon test
MIN_PAIRS_FOR_DOMAIN_TEST = 15

# Output settings
OUTPUT_DIR = "output"

# Structured output subdirectories
FIGURES_DIR = f"{OUTPUT_DIR}/figures"
TABLES_DIR = f"{OUTPUT_DIR}/tables"
DATA_DIR = f"{OUTPUT_DIR}/data"

# =============================================================================
# Domain Configuration
# =============================================================================

@dataclass(frozen=True)
class Domain:
    """Configuration for a scientific domain."""
    name: str           # Display name (e.g., "Life Sciences")
    column: str         # DataFrame column name (e.g., "life_sciences")
    short_name: str     # Short name for file paths (e.g., "life")

    def __str__(self):
        return self.name

LIFE_SCIENCES = Domain(name="Life Sciences", column="life_sciences", short_name="life")
HEALTH_SCIENCES = Domain(name="Health Sciences", column="health_sciences", short_name="health")
PHYSICAL_SCIENCES = Domain(name="Physical Sciences", column="physical_sciences", short_name="physical")
SOCIAL_SCIENCES = Domain(name="Social Sciences", column="social_sciences", short_name="social")

ALL_DOMAINS = [LIFE_SCIENCES, HEALTH_SCIENCES, PHYSICAL_SCIENCES, SOCIAL_SCIENCES]


def filter_by_domain(df: pd.DataFrame, domain: Optional[Domain] = None) -> pd.DataFrame:
    """
    Filter dataframe to researchers whose primary domain matches.

    Each researcher is assigned exactly one primary domain (the domain with
    the highest publication count). This ensures non-overlapping domain groups.
    If domain is None, returns all researchers.

    Requires the 'primary_domain' column to exist (derived by
    derive_primary_domain() in known_groups_validation.py).

    Args:
        df: DataFrame with 'primary_domain' column
        domain: Domain to filter by, or None for all researchers

    Returns:
        Filtered DataFrame
    """
    if domain is None:
        return df

    if "primary_domain" not in df.columns:
        raise KeyError(
            "Column 'primary_domain' not found. "
            "Call derive_primary_domain() before filtering by domain."
        )

    return df[df["primary_domain"] == domain.name].copy()


# =============================================================================
# Domain-Aware Output Paths
# =============================================================================

def get_figures_dir(domain: Optional[Domain] = None) -> str:
    """
    Get figures directory for a domain.

    Examples: output/figures/all/, output/figures/life/
    """
    domain_folder = domain.short_name if domain else "all"
    return f"{FIGURES_DIR}/{domain_folder}"


def get_known_groups_figures_dir() -> str:
    """Get figures directory for known-groups validation: output/figures/known_groups/"""
    return f"{FIGURES_DIR}/known_groups"


def get_data_dir(domain: Optional[Domain] = None) -> str:
    """
    Get data export directory for a domain.

    Examples: output/data/all/, output/data/life/
    """
    domain_folder = domain.short_name if domain else "all"
    return f"{DATA_DIR}/{domain_folder}"


def get_known_groups_data_dir() -> str:
    """Get data directory for known-groups exports: output/data/known_groups/"""
    return f"{DATA_DIR}/known_groups"


def get_tables_dir() -> str:
    """Get supplementary tables directory: output/tables/"""
    return TABLES_DIR


def get_domain_label(domain: Optional[Domain] = None) -> str:
    """Get display label for a domain (or empty string if None/all)."""
    return domain.name if domain else ""


def get_all_domain_configs() -> list:
    """
    Get list of domain configurations to iterate over.

    Returns list of (Domain or None, label) tuples.
    None represents "all domains" analysis.
    """
    configs = [(None, "All Domains")]  # Overall first
    for domain in ALL_DOMAINS:
        configs.append((domain, domain.name))
    return configs
