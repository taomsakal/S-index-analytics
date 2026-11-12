"""
Data loading and cleaning utilities for S-Index validation.

Pipeline: data/ -> resampled_data.csv -> cleaned_data.csv
"""

from pathlib import Path
import re

import pandas as pd


def load_raw_data(source: str) -> pd.DataFrame:
    """
    Load raw data from a file or directory of CSVs.

    If source is a directory: glob all *.csv, load each with load_openalex_data,
    concatenate with stable sort order for reproducibility.
    If source is a file: load with load_openalex_data.

    Args:
        source: Path to a single CSV file or directory containing CSV files.

    Returns:
        Combined DataFrame with standardized column names.

    Raises:
        ValueError: If source is a directory with no CSV files.
    """
    path = Path(source)
    if path.is_dir():
        csv_files = sorted(path.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in directory: {source}")
        frames = [load_openalex_data(str(f)) for f in csv_files]
        return pd.concat(frames, ignore_index=True)
    return load_openalex_data(source)


def load_openalex_data(filepath: str) -> pd.DataFrame:
    """
    Load OpenAlex researcher data from CSV.

    Expected columns: openalex_id, name, s-index, h-index, cited_by_count, num_datasets
    """
    df = pd.read_csv(filepath)

    # Standardize column names (handle hyphenated names)
    df.columns = df.columns.str.strip().str.lower().str.replace("-", "_")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset:
    - Convert numeric columns to appropriate types
    - Handle missing values
    """
    df = df.copy()  # Avoid modifying original dataframe
    numeric_cols = [
        "s_index", "h_index", "cited_by_count", "num_datasets",
        "team_size",
        # Data paper journal columns (for known-groups validation)
        "scientific_data_pubs", "essd_pubs", "gigascience_pubs", "bds_pubs",
        # Domain columns (for known-groups validation)
        "life_sciences", "social_sciences", "physical_sciences", "health_sciences",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_and_prepare(filepath: str) -> pd.DataFrame:
    """Load a CSV and coerce its numeric columns."""
    return clean_data(load_openalex_data(filepath))


# =============================================================================
# Pre-Resampling Filters
# =============================================================================

# To stay within API rate limits, the collection backend skips authors with
# more than 5,000 cited datasets and records their S-index as 0. Those zeros
# are placeholders, not measurements. They are identified as records with at
# least this many datasets and S = 0, and excluded.
PLACEHOLDER_ZERO_MIN_DATASETS = 5000


def filter_for_resampling(
    df: pd.DataFrame,
    s_index_col: str = "s_index",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Filter raw data before resampling and ESS calculation.

    Removes:
    1. Records with missing S-index values
    2. Records with num_datasets >= PLACEHOLDER_ZERO_MIN_DATASETS and
       s_index == 0 (a placeholder zero: the S-index was not computed)

    Args:
        df: Raw DataFrame
        s_index_col: Column name for S-index
        verbose: Whether to print filtering statistics

    Returns:
        Filtered DataFrame
    """
    df = df.copy()
    original_size = len(df)

    # Ensure numeric types for key columns
    if "num_datasets" in df.columns:
        df["num_datasets"] = pd.to_numeric(df["num_datasets"], errors="coerce")
    if s_index_col in df.columns:
        df[s_index_col] = pd.to_numeric(df[s_index_col], errors="coerce")
    if "team_size" in df.columns:
        df["team_size"] = pd.to_numeric(df["team_size"], errors="coerce")

    # Filter 1: Remove records with missing S-index
    if s_index_col in df.columns:
        na_s_mask = df[s_index_col].isna()
        n_na_s = na_s_mask.sum()
        df = df[~na_s_mask]
    else:
        n_na_s = 0

    # Filter 2: Remove placeholder S-index zeros (S-index not computed)
    n_placeholder_zero = 0
    if "num_datasets" in df.columns and s_index_col in df.columns:
        placeholder_zero_mask = (
            (df["num_datasets"] >= PLACEHOLDER_ZERO_MIN_DATASETS)
            & (df[s_index_col] == 0)
        )
        n_placeholder_zero = placeholder_zero_mask.sum()
        df = df[~placeholder_zero_mask]

    if verbose:
        print("  Pre-resampling filters applied:")
        print(f"    Excluded {n_na_s} records with missing S-index")
        print(f"    Excluded {n_placeholder_zero} records with placeholder S=0 "
              f"(num_datasets >= {PLACEHOLDER_ZERO_MIN_DATASETS})")
        print(f"    Remaining: {len(df)} of {original_size} ({len(df)/original_size*100:.1f}%)")

    return df.reset_index(drop=True)


# =============================================================================
# Data Cleaning Pipeline (after resampling)
# =============================================================================

# Keywords indicating an organization rather than an individual
ORGANIZATION_KEYWORDS = [
    # Institutional types
    'Institute', 'University', 'Museum', 'Hospital', 'Center', 'Centre',
    'Laboratory', 'Foundation', 'Society', 'Association',
    'Council', 'Agency', 'Department', 'Ministry', 'Organization',
    'Organisation', 'Committee', 'Board', 'Academy', 'College',
    'Administration', 'Veterinary', 'Secretariat', 'Republic',
    'School', 'Library', 'Archive', 'Network',
    # Research collaborations/teams
    'Collaboration', 'Consortium', 'Working Group', 'Task Force',
    'Group', 'Team', 'Project', 'Initiative', 'Program', 'Programme',
    # Software/data collective entities
    'Developers', 'Contributors', 'Authors', 'Curators', 'Staff', 'Admin',
    # Scientific collaborations (particle physics, astronomy, etc.)
    'ATLAS', 'CMS', 'LHCb', 'CERN', 'SXS', 'LIGO', 'VIRGO',
    # Environmental/ecological networks
    'NEON', 'Observatory', 'Survey', 'Division',
    # Data services
    'Service', 'Services', 'Trust', 'Data Center', 'Data Centre',
    # Publishing/commercial entities
    'Journal', 'Journals', 'Press', 'Publishing', 'Publisher', 'Publishers',
    'Company', 'Companies', 'Inc', 'Ltd', 'LLC', 'GmbH', 'PLC',
    'Corp', 'Corporation',
]

# Org keywords that are also common personal names: match case-sensitively only
# (e.g. "ALICE" = collaboration, "Alice" = person)
ORG_KEYWORDS_CASE_SENSITIVE = [
    'ALICE',  # particle physics; "Alice" is a common first name
]

# Substring keywords for no-space names (handles concatenated org names)
ORG_SUBSTRING_KEYWORDS = [
    'museum', 'archive', 'heritage', 'library', 'university', 'college',
    'institute', 'hospital', 'clinic', 'society', 'association',
    'foundation', 'center', 'centre', 'laborator', 'lab', 'academy',
    'school', 'department', 'ministry', 'agency', 'journal', 'press',
    'publisher', 'publishing', 'trust', 'observatory', 'survey',
    'consortium', 'collaboration', 'network', 'council',
    # Multilingual/variant org terms (common in no-space handles)
    'museo', 'museu', 'musee', 'muzea', 'muzeum',
    'ministerio', 'ministerium', 'ministere',
    'universidad', 'universite', 'universita', 'universiteit',
    'instituto', 'institut', 'biblioteca', 'bibliotheca', 'archiv',
    'wiki', 'armoury', 'libraries', 'studio', 'classroom',
]

# Known placeholder or invalid names
PLACEHOLDER_NAMES = [
    'Unav', 'Unknown', 'Anonymous', 'N/A', 'NA', 'None', 'Null',
    'Test', 'Admin', 'User', 'Author', 'Authors', 'of', 'Blank', 'Anonymity',
    # Known organizations/software that don't match keyword patterns
    'Plazi',  # Biodiversity informatics organization
    'Gbm Developers',  # R package maintainers
    'PheKnowLator Ecosystem Developers',  # Software maintainers
    'Bionomia',  # Biodiversity platform
    'ProQuest',  # Publishing/database company
    'WikiPathways',  # Community curation platform
    'MIT Election Data And Science Lab',
    'ALR Lab',
    'Eep Escuela de Estudios de Postgrado',
    'MinisteriodeCulturaPeru',
    'WirtualneMuzeaMalopolski',
    'MuseoVirtualCarnaval',
    'Varldskulturmuseerna',
    'Urheilumuseo',
    'MuseuEmporda',
    'TheRoyalArmoury',
    'TransLink',
    'ScottishRockArt',
    'Crawler.Gbif.Org',  # GBIF automated web crawler, not a person
    'Registry-Migration.Gbif.Org',  # GBIF automated migration service, not a person
    'Phr Iosr',  # Likely 'IOSR Journals' (organization), garbled name
    'themoviedb.org',  # Website, not a person
    'VesselFinder.Com',  # Website, not a person
    'SAC Administrator',  # Administrative role, not a person
    'MeteoTrentino',  # Meteorological service of Trentino, Italy
]

# Maximum plausible num_datasets for an individual researcher
# Particle physics collaborations can have many, but individual researchers shouldn't
MAX_PLAUSIBLE_DATASETS = 10000

# =============================================================================
# Manual Exclusions
# =============================================================================
# OpenAlex profiles excluded after manual review (merged author profiles,
# non-individual profiles, and metric artifacts). Each entry in the ledger
# records the ID, category, reason, evidence, and verification date.
KNOWN_EXCLUSIONS_FILE = Path(__file__).with_name("known_exclusions.csv")


def load_known_exclusions(path: Path = KNOWN_EXCLUSIONS_FILE) -> pd.DataFrame:
    """Load and validate the manual OpenAlex exclusion ledger."""
    exclusions = pd.read_csv(path, dtype=str).fillna("")
    required = {
        "openalex_id", "name", "category", "reason", "evidence", "verified_at"
    }
    missing = required.difference(exclusions.columns)
    if missing:
        raise ValueError(
            f"Manual exclusion ledger is missing columns: {sorted(missing)}"
        )
    if exclusions["openalex_id"].duplicated().any():
        duplicates = exclusions.loc[
            exclusions["openalex_id"].duplicated(), "openalex_id"
        ].tolist()
        raise ValueError(f"Duplicate OpenAlex IDs in exclusion ledger: {duplicates}")
    return exclusions


KNOWN_MANUAL_EXCLUSIONS = load_known_exclusions()


def filter_organizations(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Remove organizational records (not individual researchers).

    Filters out:
    - Names containing organization keywords (Institute, University, etc.)
    - Known placeholder names (Unav, Unknown, etc.)
    - Names that look like organizational patterns

    Should be applied AFTER resampling to preserve correct sampling weights.
    """
    df = df.copy()
    n_before = len(df)

    if 'name' not in df.columns:
        if verbose:
            print("Warning: 'name' column not found, skipping organization filter")
        return df

    # Case-sensitive org keywords (e.g. ALICE collaboration vs Alice the person)
    org_pattern_case_sensitive = r'\b(?:' + '|'.join(re.escape(kw) for kw in ORG_KEYWORDS_CASE_SENSITIVE) + r')\b'
    org_mask_sensitive = df['name'].str.contains(org_pattern_case_sensitive, case=True, na=False, regex=True)

    # Remaining org keywords (case-insensitive)
    org_pattern = r'\b(?:' + '|'.join(re.escape(kw) for kw in ORGANIZATION_KEYWORDS) + r')\b'
    org_mask_insensitive = df['name'].str.contains(org_pattern, case=False, na=False, regex=True)

    org_mask = org_mask_sensitive | org_mask_insensitive

    # Build pattern for placeholder names (exact match or contains, case-insensitive)
    placeholder_lower = [p.lower() for p in PLACEHOLDER_NAMES]
    placeholder_mask = df['name'].str.strip().str.lower().isin(placeholder_lower)
    # Also catch partial matches for known problematic patterns
    for placeholder in PLACEHOLDER_NAMES:
        if len(placeholder) > 5:  # Only for longer, specific names
            placeholder_mask |= df['name'].str.contains(re.escape(placeholder), case=False, na=False)

    # Names with "of/for/the" followed by an institution type
    institutional_pattern = r'\b(?:of|for|the)\s+(?:the\s+)?(?:University|Institute|Museum|Department|Center|Centre|Academy)\b'
    institutional_mask = df['name'].str.contains(institutional_pattern, case=False, na=False, regex=True)

    # Detect concatenated org names (no spaces but containing org substrings)
    no_space_mask = ~df['name'].str.contains(r'\s', na=False)
    org_sub_pattern = r'(?:' + '|'.join(re.escape(kw) for kw in ORG_SUBSTRING_KEYWORDS) + r')'
    org_sub_mask = no_space_mask & df['name'].str.contains(org_sub_pattern, case=False, na=False, regex=True)

    # Detect acronym-only names (likely organizations)
    acronym_mask = df['name'].str.fullmatch(r'[A-Z]{4,}', na=False)

    # Detect email-like names (including space-separated like 'ztimrei gmail.com')
    email_mask = df['name'].str.contains(r'@|gmail|yahoo|hotmail|outlook', case=False, na=False)

    # Detect 'et al' parsing errors (OpenAlex name fragments)
    et_al_mask = df['name'].str.contains(r'\bet\s*al\.?\b', case=False, na=False, regex=True)
    # Also catch names that are exactly 'et'
    et_exact_mask = df['name'].str.strip().str.lower() == 'et'

    # Combined filter
    filter_mask = (
        org_mask
        | placeholder_mask
        | institutional_mask
        | org_sub_mask
        | acronym_mask
        | email_mask
        | et_al_mask
        | et_exact_mask
    )
    df_filtered = df[~filter_mask]

    n_removed = n_before - len(df_filtered)
    if verbose and n_removed > 0:
        print(f"Filtered {n_removed} organizations/placeholders ({n_removed/n_before*100:.1f}%)")

    return df_filtered


def filter_implausible_records(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Remove records with implausible values that indicate data quality issues.

    Filters:
    - num_datasets > MAX_PLAUSIBLE_DATASETS (likely aggregated/institutional records)
    - Manually verified OpenAlex exclusions (see known_exclusions.csv)
    """
    df = df.copy()
    n_before = len(df)

    implausible_mask = pd.Series(False, index=df.index)

    if 'num_datasets' in df.columns:
        high_datasets = df['num_datasets'] > MAX_PLAUSIBLE_DATASETS
        implausible_mask |= high_datasets
        if verbose and high_datasets.sum() > 0:
            print(f"  - High num_datasets (>{MAX_PLAUSIBLE_DATASETS}): {high_datasets.sum()}")

    # Filter manually verified OpenAlex exclusions.
    if 'openalex_id' in df.columns and not KNOWN_MANUAL_EXCLUSIONS.empty:
        exclusion_ids = KNOWN_MANUAL_EXCLUSIONS['openalex_id'].tolist()
        exclusion_mask = df['openalex_id'].isin(exclusion_ids)
        implausible_mask |= exclusion_mask
        if verbose and exclusion_mask.sum() > 0:
            present = KNOWN_MANUAL_EXCLUSIONS[
                KNOWN_MANUAL_EXCLUSIONS['openalex_id'].isin(
                    df.loc[exclusion_mask, 'openalex_id']
                )
            ]
            for row in present.itertuples(index=False):
                observed_name = (
                    df.loc[df['openalex_id'] == row.openalex_id, 'name'].iloc[0]
                    if 'name' in df.columns else row.name
                )
                print(
                    f"  - Excluded {observed_name} [{row.category}]: {row.reason}"
                )

    df_filtered = df[~implausible_mask]

    n_removed = n_before - len(df_filtered)
    if verbose and n_removed > 0:
        print(f"Filtered {n_removed} implausible records ({n_removed/n_before*100:.1f}%)")

    return df_filtered


def deduplicate_authors(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Deduplicate authors who appear multiple times with the same OpenAlex ID.

    Strategy: Keep the most complete record (most non-null fields), breaking
    ties by highest h-index. This avoids biasing the S-index (the metric under
    validation) upward, since we do not select on S-index itself.

    Deduplication is by OpenAlex ID only - NOT by name, since different people
    can share the same name (especially common in Chinese, Korean, etc.).

    Should be applied AFTER resampling to preserve correct sampling weights.
    """
    df = df.copy()
    n_before = len(df)

    if 'openalex_id' not in df.columns:
        if verbose:
            print("Warning: 'openalex_id' column not found, skipping deduplication")
        return df

    # Count non-null fields per row, then sort by completeness (desc) then
    # h_index (desc) so tie-breaking never depends on the outcome variable.
    df['_n_valid'] = df.notna().sum(axis=1)
    h_col = 'h_index' if 'h_index' in df.columns else '_n_valid'
    df = df.sort_values(['_n_valid', h_col], ascending=[False, False], na_position='last')

    # Keep first occurrence (most complete / highest h-index) for each OpenAlex ID
    df_deduped = df.drop_duplicates(subset='openalex_id', keep='first')
    df_deduped = df_deduped.drop(columns=['_n_valid'])

    n_removed = n_before - len(df_deduped)
    if verbose and n_removed > 0:
        print(f"Deduplicated {n_removed} duplicate OpenAlex IDs ({n_removed/n_before*100:.1f}%)")

    return df_deduped


def normalize_name(name: str) -> str:
    """
    Normalize a name for matching name variants.

    Lowercases, keeps only ASCII letters, digits, and spaces, collapses
    whitespace, and sorts the name parts so that "John Smith" and
    "Smith, John" produce the same key.

    Names with no ASCII letters or digits (for example, names written
    entirely in Cyrillic) normalize to an empty string.
    """
    if pd.isna(name) or not isinstance(name, str):
        return ""

    normalized = re.sub(r'[^a-z0-9\s]', '', name.lower())
    return ' '.join(sorted(normalized.split()))


def deduplicate_fuzzy_names(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Deduplicate authors who share a normalized name but have different
    OpenAlex IDs (e.g. "Tokihiko TOKUZAWA" vs "TOKUZAWA, Tokihiko").

    Within each group of matching names, the record with the highest h-index
    is kept as the most established profile.

    Limitations:
    - Different people who share a name are merged into one record. This is
      accepted in exchange for not counting split profiles twice.
    - Records whose name normalizes to an empty string (see normalize_name)
      cannot be compared and are excluded. The count is reported.
    """
    df = df.copy()
    n_before = len(df)

    if 'name' not in df.columns:
        if verbose:
            print("Warning: 'name' column not found, skipping fuzzy deduplication")
        return df

    df['_normalized_name'] = df['name'].apply(normalize_name)

    has_key = df['_normalized_name'] != ''
    n_no_key = int((~has_key).sum())
    df = df[has_key]

    sort_col = 'h_index' if 'h_index' in df.columns else 's_index'
    df = df.sort_values(sort_col, ascending=False, na_position='last')
    df_deduped = df.drop_duplicates(subset='_normalized_name', keep='first')
    df_deduped = df_deduped.drop(columns=['_normalized_name'])

    if verbose:
        if n_no_key > 0:
            print(f"Excluded {n_no_key} records with no ASCII letters or digits "
                  f"in the name ({n_no_key/n_before*100:.1f}%)")
        n_merged = n_before - n_no_key - len(df_deduped)
        if n_merged > 0:
            print(f"Fuzzy-deduplicated {n_merged} similar names ({n_merged/n_before*100:.1f}%)")

    return df_deduped


def postprocess_resampled_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Apply all post-resampling preprocessing steps.

    Order matters:
    1. Filter organizations (before dedup to avoid merging org with person)
    2. Filter implausible records
    3. Deduplicate by OpenAlex ID
    4. Fuzzy deduplicate by normalized name

    Should be called AFTER resampling, not before.
    """
    if verbose:
        print("\nPost-resampling preprocessing:")

    df = filter_organizations(df, verbose=verbose)
    df = filter_implausible_records(df, verbose=verbose)
    df = deduplicate_authors(df, verbose=verbose)
    df = deduplicate_fuzzy_names(df, verbose=verbose)

    if verbose:
        print(f"Final dataset: {len(df)} researchers")

    return df


def save_cleaned_data(df: pd.DataFrame, output_path: str, verbose: bool = True) -> None:
    """Save cleaned data to CSV."""
    df.to_csv(output_path, index=False)
    if verbose:
        print(f"Saved cleaned data to: {output_path}")
