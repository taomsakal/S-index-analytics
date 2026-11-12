"""
Tests for visualization functions.

Verifies both that plots are generated and that they contain correct content.
"""

import os
import importlib
import pandas as pd
import pytest

# Set backend before importing matplotlib
os.environ["MPLBACKEND"] = "Agg"


@pytest.fixture
def sample_viz_df():
    """Sample dataframe for visualization testing with clear patterns."""
    return pd.DataFrame({
        "openalex_id": ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"],
        "name": ["Alice", "Bob", "Carol", "Dan", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack"],
        "s_index": [5, 7, 9, 6, 8, 50, 45, 3, 4, 2],  # Frank and Grace are high-S
        "h_index": [10, 15, 20, 12, 18, 5, 8, 25, 22, 30],  # Frank and Grace are low-h
        "cited_by_count": [100, 200, 300, 150, 250, 50, 80, 350, 320, 400],
        "num_datasets": [2, 3, 4, 2, 3, 20, 18, 1, 1, 1],
    })


@pytest.fixture
def viz_module():
    """Import and reload visualization module with Agg backend."""
    import visualization
    importlib.reload(visualization)
    return visualization


class TestGenerateAllPlots:
    """Tests for the generate_all_plots function."""

    def test_returns_expected_figures(self, viz_module, sample_viz_df):
        """Test that generate_all_plots returns expected number of figures."""
        figs = viz_module.generate_all_plots(sample_viz_df, save=False, show=False)
        assert len(figs) == 12

    def test_all_figures_are_valid(self, viz_module, sample_viz_df):
        """Test that all returned figures are valid matplotlib figures."""
        import matplotlib.pyplot as plt

        figs = viz_module.generate_all_plots(sample_viz_df, save=False, show=False)

        for i, fig in enumerate(figs):
            assert isinstance(fig, plt.Figure), f"Figure {i} is not a matplotlib Figure"
            assert len(fig.axes) > 0, f"Figure {i} has no axes"


class TestCorrelationMatrix:
    """Tests for correlation matrix heatmap."""

    def test_correlation_matrix_includes_s_index(self, viz_module, sample_viz_df):
        """Test that correlation matrix includes s_index."""
        fig = viz_module.plot_correlation_matrix(sample_viz_df, save=False)
        ax = fig.axes[0]

        # Get tick labels
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]

        assert "S-index" in tick_labels, "Correlation matrix should include S-index"

    def test_correlation_matrix_is_symmetric(self, viz_module, sample_viz_df):
        """Test that the correlation matrix heatmap is roughly square."""
        fig = viz_module.plot_correlation_matrix(sample_viz_df, save=False)
        ax = fig.axes[0]

        # Check that it's roughly square (aspect ratio close to 1)
        bbox = ax.get_position()
        aspect_ratio = bbox.width / bbox.height
        assert 0.5 < aspect_ratio < 2.0, "Correlation matrix should be roughly square"


class TestQuadrantAnalysis:
    """Tests for quadrant analysis plot."""

    def test_quadrant_analysis_has_correct_axes(self, viz_module, sample_viz_df):
        """Test that quadrant plot has correct axis labels."""
        fig = viz_module.plot_quadrant_analysis(sample_viz_df, save=False)
        ax = fig.axes[0]

        assert "h" in ax.get_xlabel().lower(), "X-axis should reference h-index"
        assert "s" in ax.get_ylabel().lower(), "Y-axis should reference S-index"

    def test_quadrant_analysis_has_dividing_lines(self, viz_module, sample_viz_df):
        """Test that quadrant plot has horizontal and vertical dividing lines."""
        fig = viz_module.plot_quadrant_analysis(sample_viz_df, save=False)
        ax = fig.axes[0]

        # Should have at least 2 lines (horizontal and vertical medians)
        lines = ax.get_lines()
        assert len(lines) >= 2, "Quadrant plot should have dividing lines"


class TestLogScatterPlot:
    """Tests for S vs h scatter (S on log scale, h linear)."""

    def test_log_scatter_s_log_h_linear(self, viz_module, sample_viz_df):
        """Test that S vs h scatter uses log scale on y (S) and linear on x (h)."""
        fig = viz_module.plot_s_vs_metric_scatter(sample_viz_df, save=False)
        ax = fig.axes[0]

        assert ax.get_xscale() == 'linear', "X-axis (h-index) should be linear"
        assert ax.get_yscale() == 'log', "Y-axis (S-index) should be log scale"


class TestSIndexDistribution:
    """Tests for S-index distribution plot."""

    def test_s_index_distribution_has_histogram(self, viz_module, sample_viz_df):
        """Test that S-index distribution shows a histogram."""
        fig = viz_module.plot_s_index_distribution(sample_viz_df, save=False)
        ax = fig.axes[0]

        # Check for patches (histogram bars)
        patches = ax.patches
        assert len(patches) > 0, "Distribution plot should have histogram bars"

    def test_s_index_distribution_single_wide_view(self, viz_module, sample_viz_df):
        """Test that S-index distribution has one wide panel (S > 0, linear scale)."""
        fig = viz_module.plot_s_index_distribution(sample_viz_df, save=False)

        assert len(fig.axes) == 1, "Should have 1 subplot (S > 0 distribution)"


class TestEdgeCases:
    """Edge case tests for visualization functions."""

    def test_handles_zero_s_index_values(self, viz_module):
        """Test that plots handle data with s_index = 0."""
        df = pd.DataFrame({
            "openalex_id": ["A1", "A2", "A3", "A4", "A5"],
            "name": ["A", "B", "C", "D", "E"],
            "s_index": [0, 0, 5, 10, 0],  # Most are zero
            "h_index": [10, 15, 20, 12, 18],
        })

        # Should not raise an error
        figs = viz_module.generate_all_plots(df, save=False, show=False)
        assert len(figs) == 9

    def test_handles_small_dataset(self, viz_module):
        """Test that plots handle very small datasets."""
        df = pd.DataFrame({
            "openalex_id": ["A1", "A2", "A3", "A4"],
            "name": ["A", "B", "C", "D"],
            # Add slight noise to avoid perfect linear fit (which causes zero variance)
            "s_index": [5, 11, 14, 22],
            "h_index": [10, 20, 30, 40],
        })

        # Should not raise an error (though some plots may have warnings)
        figs = viz_module.generate_all_plots(df, save=False, show=False)
        assert len(figs) == 9
