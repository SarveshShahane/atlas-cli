"""
SHAP Plot Generator — Phase 8.

Generates publication-quality SHAP visualizations:
  - Summary beeswarm plot
  - Feature importance horizontal bar chart

Plots use a dark-themed style and are saved as PNG artifacts.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for CLI
import matplotlib.pyplot as plt
import numpy as np

from atlas_cli.agents.explainability.schemas import ExplainabilityResult
from atlas_cli.core.config import settings

logger = logging.getLogger("atlas_cli")

# Dark theme colors
_BG_COLOR = "#1a1a2e"
_FG_COLOR = "#e0e0e0"
_ACCENT_CYAN = "#00d4ff"
_ACCENT_MAGENTA = "#ff6bcb"
_ACCENT_YELLOW = "#ffd93d"
_BAR_GRADIENT = ["#00d4ff", "#6c63ff", "#ff6bcb", "#ff9a3c", "#ffd93d"]


def _apply_dark_theme() -> None:
    """Apply premium dark theme to matplotlib."""
    plt.rcParams.update({
        "figure.facecolor": _BG_COLOR,
        "axes.facecolor": "#16213e",
        "axes.edgecolor": "#2a2a4a",
        "axes.labelcolor": _FG_COLOR,
        "text.color": _FG_COLOR,
        "xtick.color": _FG_COLOR,
        "ytick.color": _FG_COLOR,
        "grid.color": "#2a2a4a",
        "grid.alpha": 0.4,
        "font.family": "sans-serif",
        "font.size": 11,
        "figure.dpi": 150,
    })


def generate_plots(
    result: ExplainabilityResult,
    run_id: str,
) -> tuple[str, str]:
    """
    Generate SHAP visualization plots and save them to the run directory.

    Args:
        result: ExplainabilityResult with attached _shap_matrix and _X_test.
        run_id: Run identifier.

    Returns:
        Tuple of (summary_plot_path, importance_plot_path).
    """
    import shap

    run_dir = settings.workspace_dir / "runs" / run_id
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    _apply_dark_theme()

    shap_matrix = getattr(result, "_shap_matrix", None)
    X_test = getattr(result, "_X_test", None)

    summary_path = str(plots_dir / "shap_summary.png")
    importance_path = str(plots_dir / "shap_feature_importance.png")

    # ── SHAP Summary Beeswarm Plot ───────────────────────────────────────
    if shap_matrix is not None and X_test is not None:
        try:
            fig_summary, ax_summary = plt.subplots(figsize=(12, 8))
            plt.sca(ax_summary)

            shap.summary_plot(
                shap_matrix,
                X_test,
                feature_names=result.feature_names,
                show=False,
                plot_size=None,
                color_bar_label="Feature Value",
            )

            ax_summary.set_title(
                f"SHAP Summary — {result.model_name}",
                fontsize=14,
                fontweight="bold",
                color=_ACCENT_CYAN,
                pad=15,
            )

            fig_summary.tight_layout()
            fig_summary.savefig(
                summary_path,
                facecolor=fig_summary.get_facecolor(),
                edgecolor="none",
                bbox_inches="tight",
            )
            plt.close(fig_summary)
            logger.info(f"SHAP summary plot saved: {summary_path}")
        except Exception as exc:
            logger.warning(f"Failed to generate SHAP summary plot: {exc}")
            summary_path = ""
    else:
        summary_path = ""

    # ── Feature Importance Bar Chart ─────────────────────────────────────
    try:
        top_n = min(20, len(result.global_importances))
        top_features = result.global_importances[:top_n]

        # Reverse for horizontal bar chart (top feature at top)
        names = [f.feature_name for f in reversed(top_features)]
        values = [f.mean_abs_shap for f in reversed(top_features)]

        fig_bar, ax_bar = plt.subplots(figsize=(10, max(6, top_n * 0.4)))

        # Create gradient-colored bars
        colors = []
        n = len(values)
        for i in range(n):
            ratio = i / max(n - 1, 1)
            # Interpolate between cyan and magenta
            r1, g1, b1 = 0, 0.83, 1.0     # cyan
            r2, g2, b2 = 1.0, 0.42, 0.80   # magenta
            r = r1 + (r2 - r1) * ratio
            g = g1 + (g2 - g1) * ratio
            b = b1 + (b2 - b1) * ratio
            colors.append((r, g, b, 0.85))

        bars = ax_bar.barh(names, values, color=colors, edgecolor="#2a2a4a", linewidth=0.5)

        # Add value labels
        for bar, val in zip(bars, values):
            ax_bar.text(
                bar.get_width() + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                fontsize=9,
                color=_FG_COLOR,
                alpha=0.8,
            )

        ax_bar.set_xlabel("Mean |SHAP Value|", fontsize=12, color=_ACCENT_CYAN)
        ax_bar.set_title(
            f"Feature Importance — {result.model_name}",
            fontsize=14,
            fontweight="bold",
            color=_ACCENT_CYAN,
            pad=15,
        )
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)

        fig_bar.tight_layout()
        fig_bar.savefig(
            importance_path,
            facecolor=fig_bar.get_facecolor(),
            edgecolor="none",
            bbox_inches="tight",
        )
        plt.close(fig_bar)
        logger.info(f"Feature importance plot saved: {importance_path}")
    except Exception as exc:
        logger.warning(f"Failed to generate feature importance plot: {exc}")
        importance_path = ""

    return summary_path, importance_path
