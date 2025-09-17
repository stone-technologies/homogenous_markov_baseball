
"""
One-shot pipeline: load CSV, compute Aux+Auto DP predictions (baseline + calibration),
and write CSV tables and RGB-only figures.

Public API
----------
reproduce_everything(csv_path, output_dir=".", do_calibration=True)
  -> (team_df, summary_df)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .engine import (
    probs_from_team_numbers_with_sf_roe,
    expected_runs_per_game_with_sf_roe,
)


@dataclass
class CalibrationResult:
    """Holds per-team calibration outputs."""
    theta_long_single: float
    theta_long_double: float
    roe_count: int
    pred_r_per_game: float
    error: float


def _team_to_event_probs(row: pd.Series, roe_count: int) -> Dict[str, float]:
    """Map a team row to per-PA event probabilities (includes AO and ROE)."""
    return probs_from_team_numbers_with_sf_roe(
        pa=float(row["PA"]),
        h=float(row["H"]),
        bb=float(row["BB"]),
        hbp=float(row["HBP"]),
        doubles=float(row["2B"]),
        triples=float(row["3B"]),
        hr=float(row["HR"]),
        sf=float(row["SF"]),
        roe=float(roe_count),
    )


def _predict_rpg(
    row: pd.Series,
    theta_long_single: float,
    theta_long_double: float,
    roe_count: int,
) -> float:
    """Predicted runs/game given parameters for a team row."""
    probs = _team_to_event_probs(row, roe_count)
    return float(
        expected_runs_per_game_with_sf_roe(
            probs,
            theta_long_single=theta_long_single,
            theta_long_double=theta_long_double,
            eps=1e-14,
        )
    )


def _calibrate_team(
    row: pd.Series,
    theta_grid: Optional[List[float]] = None,
    theta_d_grid: Optional[List[float]] = None,
    roe_grid: Optional[List[int]] = None,
) -> CalibrationResult:
    """
    Grid-search calibration per team; minimizes squared R/G error.
    Defaults are compact (fast). Expand grids for tighter fit.
    """
    if theta_grid is None:
        theta_grid = [0.7, 0.8]
    if theta_d_grid is None:
        theta_d_grid = [0.8, 0.9, 1.0]
    if roe_grid is None:
        roe_grid = [30, 35, 40]

    target = float(row["R/G"])
    best = CalibrationResult(0.8, 0.9, 35, pred_r_per_game=0.0, error=float("inf"))

    for th_s in theta_grid:
        for th_d in theta_d_grid:
            for roe in roe_grid:
                pred = _predict_rpg(row, th_s, th_d, roe)
                err = (pred - target) ** 2
                if err < best.error:
                    best = CalibrationResult(th_s, th_d, roe, pred, err)

    return best


def reproduce_everything(
    csv_path: str,
    output_dir: str = ".",
    baseline_theta_single: float = 0.80,
    baseline_theta_double: float = 0.80,
    baseline_roe: int = 35,
    do_calibration: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end pipeline:
      1) Load the 2024 team batting CSV.
      2) Compute baseline DP predictions (SF + ROE, long/short splits).
      3) Optionally calibrate (theta_long_single, theta_long_double, ROE) per team.
      4) Save tables and RGB-only figures.

    The CSV must contain columns: Tm, R/G, PA, H, BB, HBP, 2B, 3B, HR, SF
    (ROE is estimated via `baseline_roe` or calibrated per team).
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    required = {"Tm", "R/G", "PA", "H", "BB", "HBP", "2B", "3B", "HR", "SF"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    rows: List[dict] = []
    calib_rows: List[dict] = []

    for _, row in df.iterrows():
        tm = str(row["Tm"])
        actual = float(row["R/G"])

        # Baseline prediction with global parameters
        pred_base = _predict_rpg(row, baseline_theta_single, baseline_theta_double, baseline_roe)

        team_out = {
            "Tm": tm,
            "R/G_actual": actual,
            "Pred_R/G_baseline": pred_base,
            "theta_single_base": baseline_theta_single,
            "theta_double_base": baseline_theta_double,
            "ROE_base": baseline_roe,
            "abs_err_base": abs(pred_base - actual),
        }

        if do_calibration:
            cal = _calibrate_team(row)
            team_out.update(
                {
                    "Pred_R/G_cal": cal.pred_r_per_game,
                    "theta_single_cal": cal.theta_long_single,
                    "theta_double_cal": cal.theta_long_double,
                    "ROE_cal": cal.roe_count,
                    "abs_err_cal": abs(cal.pred_r_per_game - actual),
                }
            )
            calib_rows.append(
                {
                    "Tm": tm,
                    "theta_single": cal.theta_long_single,
                    "theta_double": cal.theta_long_double,
                    "ROE": cal.roe_count,
                }
            )

        rows.append(team_out)

    results = pd.DataFrame(rows).sort_values("R/G_actual", ascending=False).reset_index(drop=True)
    calib_df = pd.DataFrame(calib_rows).sort_values("Tm").reset_index(drop=True) if do_calibration else pd.DataFrame()

    def _rmse(col_pred: str) -> float:
        return float(np.sqrt(np.mean((results[col_pred] - results["R/G_actual"]) ** 2)))

    summary = {"RMSE_baseline": _rmse("Pred_R/G_baseline"), "MAE_baseline": float(np.mean(results["abs_err_base"]))}
    if do_calibration:
        summary.update({"RMSE_cal": _rmse("Pred_R/G_cal"), "MAE_cal": float(np.mean(results["abs_err_cal"]))})
    summary_df = pd.DataFrame([summary])

    # Write CSVs
    team_csv = os.path.join(output_dir, "dp_aux_auto_team_predictions.csv")
    results.to_csv(team_csv, index=False)

    summary_csv = os.path.join(output_dir, "dp_aux_auto_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    if do_calibration and not calib_df.empty:
        calib_csv = os.path.join(output_dir, "dp_aux_auto_calibration_params.csv")
        calib_df.to_csv(calib_csv, index=False)

    # Figures: RGB only
    # Baseline scatter
    fig1 = plt.figure(figsize=(6, 6))
    plt.scatter(results["R/G_actual"], results["Pred_R/G_baseline"], c="b", s=32, alpha=0.9, edgecolors="none")
    lo = min(results["R/G_actual"].min(), results["Pred_R/G_baseline"].min()) - 0.2
    hi = max(results["R/G_actual"].max(), results["Pred_R/G_baseline"].max()) + 0.2
    plt.plot([lo, hi], [lo, hi], "r-", linewidth=1.5)
    plt.xlim([lo, hi])
    plt.ylim([lo, hi])
    plt.xlabel("Actual R/G")
    plt.ylabel("Predicted R/G (Baseline)")
    plt.title("Predicted vs. Actual R/G — Baseline")
    fig1_path = os.path.join(output_dir, "scatter_baseline_rgb.png")
    plt.savefig(fig1_path, dpi=190, bbox_inches="tight")
    plt.close(fig1)

    # Calibrated scatter
    if do_calibration:
        fig2 = plt.figure(figsize=(6, 6))
        plt.scatter(results["R/G_actual"], results["Pred_R/G_cal"], c="g", s=32, alpha=0.9, edgecolors="none")
        lo = min(results["R/G_actual"].min(), results["Pred_R/G_cal"].min()) - 0.2
        hi = max(results["R/G_actual"].max(), results["Pred_R/G_cal"].max()) + 0.2
        plt.plot([lo, hi], [lo, hi], "r-", linewidth=1.5)
        plt.xlim([lo, hi])
        plt.ylim([lo, hi])
        plt.xlabel("Actual R/G")
        plt.ylabel("Predicted R/G (Calibrated)")
        plt.title("Predicted vs. Actual R/G — Calibrated")
        fig2_path = os.path.join(output_dir, "scatter_calibrated_rgb.png")
        plt.savefig(fig2_path, dpi=190, bbox_inches="tight")
        plt.close(fig2)

    # Residuals
    fig3 = plt.figure(figsize=(10, 6))
    res_base = results.set_index("Tm")["Pred_R/G_baseline"] - results.set_index("Tm")["R/G_actual"]
    res_base = res_base.sort_values()
    plt.bar(res_base.index, res_base.values, color="b")
    plt.axhline(0, color="r", linewidth=1.0)
    plt.xticks(rotation=90)
    plt.ylabel("Residual (Pred - Actual) R/G")
    plt.title("Residuals — Baseline")
    fig3_path = os.path.join(output_dir, "residuals_baseline_rgb.png")
    plt.savefig(fig3_path, dpi=190, bbox_inches="tight")
    plt.close(fig3)

    if do_calibration:
        fig4 = plt.figure(figsize=(10, 6))
        res_cal = results.set_index("Tm")["Pred_R/G_cal"] - results.set_index("Tm")["R/G_actual"]
        res_cal = res_cal.sort_values()
        plt.bar(res_cal.index, res_cal.values, color="g")
        plt.axhline(0, color="r", linewidth=1.0)
        plt.xticks(rotation=90)
        plt.ylabel("Residual (Pred - Actual) R/G")
        plt.title("Residuals — Calibrated")
        fig4_path = os.path.join(output_dir, "residuals_calibrated_rgb.png")
        plt.savefig(fig4_path, dpi=190, bbox_inches="tight")
        plt.close(fig4)

    # Line plot
    fig5 = plt.figure(figsize=(12, 6))
    df_line = results.copy().sort_values("R/G_actual").reset_index(drop=True)
    x = np.arange(len(df_line))
    plt.plot(x, df_line["R/G_actual"], "b-", label="Actual R/G")
    plt.plot(x, df_line["Pred_R/G_baseline"], "r-", label="Pred R/G (Baseline)")
    if do_calibration:
        plt.plot(x, df_line["Pred_R/G_cal"], "g-", label="Pred R/G (Calibrated)")
    plt.xlabel("Teams (sorted by Actual R/G)")
    plt.ylabel("Runs per Game")
    plt.title("Actual vs Predicted R/G")
    plt.legend()
    fig5_path = os.path.join(output_dir, "rg_lineplot_rgb.png")
    plt.savefig(fig5_path, dpi=190, bbox_inches="tight")
    plt.close(fig5)

    return results, summary_df
