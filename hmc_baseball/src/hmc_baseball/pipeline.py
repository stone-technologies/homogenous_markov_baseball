from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .engine import (
    probs_from_team_numbers_with,
    expected_runs_per_game_with,
)

__author__ = 'kqureshi, jorlin'

@dataclass
class CalibrationResult:
    """Stores calibration results for team prediction."""
    theta_long_single: float
    theta_long_double: float
    roe_count: int
    pred_r_per_game: float
    error: float


def team_to_event_probs(row: pd.Series, roe_count: int) -> Dict[str, float]:
    """Converts team data into event probabilities including SF and ROE."""
    return probs_from_team_numbers_with(
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


def predict_rpg(
    row: pd.Series,
    theta_single: float,
    theta_double: float,
    roe_count: int,
) -> float:
    """Predicts runs per game using provided parameters."""
    event_probs = team_to_event_probs(row, roe_count)
    return float(
        expected_runs_per_game_with(
            event_probs, theta_single, theta_double, eps=1e-14
        )
    )


def calibrate_team(
    row: pd.Series,
    theta_grid: Optional[List[float]] = None,
    theta_d_grid: Optional[List[float]] = None,
    roe_grid: Optional[List[int]] = None,
) -> CalibrationResult:
    """Calibrates parameters to minimize prediction error for a team row."""
    if theta_grid is None:
        theta_grid = [0.7, 0.8]
    if theta_d_grid is None:
        theta_d_grid = [0.8, 0.9, 1.0]
    if roe_grid is None:
        roe_grid = [30, 35, 40]

    target = float(row["R/G"])
    best = CalibrationResult(0.8, 0.9, 35, 0.0, float("inf"))
    for th_single in theta_grid:
        for th_double in theta_d_grid:
            for roe in roe_grid:
                pred = predict_rpg(row, th_single, th_double, roe)
                error = (pred - target) ** 2
                if error < best.error:
                    best = CalibrationResult(th_single, th_double, roe, pred, error)
    return best


def main(
    csv_path: str,
    output_dir: str = ".",
    baseline_theta_single: float = 0.80,
    baseline_theta_double: float = 0.80,
    baseline_roe: int = 35,
    do_calibration: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs team run prediction and calibration, saves outputs, and returns result DataFrames.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    required_cols = {
        "Tm", "R/G", "PA", "H", "BB", "HBP", "2B", "3B", "HR", "SF"
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {sorted(missing_cols)}")

    results_rows: List[Dict] = []
    calibration_rows: List[Dict] = []

    for _, row in df.iterrows():
        team = str(row["Tm"])
        actual_rpg = float(row["R/G"])
        pred_baseline = predict_rpg(
            row, baseline_theta_single, baseline_theta_double, baseline_roe
        )

        team_result = {
            "Tm": team,
            "R/G_actual": actual_rpg,
            "Pred_R/G_baseline": pred_baseline,
            "theta_single_base": baseline_theta_single,
            "theta_double_base": baseline_theta_double,
            "ROE_base": baseline_roe,
            "abs_err_base": abs(pred_baseline - actual_rpg),
        }

        if do_calibration:
            calibration = calibrate_team(row)
            team_result.update({
                "Pred_R/G_cal": calibration.pred_r_per_game,
                "theta_single_cal": calibration.theta_long_single,
                "theta_double_cal": calibration.theta_long_double,
                "ROE_cal": calibration.roe_count,
                "abs_err_cal": abs(calibration.pred_r_per_game - actual_rpg),
            })
            calibration_rows.append({
                "Tm": team,
                "theta_single": calibration.theta_long_single,
                "theta_double": calibration.theta_long_double,
                "ROE": calibration.roe_count,
            })

        results_rows.append(team_result)

    results_df = pd.DataFrame(results_rows).sort_values(
        "R/G_actual", ascending=False
    ).reset_index(drop=True)

    calibration_df = (
        pd.DataFrame(calibration_rows).sort_values("Tm").reset_index(drop=True)
        if do_calibration else pd.DataFrame()
    )

    def rmse(pred_col: str) -> float:
        return float(np.sqrt(
            np.mean((results_df[pred_col] - results_df["R/G_actual"]) ** 2)
        ))

    summary = {
        "RMSE_baseline": rmse("Pred_R/G_baseline"),
        "MAE_baseline": float(np.mean(results_df["abs_err_base"]))
    }
    if do_calibration:
        summary.update({
            "RMSE_cal": rmse("Pred_R/G_cal"),
            "MAE_cal": float(np.mean(results_df["abs_err_cal"])),
        })
    summary_df = pd.DataFrame([summary])

    results_df.to_csv(
        os.path.join(output_dir, "dp_aux_auto_team_predictions.csv"), index=False
    )
    summary_df.to_csv(
        os.path.join(output_dir, "dp_aux_auto_summary.csv"), index=False
    )
    if do_calibration and not calibration_df.empty:
        calibration_df.to_csv(
            os.path.join(output_dir, "dp_aux_auto_calibration_params.csv"), index=False
        )

    # Plot: Baseline scatter
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(
        results_df["R/G_actual"],
        results_df["Pred_R/G_baseline"],
        c="b", s=32, alpha=0.9, edgecolors="none"
    )
    lo = min(
        results_df["R/G_actual"].min(), results_df["Pred_R/G_baseline"].min()
    ) - 0.2
    hi = max(
        results_df["R/G_actual"].max(), results_df["Pred_R/G_baseline"].max()
    ) + 0.2
    plt.plot([lo, hi], [lo, hi], "r-", linewidth=1.5)
    plt.xlim([lo, hi])
    plt.ylim([lo, hi])
    plt.xlabel("Actual R/G")
    plt.ylabel("Predicted R/G (Baseline)")
    plt.title("Predicted vs. Actual R/G — Baseline")
    plt.savefig(
        os.path.join(output_dir, "scatter_baseline.png"),
        dpi=190, bbox_inches="tight")
    plt.close(fig)

    if do_calibration:
        fig = plt.figure(figsize=(6, 6))
        plt.scatter(
            results_df["R/G_actual"],
            results_df["Pred_R/G_cal"],
            c="g", s=32, alpha=0.9, edgecolors="none"
        )
        lo = min(
            results_df["R/G_actual"].min(), results_df["Pred_R/G_cal"].min()
        ) - 0.2
        hi = max(
            results_df["R/G_actual"].max(), results_df["Pred_R/G_cal"].max()
        ) + 0.2
        plt.plot([lo, hi], [lo, hi], "r-", linewidth=1.5)
        plt.xlim([lo, hi])
        plt.ylim([lo, hi])
        plt.xlabel("Actual R/G")
        plt.ylabel("Predicted R/G (Calibrated)")
        plt.title("Predicted vs. Actual R/G — Calibrated")
        plt.savefig(
            os.path.join(output_dir, "scatter_calibrated.png"),
            dpi=190, bbox_inches="tight"
        )
        plt.close(fig)

    # Residuals plots: Baseline
    fig = plt.figure(figsize=(10, 6))
    res_base = (
        results_df.set_index("Tm")["Pred_R/G_baseline"]
        - results_df.set_index("Tm")["R/G_actual"]
    )
    res_base = res_base.sort_values()
    plt.bar(res_base.index, res_base.values, color="b")
    plt.axhline(0, color="r", linewidth=1.0)
    plt.xticks(rotation=90)
    plt.ylabel("Residual (Pred - Actual) R/G")
    plt.title("Residuals — Baseline")
    plt.savefig(
        os.path.join(output_dir, "residuals_baseline.png"),
        dpi=190, bbox_inches="tight"
    )
    plt.close(fig)

    if do_calibration:
        fig = plt.figure(figsize=(10, 6))
        res_cal = (
            results_df.set_index("Tm")["Pred_R/G_cal"]
            - results_df.set_index("Tm")["R/G_actual"]
        )
        res_cal = res_cal.sort_values()
        plt.bar(res_cal.index, res_cal.values, color="g")
        plt.axhline(0, color="r", linewidth=1.0)
        plt.xticks(rotation=90)
        plt.ylabel("Residual (Pred - Actual) R/G")
        plt.title("Residuals — Calibrated")
        plt.savefig(
            os.path.join(output_dir, "residuals_calibrated.png"),
            dpi=190, bbox_inches="tight"
        )
        plt.close(fig)

    # Line plot for actual and predicted R/G
    fig = plt.figure(figsize=(12, 6))
    sorted_df = results_df.copy().sort_values("R/G_actual").reset_index(drop=True)
    x_vals = np.arange(len(sorted_df))
    plt.plot(x_vals, sorted_df["R/G_actual"], "b-", label="Actual R/G")
    plt.plot(x_vals, sorted_df["Pred_R/G_baseline"], "r-", label="Pred R/G (Baseline)")
    if do_calibration:
        plt.plot(x_vals, sorted_df["Pred_R/G_cal"], "g-", label="Pred R/G (Calibrated)")
    plt.xlabel("Teams (sorted by Actual R/G)")
    plt.ylabel("Runs per Game")
    plt.title("Actual vs Predicted R/G")
    plt.legend()
    plt.savefig(
        os.path.join(output_dir, "rg_lineplot.png"),
        dpi=190, bbox_inches="tight"
    )
    plt.close(fig)

    return results_df, summary_df
