
"""
aux_auto_dp: Auxiliary + Automatic DP with AO/ROE for MLB inning scoring.
"""
from .engine import (
    probs_from_team_numbers_with_sf_roe,
    p_end_distribution,
    inning_run_pmf_with_sf_roe,
    expected_runs_per_inning_with_sf_roe,
    expected_runs_per_game_with_sf_roe,
)
from .pipeline import reproduce_everything
