
import numpy as np
from aux_auto_dp.engine import (
    probs_from_team_numbers_with_sf_roe,
    expected_runs_per_inning_with_sf_roe,
)

def test_engine_runs_smoke():
    probs = probs_from_team_numbers_with_sf_roe(
        pa=6082, h=1327, bb=498, hbp=67, doubles=259, triples=23, hr=182, sf=42, roe=35
    )
    r_inn = expected_runs_per_inning_with_sf_roe(probs, theta_long_single=0.8, theta_long_double=0.9, eps=1e-12)
    assert 0.2 < r_inn < 1.0  # coarse sanity for MLB environment
