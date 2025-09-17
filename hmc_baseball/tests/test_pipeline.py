
import os
import pandas as pd
from aux_auto_dp.pipeline import reproduce_everything

def test_pipeline_smoke(tmp_path):
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "2024_baseball_stats.csv")
    csv_path = os.path.abspath(csv_path)
    if not os.path.exists(csv_path):
        # allow test to gracefully skip if data not present
        return
    out_dir = tmp_path / "out"
    out_dir.mkdir(exist_ok=True)
    team_df, summary_df = reproduce_everything(csv_path, output_dir=str(out_dir), do_calibration=False)
    assert not team_df.empty
    assert not summary_df.empty
