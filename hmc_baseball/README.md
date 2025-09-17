
# aux-auto-dp-mlb

Production-grade implementation of the **Auxiliary + Automatic** DP for MLB inning scoring (homogeneous hitters),
including **advancing outs** (sac-fly-like) and **ROE** as on-base channel.

## Install (editable)

```bash
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
```

## Reproduce results

```bash
aux-dp --csv data/2024_baseball_stats.csv --out out
```

or from Python:

```python
from aux_auto_dp import reproduce_everything
team_df, summary_df = reproduce_everything("data/2024_baseball_stats.csv", output_dir="out", do_calibration=True)
```

The command writes CSV tables and RGB-only figures to the chosen output directory.
See `notebooks/Report.ipynb` for a fully reproducible notebook.

## Testing

```bash
pytest -q
```

## Structure

```
src/aux_auto_dp/
  engine.py          # DP engine (AO + ROE), exact suffix enumeration
  pipeline.py        # One-shot pipeline (baseline + calibration + plots)
  cli.py             # CLI entry point
notebooks/
  Report.ipynb       # Reproduce key results
data/
  2024_baseball_stats.csv (provided by you)
tests/
  test_engine.py
  test_pipeline.py
```

## License

MIT
