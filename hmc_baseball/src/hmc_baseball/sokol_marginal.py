#!/usr/bin/env python3
"""sokol_marginal.py

Usage:
    python sokol_marginal.py --data 1998_NL_stats.csv
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional
import numpy as np
import pandas as pd
import logging
import sys

__author__ = 'kqureshi'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def load_1998_nl_totals(csv_path: Path) -> Dict[str, int]:
    if not csv_path.exists():
        logging.error(f"Input CSV file not found: {csv_path}")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    if df.shape[0] == 0:
        logging.error("CSV does not contain any data rows.")
        sys.exit(1)
    row = df.iloc[0]
    required_fields = ["PA", "AB", "H", "2B", "3B", "HR", "BB", "SO"]
    for field in required_fields:
        if field not in row:
            logging.error(f"Missing column: {field}")
            sys.exit(1)
    PA = int(row["PA"])
    AB = int(row["AB"])
    H = int(row["H"])
    _2B = int(row["2B"])
    _3B = int(row["3B"])
    HR = int(row["HR"])
    BB = int(row["BB"])
    HBP = int(row.get("HBP", 0))
    SO = int(row["SO"])
    SB = int(row.get("SB", 0))
    CS = int(row.get("CS", 0))
    SH = int(row.get("SH", 0))
    SF = int(row.get("SF", 0))
    GDP = int(row.get("GDP", 0))

    singles = H - _2B - _3B - HR
    AB_outs = AB - H
    nonK_outs = AB_outs - SO
    return {
        "PA": PA, "AB": AB, "H": H, "1B": singles, "2B": _2B, "3B": _3B, "HR": HR,
        "BB": BB, "HBP": HBP, "SO": SO, "SB": SB, "CS": CS, "SH": SH, "SF": SF, "GDP": GDP,
        "AB_outs": AB_outs, "nonK_outs": nonK_outs
    }

BASE_MASKS = list(range(8))  # 0-7, bitwise: 1st, 2nd, 3rd base
OUTS = [0, 1, 2]
STATE_TO_IDX = {(b, o): idx for idx, (b, o) in enumerate((b, o) for b in BASE_MASKS for o in OUTS)}
IDX_TO_STATE = {idx: bo for bo, idx in STATE_TO_IDX.items()}
ABSORB_IDX = 24  # "3 outs" state

def pc3(b: int) -> int:
    """Population count for base occupancy bits."""
    return (b & 1) + ((b >> 1) & 1) + ((b >> 2) & 1)

def t_runs_pa(b: int, o: int, nb: int, no: int) -> int:
    """Run calculation for plate appearance events."""
    return 1 + (pc3(b) + o) - (pc3(nb) + no)

def t_runs_nonpa(b: int, o: int, nb: int, no: int) -> int:
    """Run calculation for non-PA events (SB/CS)."""
    return (pc3(b) + o) - (pc3(nb) + no)

# ------------------------- Event transition model -----------------------

class AdvParams:
    """Advancement probability parameters."""
    p_1B_1st_to_3rd: float = 0.50
    p_1B_2nd_scores: float = 0.60
    p_2B_1st_scores: float = 0.40
    p_FO_3rd_scores: float = 0.30
    p_FO_2nd_to_3rd: float = 0.10
    p_GO_3rd_scores: float = 0.25

def trans_event(b: int, o: int, e: str, p: AdvParams) -> List[Tuple[float, int, int]]:
    """Stochastic transitions for PA events."""
    b1, b2, b3 = bool(b & 1), bool(b & 2), bool(b & 4)
    res: List[Tuple[float, int, int]] = []

    def add(nb: int, no: int, pr: float):
        if pr > 0:
            res.append((pr, nb, no))

    if e == "K":
        add(b, min(o+1, 3), 1.0)
    elif e == "BB":
        nb1, nb2, nb3 = b1, b2, b3
        if nb1:
            if nb2:
                if nb3:
                    pass
                nb3 = True
            nb2 = True
        nb1 = True
        nb = (1 if nb1 else 0) | (2 if nb2 else 0) | (4 if nb3 else 0)
        add(nb, o, 1.0)
    elif e == "1B":
        p13 = p.p_1B_1st_to_3rd if b1 else 0.0
        p2H = p.p_1B_2nd_scores if b2 else 0.0
        for r1_to3, pr1 in ([(True, p13),(False, 1-p13)] if b1 else [(False, 1.0)]):
            for r2_scores, pr2 in ([(True, p2H),(False, 1-p2H)] if b2 else [(False, 1.0)]):
                nb1 = True
                nb2 = (b1 and not r1_to3)
                nb3 = (b2 and not r2_scores) or (b1 and r1_to3)
                nb = (1 if nb1 else 0) | (2 if nb2 else 0) | (4 if nb3 else 0)
                add(nb, o, pr1 * pr2)
    elif e == "2B":
        p1H = p.p_2B_1st_scores if b1 else 0.0
        for r1_scores, pr in ([(True, p1H), (False, 1-p1H)] if b1 else [(False, 1.0)]):
            nb1 = False
            nb2 = True
            nb3 = (b1 and not r1_scores)
            nb = (1 if nb1 else 0) | (2 if nb2 else 0) | (4 if nb3 else 0)
            add(nb, o, pr)
    elif e == "3B":
        add(4, o, 1.0)
    elif e == "HR":
        add(0, o, 1.0)
    elif e == "FO":
        o2 = min(o + 1, 3)
        p_sf = p.p_FO_3rd_scores if (b3 and o < 2) else 0.0
        p_tag = p.p_FO_2nd_to_3rd if (b2 and o < 2) else 0.0
        for third_scores, pr3 in ([(True, p_sf), (False, 1-p_sf)] if p_sf > 0 else [(False, 1.0)]):
            for second_tags, pr2 in ([(True, p_tag), (False, 1-p_tag)] if p_tag > 0 else [(False, 1.0)]):
                nb1 = b1
                nb2 = (b2 and not second_tags)
                nb3 = (b3 and not third_scores) or (b2 and second_tags)
                nb = (1 if nb1 else 0) | (2 if nb2 else 0) | (4 if nb3 else 0)
                add(nb, o2, pr3 * pr2)
    elif e == "GO":
        o2 = min(o + 1, 3)
        p_go = p.p_GO_3rd_scores if (b3 and o < 2) else 0.0
        for third_scores, pr in ([(True, p_go), (False, 1-p_go)] if p_go > 0 else [(False, 1.0)]):
            nb1, nb2, nb3n = b1, b2, (b3 and not third_scores)
            nb = (1 if nb1 else 0) | (2 if nb2 else 0) | (4 if nb3n else 0)
            add(nb, o2, pr)
    elif e == "DP":
        if (b & 1) and (o < 2):
            nb = b & (~1)
            add(nb, min(o+2, 3), 1.0)
        else:
            return trans_event(b, o, "GO", p)
    else:
        raise ValueError(f"Unknown event: {e!r}")
    return res

# ------------------- Markov chains: build and solve ---------------------

def build_absorbing(event_probs: Dict[str, float], p: AdvParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = 24
    P = np.zeros((n + 1, n + 1))
    r = np.zeros(n)
    for i in range(n):
        b, o = IDX_TO_STATE[i]
        for e, pe in event_probs.items():
            for pr, nb, no in trans_event(b, o, e, p):
                j = ABSORB_IDX if no >= 3 else STATE_TO_IDX[(nb, no)]
                P[i, j] += pe * pr
                r[i] += pe * pr * t_runs_pa(b, o, nb, min(no, 3))
    P[ABSORB_IDX, ABSORB_IDX] = 1.0
    v = np.linalg.solve(np.eye(n) - P[:n, :n], r)
    return P, r, v

def build_wraparound(event_probs: Dict[str, float], p: AdvParams) -> np.ndarray:
    n = 24
    Pw = np.zeros((n, n))
    start = STATE_TO_IDX[(0,0)]
    for i in range(n):
        b, o = IDX_TO_STATE[i]
        for e, pe in event_probs.items():
            for pr, nb, no in trans_event(b, o, e, p):
                j = start if no >= 3 else STATE_TO_IDX[(nb, no)]
                Pw[i, j] += pe * pr
    return Pw

def stationary(Pw: np.ndarray) -> np.ndarray:
    n = Pw.shape[0]
    M = Pw.T - np.eye(n)
    M[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    return np.linalg.solve(M, b)

# -------------------------- Event value calculators ---------------------

def value_pa_event(event: str, pi: np.ndarray, v: np.ndarray, p: AdvParams) -> Tuple[float, float, float]:
    n = 24
    real = pot = tot = 0.0
    for i in range(n):
        b, o = IDX_TO_STATE[i]
        for pr, nb, no in trans_event(b, o, event, p):
            t = t_runs_pa(b, o, nb, min(no, 3))
            nv = 0.0 if no >= 3 else v[STATE_TO_IDX[(nb, no)]]
            real += pi[i] * pr * t
            pot += pi[i] * pr * (nv - v[i])
            tot += pi[i] * pr * (t + nv - v[i])
    return real, pot, tot

def value_dp_conditional(pi: np.ndarray, v: np.ndarray) -> Tuple[float, float, float]:
    n = 24
    real = pot = tot = den = 0.0
    for i in range(n):
        b, o = IDX_TO_STATE[i]
        if (b & 1) and (o < 2):
            nb = b & (~1)
            no = min(o+2, 3)
            t = t_runs_pa(b, o, nb, min(no, 3))
            nv = 0.0 if no >= 3 else v[STATE_TO_IDX[(nb, no)]]
            w = pi[i]
            real += w * t
            pot += w * (nv - v[i])
            tot += w * (t + nv - v[i])
            den += w
    return (real / den, pot / den, tot / den) if den else (0.0, 0.0, 0.0)

def value_sb_cs_conditional(pi: np.ndarray, v: np.ndarray) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    n = 24
    # SB
    real = pot = tot = den = 0.0
    for i in range(n):
        b, o = IDX_TO_STATE[i]
        trans = []
        if (b & 1) and not (b & 2):
            trans = [(((b | 2) & (~1)), o)]
        elif (b & 2) and not (b & 4):
            trans = [(((b | 4) & (~2)), o)]
        elif (b & 1) and (b & 2) and not (b & 4):
            trans = [(((b | 6) & (~1)), o)]  # 12->23
        if not trans:
            continue
        w = pi[i]
        for nb, no in trans:
            t = t_runs_nonpa(b, o, nb, no)
            nv = v[STATE_TO_IDX[(nb, no)]]
            real += w * t
            pot += w * (nv - v[i])
            tot += w * (t + nv - v[i])
            den += w
    sb = (real / den, pot / den, tot / den) if den else (0.0, 0.0, 0.0)

    # CS
    real = pot = tot = den = 0.0
    for i in range(n):
        b, o = IDX_TO_STATE[i]
        trans = []
        if (b & 1) and (b & 2) and not (b & 4):
            trans = [(4, min(o+1, 3))]   # 12->3, +1 out
        elif (b & 1):
            trans = [(b & (~1), min(o+1, 3))]
        elif (b & 2):
            trans = [(b & (~2), min(o+1, 3))]
        if not trans:
            continue
        w = pi[i]
        for nb, no in trans:
            t = t_runs_nonpa(b, o, nb, no)
            nv = 0.0 if no >= 3 else v[STATE_TO_IDX[(nb, no)]]
            real += w * t
            pot += w * (nv - v[i])
            tot += w * (t + nv - v[i])
            den += w
    cs = (real / den, pot / den, tot / den) if den else (0.0, 0.0, 0.0)
    return sb, cs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="1998_NL_stats.csv")
    parser.add_argument("--fly-share", type=float, default=0.45)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory")
    args = parser.parse_args()

    T = load_1998_nl_totals(args.data)

    bip_outs = T["nonK_outs"] - T["GDP"]
    FO = int(round(bip_outs * args.fly_share)) + T["SF"]
    GO = (bip_outs - int(round(bip_outs * args.fly_share))) + T["SH"]

    # Plate appearance event probabilities
    event_counts_pa = {
        "1B": T["1B"],
        "2B": T["2B"],
        "3B": T["3B"],
        "HR": T["HR"],
        "BB": T["BB"] + T["HBP"],
        "K": T["SO"],
        "FO": FO,
        "GO": GO,
        "DP": T["GDP"]
    }
    delta = T["PA"] - sum(event_counts_pa.values())
    if abs(delta) > 0:
        logging.warning(f"Adjusting for rounding error: GO += {delta}")
        event_counts_pa["GO"] += delta
    event_probs = {k: v / T["PA"] for k, v in event_counts_pa.items()}

    # Advancement parameters
    adv_params = AdvParams()

    # Solve Markov chains
    _, _, v = build_absorbing(event_probs, adv_params)
    P_wrap = build_wraparound(event_probs, adv_params)
    pi = stationary(P_wrap)

    # Event values
    pa_events = ["1B", "2B", "3B", "HR", "BB", "K", "FO", "GO"]
    rows = []
    for e in pa_events:
        real, pot, tot = value_pa_event(e, pi, v, adv_params)
        rows.append((e, real, pot, tot))
    dp_real, dp_pot, dp_tot = value_dp_conditional(pi, v)
    rows.append(("DP", dp_real, dp_pot, dp_tot))
    (sb_real, sb_pot, sb_tot), (cs_real, cs_pot, cs_tot) = value_sb_cs_conditional(pi, v)
    rows.append(("SB", sb_real, sb_pot, sb_tot))
    rows.append(("CS", cs_real, cs_pot, cs_tot))

    out_df = pd.DataFrame(rows, columns=["Event", "Realization", "Potential", "Total"])

    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)
    ev_file = output_dir / "event_values_1998NL.csv"
    state_file = output_dir / "states_1998NL.csv"
    out_df.to_csv(ev_file, index=False)

    # State table for reference
    states = []
    for i in range(24):
        b, o = IDX_TO_STATE[i]
        bases = ''.join(d for d, bit in zip(['1', '2', '3'], [1, 2, 4]) if (b & bit)) or "Empty"
        states.append({
            "state_idx": i,
            "bases": bases,
            "outs": o,
            "v": float(v[i]),
            "pi": float(pi[i])
        })
    pd.DataFrame(states).to_csv(state_file, index=False)

    # Print summary
    v_empty0 = float(v[STATE_TO_IDX[(0, 0)]])
    print("\n1998 NL — Sokol §4 event values (runs/event):")
    with pd.option_context("display.float_format", lambda x: f"{x:.3f}"):
        print(out_df.to_string(index=False))
    print(f"\nExpected runs/inning v(empty,0): {v_empty0:.3f}")
    print(f"Outputs saved to: {ev_file} and {state_file}")

if __name__ == "__main__":
    main()

