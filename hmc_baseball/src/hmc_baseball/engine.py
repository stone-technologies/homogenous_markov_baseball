from __future__ import annotations

from typing import Dict, List, Tuple
import math
import itertools
import numpy as np

ONBASE = ["BB", "1B", "2B", "3B", "HR", "ROE"]

__author__ = 'kqureshi, jorlin'

def probs_from_team_numbers_with(
    pa: float, h: float, bb: float, hbp: float, doubles: float, triples: float,
    hr: float, sf: float = 0.0, roe: float = 0.0
) -> Dict[str, float]:
    """
    Calculate event probabilities from team statistics
    """
    singles = max(h - doubles - triples - hr, 0.0)
    p_bb = (bb + hbp) / pa
    p_1b = singles / pa
    p_2b = doubles / pa
    p_3b = triples / pa
    p_hr = hr / pa
    p_roe = roe / pa
    p_ao = sf / pa

    p_out = max(1.0 - (p_bb + p_1b + p_2b + p_3b + p_hr + p_roe + p_ao), 0.0)
    probs = {
        "OUT": p_out,
        "AO": p_ao,
        "BB": p_bb,
        "1B": p_1b,
        "2B": p_2b,
        "3B": p_3b,
        "HR": p_hr,
        "ROE": p_roe
    }
    s = sum(probs.values())
    for k in probs:
        probs[k] /= s
    return probs

def p_end_distribution(
    q_out_total: float,
    eps: float = 1e-12,
    max_b: int | None = None
) -> np.ndarray:
    """
    Compute the probability distribution for total batters until the end of an inning.
    """
    if not (0.0 < q_out_total < 1.0):
        raise ValueError("q_out_total must be in (0, 1)")
    pob = 1.0 - q_out_total
    if max_b is None:
        tail_b = int(np.ceil(np.log(max(eps, 1e-300)) / np.log(max(1e-300, pob))))
        max_b = max(12, 3 + tail_b + 10)
    P = np.zeros((max_b + 1, 3))
    P[0, 0] = 1.0
    p_end = np.zeros(max_b + 1)
    for b in range(1, max_b + 1):
        P[b, 0] = P[b - 1, 0] * pob
        P[b, 1] = P[b - 1, 1] * pob + P[b - 1, 0] * q_out_total
        P[b, 2] = P[b - 1, 2] * pob + P[b - 1, 1] * q_out_total
        if b >= 3:
            p_end[b] = P[b - 1, 2] * q_out_total
    s = p_end.sum()
    if s > 0:
        p_end /= s
    return p_end

def aux_pmf_m_with(
    probs: Dict[str, float],
    m: int,
    theta_long_single: float = 0.6,
    theta_long_double: float = 0.9
) -> np.ndarray:
    """
    Computes auxiliary run-scoring probability mass function for given base state.
    """
    assert 0 <= m <= 3
    if m == 0:
        pmf = np.zeros(4)
        pmf[0] = 1.0
        return pmf

    pob = sum(probs[k] for k in ONBASE)
    if pob <= 0:
        raise ValueError("On-base probability must be positive.")
    on = {k: probs[k] / pob for k in ONBASE}

    q_out = probs["OUT"] + probs["AO"]
    theta_ao = 0.0 if q_out == 0 else probs["AO"] / q_out

    types = [
        "BB", "1B_s", "1B_l", "2B_s", "2B_l", "3B", "HR", "ROE"
    ]
    w = {
        "BB": on["BB"],
        "1B_s": on["1B"] * (1 - theta_long_single),
        "1B_l": on["1B"] * theta_long_single,
        "2B_s": on["2B"] * (1 - theta_long_double),
        "2B_l": on["2B"] * theta_long_double,
        "3B": on["3B"],
        "HR": on["HR"],
        "ROE": on["ROE"]
    }

    def sim(ob_types: List[str], ao_bits: List[int]) -> int:
        first = second = third = None
        scored = set()

        def force():
            nonlocal first, second, third
            if first is not None and second is not None and third is not None:
                scored.add(third)
                third, second, first = second, first, None
            elif first is not None and second is not None:
                third, second, first = second, first, None
            elif first is not None:
                second, first = first, None

        def adv_all(n: int):
            nonlocal first, second, third
            if third is not None:
                if n >= 1:
                    scored.add(third)
                    third = None
            if second is not None:
                if n == 1:
                    third = second
                else:
                    scored.add(second)
                second = None
            if first is not None:
                if n == 1:
                    second = first
                elif n == 2:
                    third = first
                else:
                    scored.add(first)
                first = None

        ao_iter = iter(ao_bits)
        for idx, ev in enumerate(ob_types + ["OUT", "OUT", "OUT"]):
            if ev == "OUT":
                if next(ao_iter) == 1:
                    adv_all(1)
                continue
            if ev == "BB":
                force()
                first = idx
            elif ev in ("1B_s", "ROE"):
                adv_all(1)
                first = idx
            elif ev == "1B_l":
                adv_all(2)
                first = idx
            elif ev == "2B_s":
                adv_all(2)
                second = idx
            elif ev == "2B_l":
                adv_all(3)
                second = idx
            elif ev == "3B":
                adv_all(3)
                third = idx
            elif ev == "HR":
                if first is not None:
                    scored.add(first)
                    first = None
                if second is not None:
                    scored.add(second)
                    second = None
                if third is not None:
                    scored.add(third)
                    third = None
                scored.add(idx)
        return len(scored)

    L = 3 + m
    arrangements = list(itertools.combinations(range(L - 1), m))
    pmf = np.zeros(4)
    for ao_bits in itertools.product([0, 1], repeat=3):
        p_outpat = (
            (theta_ao if ao_bits[0] else (1 - theta_ao)) *
            (theta_ao if ao_bits[1] else (1 - theta_ao)) *
            (theta_ao if ao_bits[2] else (1 - theta_ao))
        )
        for ob_types in itertools.product(types, repeat=m):
            p_types = 1.0
            for ev in ob_types:
                p_types *= w[ev]
            r = sim(list(ob_types), list(ao_bits))
            pmf[r] += p_outpat * p_types
    pmf /= len(arrangements) if len(arrangements) > 0 else 1.0
    s = pmf.sum()
    if s > 0:
        pmf /= s
    return pmf

def inning_run_pmf_with(
    probs: Dict[str, float],
    theta_long_single: float = 0.6,
    theta_long_double: float = 0.9,
    eps: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the inning run PMF and batter-out distribution with SF and ROE adjustments.
    """
    q = probs["OUT"] + probs["AO"]
    p_end = p_end_distribution(q, eps=eps)
    aux_pmfs = [
        aux_pmf_m_with(probs, m, theta_long_single, theta_long_double)
        for m in range(4)
    ]
    max_b_eff = np.nonzero(p_end > eps)[0][-1] if np.any(p_end > eps) else len(p_end) - 1
    max_r = max(20, max_b_eff)
    pmf = np.zeros(max_r + 1)
    for b in range(3, max_b_eff + 1):
        if p_end[b] < 1e-16:
            continue
        m = max(0, min(3, b - 3))
        aut = max(0, b - 6)
        aux = aux_pmfs[m]
        for r_aux, p in enumerate(aux):
            r = aut + r_aux
            if r <= max_r:
                pmf[r] += p_end[b] * p
    s = pmf.sum()
    if s > 0:
        pmf /= s
    while len(pmf) > 1 and pmf[-1] < 1e-15:
        pmf = pmf[:-1]
    return pmf, p_end

def expected_runs_per_inning_with(
    probs: Dict[str, float],
    theta_long_single: float = 0.6,
    theta_long_double: float = 0.9,
    eps: float = 1e-12
) -> float:
    """
    Returns expected runs per inning given event probabilities, SF, and ROE.
    """
    pmf, _ = inning_run_pmf_with(probs, theta_long_single, theta_long_double, eps)
    return float(np.dot(np.arange(len(pmf)), pmf))

def expected_runs_per_game_with(
    probs: Dict[str, float],
    theta_long_single: float = 0.6,
    theta_long_double: float = 0.9,
    eps: float = 1e-12,
    innings: int = 9
) -> float:
    """
    Returns expected runs per game for given probabilities and settings.
    """
    return innings * expected_runs_per_inning_with(
        probs, theta_long_single, theta_long_double, eps
    )
