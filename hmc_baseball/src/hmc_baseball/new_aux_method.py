#!/usr/bin/env python3
import pandas as pd, numpy as np, math, json, os
from itertools import product

def load_league_mix(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    def col(name, default=0):
        return df[cols[name.lower()]].fillna(0) if name.lower() in cols else pd.Series([default]*len(df))
    PA,H = col("PA"), col("H")
    BB,HBP = col("BB"), col("HBP")
    D2,D3,HR = col("2B"), col("3B"), col("HR")
    S1 = (H - D2 - D3 - HR).clip(lower=0)
    PA_tot = float(PA.sum())
    p = {
        "BB": float((BB.sum()+HBP.sum()))/PA_tot,
        "1B": float(S1.sum())/PA_tot,
        "2B": float(D2.sum())/PA_tot,
        "3B": float(D3.sum())/PA_tot,
        "HR": float(HR.sum())/PA_tot,
    }
    p["OUT"] = 1.0 - sum(p.values())
    return p

def advancement_runs(b1,b2,b3,e):
    if e in ("BB","1B_S"):
        runs = 1 if b3 else 0
        return 1, b1, b2, runs
    if e=="1B_L":
        runs = (1 if b3 else 0) + (1 if b2 else 0)
        return 1, 0, b1, runs
    if e=="2B_S":
        runs = (1 if b3 else 0) + (1 if b2 else 0)
        return 0, 1, b1, runs
    if e=="2B_L":
        runs = (1 if b3 else 0) + (1 if b2 else 0) + (1 if b1 else 0)
        return 0, 1, 0, runs
    if e=="3B":
        runs = (1 if b1 else 0) + (1 if b2 else 0) + (1 if b3 else 0) + 0
        return 0, 0, 1, runs
    if e=="HR":
        runs = (1 if b1 else 0) + (1 if b2 else 0) + (1 if b3 else 0) + 1
        return 0, 0, 0, runs
    raise ValueError

def enumerate_b(q):
    # q is dict over non-outs
    evs = list(q.keys()); ps = [q[e] for e in evs]
    # b1
    b1 = 0.0
    for i,e1 in enumerate(evs):
        p=ps[i]; a1,a2,a3=0,0,0
        a1,a2,a3,r = advancement_runs(a1,a2,a3,e1)
        b1 += p*r
    # b2
    b2 = 0.0
    for i,e1 in enumerate(evs):
        for j,e2 in enumerate(evs):
            p=ps[i]*ps[j]; a1,a2,a3=0,0,0
            a1,a2,a3,r1 = advancement_runs(a1,a2,a3,e1)
            a1,a2,a3,r2 = advancement_runs(a1,a2,a3,e2)
            b2 += p*(r1+r2)
    # b3
    b3 = 0.0
    for i,e1 in enumerate(evs):
        for j,e2 in enumerate(evs):
            for k,e3 in enumerate(evs):
                p=ps[i]*ps[j]*ps[k]; a1,a2,a3=0,0,0
                a1,a2,a3,r1 = advancement_runs(a1,a2,a3,e1)
                a1,a2,a3,r2 = advancement_runs(a1,a2,a3,e2)
                a1,a2,a3,r3 = advancement_runs(a1,a2,a3,e3)
                b3 += p*(r1+r2+r3)
    return b1,b2,b3

def bucket_weights(p_out):
    f0 = p_out**3
    f1 = 3*(p_out**3)*(1-p_out)
    f2 = 6*(p_out**3)*(1-p_out)**2
    fge3 = 1 - f0 - f1 - f2
    return f0,f1,f2,fge3

def pout(i,j,ell,p_out,p_on):
    t = j - i
    if t<0 or ell>2: return 0.0
    return math.comb(t,ell) * (p_out**ell) * (p_on**(t-ell))

def f_ijk(i,j,k,p_out,p_on):
    if j==k-5: return pout(i,j,0,p_out,p_on)*p_on * pout(j+1,k,2,p_out,p_on)*p_out
    if j==k-4: return pout(i,j,1,p_out,p_on)*p_on * pout(j+1,k,1,p_out,p_on)*p_out
    if j==k-3: return pout(i,j,2,p_out,p_on)*p_on * pout(j+1,k,0,p_out,p_on)*p_out
    return 0.0

def aux_ge3_via_memo(i, p_out, p_on, b3, k_max=17):
    tot = 0.0
    for k in range(i+3, i+k_max+1):
        for j in range(k-5, k-2):
            L = k - j - 1
            # conditional probability 1/C(L,2) for (j2,j3) given exactly L-2 outs
            if L>=2:
                aux_jk = b3  # uniform average over positions
            else:
                aux_jk = 0.0
            tot += f_ijk(i,j,k,p_out,p_on) * aux_jk
    return tot

def main(csv_path="2024_baseball_stats.csv", target_rg=4.40, out_dir="./artifacts_doc_exact"):
    os.makedirs(out_dir, exist_ok=True)
    p = load_league_mix(csv_path)
    lam1, lam2 = 0.15, 0.25  
    p1S, p1L = (1-lam1)*p["1B"], lam1*p["1B"]
    p2S, p2L = (1-lam2)*p["2B"], lam2*p["2B"]
    p_out = p["OUT"]; p_on = 1-p_out
    nonout = {"BB":p["BB"],"1B_S":p1S,"1B_L":p1L,"2B_S":p2S,"2B_L":p2L,"3B":p["3B"],"HR":p["HR"]}
    Z = sum(nonout.values())
    q = {k:v/Z for k,v in nonout.items()}
    b1,b2,b3 = enumerate_b(q)
    f0,f1,f2,fge3 = bucket_weights(p_out)
    # Automatic runs
    E_auto = 0.0
    for kk in range(4,400):
        E_auto += (kk-3)*math.comb(kk+2,kk)*((1-p_out)**kk)*(p_out**3)
    # Suffix sum (>=3 arrivals)
    E_aux_ge3 = aux_ge3_via_memo(i=0, p_out=p_out, p_on=p_on, b3=b3, k_max=17)
    E_runs_inning = E_auto + f1*b1 + f2*b2 + E_aux_ge3
    E_runs_game = (innings*E_runs_inning) + sfroe
    with open(os.path.join(out_dir,"summary.json"),"w") as f:
        json.dump({
            "lam1":lam1,"lam2":lam2,
            "p_out":p_out,"p_on":p_on,
            "b1":b1,"b2":b2,"b3":b3,
            "f0":f0,"f1":f1,"f2":f2,"fge3":fge3,
            "E_auto":E_auto,"E_aux_ge3":E_aux_ge3,
            "E_runs_per_inning":E_runs_inning,
            "E_runs_per_game":E_runs_game
        }, f, indent=2)
    print("Done. E[r/g] =", round(E_runs_game,6))

main()
