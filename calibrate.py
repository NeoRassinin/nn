import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, nnls

import equations as eq


def load(path):
    df = pd.read_csv(path)
    df["memory"] = pd.to_numeric(df["memory"], errors="coerce")
    df["oom"] = df["status"].eq("OOM")
    return df


def rel_err(pred, meas):
    return (np.asarray(pred) - np.asarray(meas)) / np.asarray(meas)


def summary(pred, meas):
    e = np.abs(rel_err(pred, meas))
    return dict(mape=float(np.mean(e)), median_ape=float(np.median(e)), max_ape=float(np.max(e)), n=int(len(e)))


def fit_latency(S, B, T, fixed=None):
    fixed = fixed or {}
    free = [k for k in eq.THETA_KEYS if k not in fixed]
    inits = dict(P=[2e12, 6e12], BW=[1e11, 3e11], t_kernel=[2e-6, 8e-6], t_launch=[5e-6, 2e-5])

    def resid(logp):
        th = dict(fixed, **dict(zip(free, np.exp(logp))))
        return np.log(eq.latency(S, B, th)) - np.log(T)

    best = None
    for x0 in itertools.product(*[inits[k] for k in free]):
        r = least_squares(resid, np.log(x0), method="trf", loss="soft_l1", f_scale=0.1, max_nfev=4000)
        if best is None or r.cost < best.cost:
            best = r
    th = dict(fixed, **dict(zip(free, np.exp(best.x))))
    return {k: float(th[k]) for k in eq.THETA_KEYS}


def fit_energy(S, B, E, th_lat):
    T = eq.latency(S, B, th_lat)
    A = np.stack([T, eq.flops(S, B), eq.bytes_moved(S, B)], axis=1)
    A = A / E[:, None]                                
    scale = A.max(axis=0)
    coef, _ = nnls(A / scale, np.ones_like(E))
    coef = coef / scale
    return dict(P_static=float(coef[0]), e_flop=float(coef[1]), e_byte=float(coef[2]),
                latency=th_lat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/measurements.csv")
    ap.add_argument("--out", default="results/theta.json")
    args = ap.parse_args()

    df = load(args.csv)
    ok = df[~df.oom].copy()
    tr, va = ok[ok.is_validation == 0], ok[ok.is_validation == 1]
    arr = lambda d, c: d[c].to_numpy(dtype=float)

    try:
        meta = json.load(open(os.path.join(os.path.dirname(args.csv), "meta.json")))
    except FileNotFoundError:
        meta = {}
    fixed = {"BW": meta["bw_copy_bytes_per_s"]} if "bw_copy_bytes_per_s" in meta else {}
    th = fit_latency(arr(tr, "S"), arr(tr, "B"), arr(tr, "latency_s"), fixed)
    the = fit_energy(arr(tr, "S"), arr(tr, "B"), arr(tr, "energy_j"), th)

    metrics = {}
    for name, d in (("train", tr), ("validation", va), ("all", ok)):
        if len(d) == 0:
            continue
        S, B = arr(d, "S"), arr(d, "B")
        metrics[name] = dict(
            memory=summary(eq.memory(S, B), arr(d, "memory")),
            latency=summary(eq.latency(S, B, th), arr(d, "latency_s")),
            energy=summary(eq.energy(S, B, the), arr(d, "energy_j")),
        )
        fc = d["flops_counted"].to_numpy(dtype=float)
        if np.isfinite(fc).all():
            metrics[name]["flops_matmul_vs_torch_counter"] = summary(eq.flops_matmul(S, B), fc)

    reg = eq.regime(arr(ok, "S"), arr(ok, "B"), th)
    names = np.array(["launch", "memory", "compute"])
    metrics["regime_counts"] = {n: int((names[reg] == n).sum()) for n in names}
    for n in names:
        m = names[reg] == n
        if m.any():
            metrics[f"latency_mape_{n}"] = float(np.mean(np.abs(rel_err(
                eq.latency(arr(ok, "S")[m], arr(ok, "B")[m], th), arr(ok, "latency_s")[m]))))

    base = ok["mem_baseline_bytes"].to_numpy(float) - eq.input_bytes(arr(ok, "S"), arr(ok, "B"))
    metrics["memory_constant_check"] = dict(
        measured_baseline_minus_input_bytes_median=float(np.median(base)),
        assumed_weights_plus_cublas_bytes=float(eq.WEIGHT_BYTES + eq.CUBLAS_WS_BYTES))

    try:
        cap = meta["gpu_total_memory_bytes"]
        pred_oom = eq.memory(arr(df, "S"), arr(df, "B")) > cap
        metrics["oom"] = dict(measured=int(df.oom.sum()), predicted=int(pred_oom.sum()),
                              agree=int((pred_oom == df.oom.to_numpy()).sum()), total=int(len(df)))
    except Exception:
        pass

    if "gemm_fp32_flops_per_s" in meta:
        metrics["P_fit_over_gemm_peak"] = th["P"] / meta["gemm_fp32_flops_per_s"]
    out = dict(latency=th, energy=the, metrics=metrics, fixed_from_microbench=list(fixed),
               model="T=max(N_ops*t_launch, sum_l max(F_l/P, M_l/BW) + N_k*t_kernel); "
                     "E=P_static*T + e_flop*F + e_byte*M")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
