import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import equations as eq
from calibrate import load

S_DENSE = np.arange(32, 513, 16, dtype=float)
B_DENSE = np.unique(np.round(np.logspace(0, np.log10(256), 120))).astype(float)


def savefig(fig, d, name):
    fig.tight_layout()
    fig.savefig(os.path.join(d, name), dpi=150)
    plt.close(fig)


def sel_levels(values, k):
    values = sorted(set(values))
    idx = np.unique(np.round(np.linspace(0, len(values) - 1, k)).astype(int))
    return [values[i] for i in idx]


def curves(ok, df, pred_fn, col, ylabel, fname, d, scale=1.0, by="B", cap=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, x_name, l_name, dense in ((axes[0], "S", "B", S_DENSE), (axes[1], "B", "S", B_DENSE)):
        levels = sel_levels(df[l_name].unique(), 6)
        cmap = plt.cm.viridis(np.linspace(0, 0.9, len(levels)))
        for c, lv in zip(cmap, levels):
            xs = dense
            S, B = (xs, lv) if x_name == "S" else (lv, xs)
            ax.plot(xs, pred_fn(S, B) * scale, "-", color=c, lw=1.5, label=f"{l_name}={lv} (pred)")
            m = ok[ok[l_name] == lv]
            tr, va = m[m.is_validation == 0], m[m.is_validation == 1]
            ax.plot(tr[x_name], tr[col] * scale, "o", color=c, ms=6)
            ax.plot(va[x_name], va[col] * scale, "D", color=c, ms=6, mfc="white")
            o = df[(df[l_name] == lv) & df.oom]
            if len(o) and cap is not None:
                ax.plot(o[x_name], np.full(len(o), cap * scale), "x", color=c, ms=9, mew=2)
        if cap is not None:
            ax.axhline(cap * scale, color="red", ls="--", lw=1, label="GPU capacity")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("image size S [px]" if x_name == "S" else "batch size B")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].legend(fontsize=7, ncol=2)
    axes[0].set_title("lines: prediction, ●: measured (train), ◇: measured (validation), ×: OOM",
                      fontsize=9)
    savefig(fig, d, fname)


def pred_vs_meas(ok, pred, col, label, unit, fname, d, scale=1.0):
    fig, ax = plt.subplots(figsize=(6, 6))
    for flag, mk, lab in ((0, "o", "train"), (1, "D", "validation")):
        m = ok.is_validation.to_numpy() == flag
        ax.scatter(ok[col][m] * scale, pred[m] * scale, marker=mk, s=25, alpha=0.8, label=lab,
                   c=np.log2(ok.B[m]), cmap="viridis")
    lo = min(ok[col].min(), pred.min()) * scale * 0.8
    hi = max(ok[col].max(), pred.max()) * scale * 1.25
    xs = np.array([lo, hi])
    ax.plot(xs, xs, "k-", lw=1, label="y = x")
    ax.plot(xs, xs * 1.2, "k:", lw=0.8, label="±20%")
    ax.plot(xs, xs / 1.2, "k:", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(f"measured {label} [{unit}]"); ax.set_ylabel(f"predicted {label} [{unit}]")
    ax.set_title(f"{label}: predicted vs measured (colour = log2 B)")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    savefig(fig, d, fname)


def err_heatmap(ok, df, pred, col, label, fname, d):
    Ss, Bs = sorted(df.S.unique()), sorted(df.B.unique())
    grid = np.full((len(Ss), len(Bs)), np.nan)
    for (s, b), e in zip(zip(ok.S, ok.B), 100 * (pred - ok[col].to_numpy()) / ok[col].to_numpy()):
        grid[Ss.index(s), Bs.index(b)] = e
    fig, ax = plt.subplots(figsize=(11, 6))
    v = np.nanmax(np.abs(grid)) if np.isfinite(grid).any() else 1
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-min(v, 100), vmax=min(v, 100), aspect="auto")
    for i in range(len(Ss)):
        for j in range(len(Bs)):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=7)
            elif ((df.S == Ss[i]) & (df.B == Bs[j]) & df.oom).any():
                ax.text(j, i, "OOM", ha="center", va="center", fontsize=7, color="red")
    ax.set_xticks(range(len(Bs)), Bs); ax.set_yticks(range(len(Ss)), Ss)
    ax.set_xlabel("batch size B"); ax.set_ylabel("image size S [px]")
    ax.set_title(f"{label}: signed relative error (pred − meas)/meas [%]")
    fig.colorbar(im, ax=ax, label="%")
    savefig(fig, d, fname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/measurements.csv")
    ap.add_argument("--theta", default="results/theta.json")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load(args.csv)
    ok = df[~df.oom].reset_index(drop=True)
    theta = json.load(open(args.theta))
    th, the = theta["latency"], theta["energy"]
    try:
        cap = json.load(open(os.path.join(os.path.dirname(args.csv), "meta.json")))["gpu_total_memory_bytes"]
    except Exception:
        cap = None
    S, B = ok.S.to_numpy(float), ok.B.to_numpy(float)

    if ok.flops_counted.notna().any():
        curves(ok, df, eq.flops_matmul, "flops_counted", "conv+linear FLOPs", "flops_curves.png", args.out)
        pred_vs_meas(ok, eq.flops_matmul(S, B), "flops_counted", "FLOPs (conv+linear)", "FLOP",
                     "flops_pred_vs_meas.png", args.out)

    curves(ok, df, eq.memory, "memory", "peak allocated memory [MiB]", "memory_curves.png",
           args.out, scale=1 / 2**20, cap=cap)
    pred_vs_meas(ok, eq.memory(S, B), "memory", "peak memory", "MiB", "memory_pred_vs_meas.png",
                 args.out, 1 / 2**20)
    err_heatmap(ok, df, eq.memory(S, B), "memory", "memory", "memory_error_map.png", args.out)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    SS, BB = np.meshgrid(np.linspace(32, 512, 200), np.logspace(0, np.log10(256), 200))
    cs = ax.contourf(SS, BB, eq.memory(SS, BB) / 2**30, levels=20, cmap="Blues")
    fig.colorbar(cs, label="predicted peak memory [GiB]")
    if cap is not None:
        c = ax.contour(SS, BB, eq.memory(SS, BB), levels=[cap], colors="red")
        ax.clabel(c, fmt={cap: "predicted OOM"})
        b_oom = (cap - eq.memory(512, 0)) / (eq.memory(512, 1) - eq.memory(512, 0))
        ax.set_title(f"measured: ● ok, × OOM.  Predicted OOM at S=512 starts at B≈{b_oom:.0f}", fontsize=9)
    ax.scatter(ok.S, ok.B, c="k", s=12, label="measured ok")
    o = df[df.oom]
    ax.scatter(o.S, o.B, c="red", marker="x", s=50, label="measured OOM")
    ax.set_yscale("log", base=2); ax.set_xlabel("image size S [px]"); ax.set_ylabel("batch size B")
    ax.legend(loc="lower left")
    savefig(fig, args.out, "memory_oom_map.png")

    curves(ok, df, lambda s, b: eq.latency(s, b, th), "latency_s", "latency [ms]",
           "latency_curves.png", args.out, scale=1e3)
    pred_vs_meas(ok, eq.latency(S, B, th), "latency_s", "latency", "ms", "latency_pred_vs_meas.png",
                 args.out, 1e3)
    err_heatmap(ok, df, eq.latency(S, B, th), "latency_s", "latency", "latency_error_map.png", args.out)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    reg = eq.regime(SS, BB, th)
    ax.contourf(SS, BB, reg, levels=[-0.5, 0.5, 1.5, 2.5], colors=["#fde0dd", "#c6dbef", "#c7e9c0"])
    ax.contour(SS, BB, reg, levels=[0.5, 1.5], colors="k", linewidths=0.8)
    names = np.array(["launch-bound", "memory-bound", "compute-bound"])
    r_pts = eq.regime(S, B, th)
    t_meas = ok.latency_s.to_numpy()
    sc = ax.scatter(S, B, c=np.log10(t_meas * 1e3), cmap="magma", s=40, edgecolors="k")
    fig.colorbar(sc, label="measured latency, log10 [ms]")
    for i, n in enumerate(names):
        ax.plot([], [], "s", color=["#fde0dd", "#c6dbef", "#c7e9c0"][i], ms=10, label=f"predicted {n}")
    ax.set_yscale("log", base=2); ax.set_xlabel("image size S [px]"); ax.set_ylabel("batch size B")
    ax.set_title("Predicted regime (background) and measured latency (points)")
    ax.legend(loc="lower left", fontsize=8)
    savefig(fig, args.out, "latency_regimes.png")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    F = eq.flops(S, B)
    ax.scatter(F, F / t_meas / 1e12, c=np.log2(B), cmap="viridis", s=25, label="measured")
    for s in sel_levels(ok.S.unique(), 4):
        f = eq.flops(s, B_DENSE)
        ax.plot(f, f / eq.latency(s, B_DENSE, th) / 1e12, "-", lw=1, label=f"pred S={s}")
    ax.axhline(th["P"] / 1e12, color="r", ls="--", lw=1, label=f"fitted P = {th['P']/1e12:.2f} TFLOP/s")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("FLOPs per forward pass"); ax.set_ylabel("achieved throughput [TFLOP/s]")
    ax.set_title("Throughput vs work (colour = log2 B)"); ax.legend(fontsize=8); ax.grid(True, which="both", alpha=.3)
    savefig(fig, args.out, "latency_throughput.png")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    ls, lb = np.meshgrid(np.log2(np.linspace(32, 512, 40)), np.linspace(0, 8, 40))
    ax.plot_surface(ls, lb, np.log10(eq.latency(2**ls, 2**lb, th) * 1e3), cmap="viridis", alpha=0.6)
    ax.scatter(np.log2(S), np.log2(B), np.log10(t_meas * 1e3), c="r", s=10)
    ax.set_xlabel("log2 S"); ax.set_ylabel("log2 B"); ax.set_zlabel("log10 latency [ms]")
    ax.set_title("latency: surface = prediction, red = measured")
    savefig(fig, args.out, "latency_surface.png")

    curves(ok, df, lambda s, b: eq.energy(s, b, the), "energy_j", "energy per pass [J]",
           "energy_curves.png", args.out)
    pred_vs_meas(ok, eq.energy(S, B, the), "energy_j", "energy", "J", "energy_pred_vs_meas.png", args.out)
    err_heatmap(ok, df, eq.energy(S, B, the), "energy_j", "energy", "energy_error_map.png", args.out)

    print("figures written to", args.out)


if __name__ == "__main__":
    main()
