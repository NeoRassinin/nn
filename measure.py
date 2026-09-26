import argparse
import csv
import gc
import json
import math
import os
import platform
import threading
import time

import numpy as np
import torch

from models import build_model
import equations as eq

BASE_S = [32, 64, 128, 224, 256, 384, 512]
BASE_B = [1, 2, 4, 8, 16, 32, 64, 128, 256]

LAT_TARGET_S = 1.0       
LAT_MIN_REP, LAT_MAX_REP = 10, 300
ENERGY_MIN_S = 2.0       
ENERGY_MIN_REP = 5
WARMUP = 3

FIELDS = ["S", "B", "is_validation", "status", "latency_s", "memory", "energy_j",
          "latency_p10_s", "latency_p90_s", "n_rep", "mem_baseline_bytes",
          "n_energy", "power_avg_w", "flops_counted", "error_stage"]


def make_grid(seed):
    rng = np.random.default_rng(seed)
    s_pool = [s for s in range(32, 513, 16) if s not in BASE_S]
    b_pool = [b for b in range(1, 257) if b & (b - 1)]        
    s_rand = sorted(int(x) for x in rng.choice(s_pool, 4, replace=False))
    b_rand = sorted(int(x) for x in rng.choice(b_pool, 3, replace=False))
    return sorted(BASE_S + s_rand), sorted(BASE_B + b_rand), s_rand, b_rand


class EnergyMeter:

    def __init__(self, index=0):
        import pynvml
        self.nv = pynvml
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(index)
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h)
            self.method = "nvml_total_energy_counter"
        except pynvml.NVMLError:
            self.method = "power_sampling_10ms"

    def start(self):
        if self.method == "nvml_total_energy_counter":
            self.e0 = self.nv.nvmlDeviceGetTotalEnergyConsumption(self.h)   # mJ
        else:
            self.samples, self._stop = [], False
            def loop():
                while not self._stop:
                    self.samples.append((time.perf_counter(),
                                         self.nv.nvmlDeviceGetPowerUsage(self.h) / 1000.0))
                    time.sleep(0.01)
            self.th = threading.Thread(target=loop, daemon=True)
            self.th.start()

    def stop(self):
        if self.method == "nvml_total_energy_counter":
            return (self.nv.nvmlDeviceGetTotalEnergyConsumption(self.h) - self.e0) / 1000.0
        self._stop = True
        self.th.join()
        t = np.array([s[0] for s in self.samples]); p = np.array([s[1] for s in self.samples])
        trapz = getattr(np, "trapezoid", None) or np.trapz
        return float(trapz(p, t)) if len(t) > 1 else float("nan")



def sync():
    torch.cuda.synchronize()


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def time_runs(model, x, n):
    ts = []
    for _ in range(n):
        sync()
        t0 = time.perf_counter()
        model(x)
        sync()
        ts.append(time.perf_counter() - t0)
    return np.array(ts)


def count_flops(S, B):

    try:
        from torch.utils.flop_counter import FlopCounterMode
        m = build_model().to("meta").eval()
        with torch.inference_mode(), FlopCounterMode(display=False) as fc:
            m(torch.empty(B, 3, S, S, device="meta"))
        return float(fc.get_total_flops())
    except Exception:
        return float("nan")


def measure_one(model, meter, S, B):
    row = dict(S=S, B=B, flops_counted=count_flops(S, B))
    stage = "alloc_input"
    x = None
    try:
        x = torch.randn(B, 3, S, S, device="cuda")
        stage = "warmup"
        for _ in range(WARMUP):
            model(x)
        sync()


        stage = "memory"
        cleanup()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        y = model(x)
        sync()
        peak = torch.cuda.max_memory_allocated()
        del y
        row.update(memory=peak, mem_baseline_bytes=base)

        stage = "latency"
        t_est = float(np.median(time_runs(model, x, 3)))
        n = int(np.clip(LAT_TARGET_S / max(t_est, 1e-6), LAT_MIN_REP, LAT_MAX_REP))
        ts = time_runs(model, x, n)
        row.update(latency_s=float(np.median(ts)), latency_p10_s=float(np.percentile(ts, 10)),
                   latency_p90_s=float(np.percentile(ts, 90)), n_rep=n)

        stage = "energy"
        ne = max(ENERGY_MIN_REP, math.ceil(ENERGY_MIN_S / row["latency_s"]))
        sync()
        meter.start()
        t0 = time.perf_counter()
        time_runs(model, x, ne)
        wall = time.perf_counter() - t0
        e = meter.stop()
        row.update(energy_j=e / ne, n_energy=ne, power_avg_w=e / wall)
        row["status"] = "ok"
    except torch.cuda.OutOfMemoryError:
        row.update(status="OOM", memory="OOM", error_stage=stage)
    finally:
        del x
        cleanup()
    return row


def microbench():

    n = 64 * 2**20                          
    a = torch.empty(n, device="cuda"); b = torch.empty_like(a)
    for _ in range(3):
        b.copy_(a)
    ts = []
    for _ in range(20):
        sync(); t0 = time.perf_counter(); b.copy_(a); sync(); ts.append(time.perf_counter() - t0)
    bw = 2 * n * 4 / float(np.median(ts))          
    del a, b
    m = 4096
    x = torch.randn(m, m, device="cuda"); y = torch.randn(m, m, device="cuda")
    for _ in range(3):
        x @ y
    ts = []
    for _ in range(10):
        sync(); t0 = time.perf_counter(); x @ y; sync(); ts.append(time.perf_counter() - t0)
    p = 2 * m**3 / float(np.median(ts))
    del x, y
    cleanup()
    return dict(bw_copy_bytes_per_s=bw, gemm_fp32_flops_per_s=p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/measurements.csv")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--quick", action="store_true", help="3x3 smoke test")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.is_available(), "no GPU: Runtime -> Change runtime type -> T4 GPU"

    sizes, batches, s_rand, b_rand = make_grid(args.seed)
    if args.quick:
        sizes, batches = [32, 224, 512], [1, 16, 256]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model = build_model().cuda().eval()
    meter = EnergyMeter()

    props = torch.cuda.get_device_properties(0)
    meta = dict(gpu=props.name, gpu_total_memory_bytes=props.total_memory,
                sm_count=props.multi_processor_count,
                torch=torch.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), python=platform.python_version(),
                driver=meter.nv.nvmlSystemGetDriverVersion(),
                energy_method=meter.method, seed=args.seed,
                sizes=sizes, batches=batches, s_random=s_rand, b_random=b_rand,
                use_bn=eq.USE_BN)
    meta.update(microbench())
    with open(os.path.join(os.path.dirname(args.out), "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(json.dumps(meta, indent=2, default=str))

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {(int(r["S"]), int(r["B"])) for r in csv.DictReader(f)}
    new_file = not os.path.exists(args.out)

    with torch.inference_mode():
        for _ in range(5):
            model(torch.randn(2, 3, 64, 64, device="cuda"))
        sync()

        with open(args.out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                w.writeheader()
            total = len(sizes) * len(batches)
            k = 0
            for S in sizes:
                for B in batches:
                    k += 1
                    if (S, B) in done:
                        continue
                    row = measure_one(model, meter, S, B)
                    row["is_validation"] = int(S in s_rand or B in b_rand)
                    w.writerow(row)
                    f.flush()
                    if row["status"] == "ok":
                        print(f"[{k:3d}/{total}] S={S:3d} B={B:3d}  "
                              f"t={row['latency_s']*1e3:9.3f} ms  mem={row['memory']/2**20:9.1f} MiB  "
                              f"E={row['energy_j']:.4f} J  (pred mem {eq.memory(S, B)/2**20:9.1f} MiB)")
                    else:
                        print(f"[{k:3d}/{total}] S={S:3d} B={B:3d}  OOM at {row['error_stage']}  "
                              f"(pred mem {eq.memory(S, B)/2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
