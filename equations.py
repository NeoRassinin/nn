import numpy as np


ARCH = [
    (3,   32,  7, 2, True),    # conv7x7 s2 -> S/2, maxpool 3x3 s2 p1 -> S/4
    (32,  64,  5, 1, False),   # S/4
    (64,  128, 3, 2, False),   # S/8
    (128, 256, 1, 1, False),   # S/8
    (256, 256, 3, 2, False),   # S/16
    (256, 512, 1, 1, False),   # S/16
]
USE_BN = True          
HIDDEN = 256
NUM_CLASSES = 100

BYTES = 4              # FP32
ALLOC_GRAN = 512       

CUBLAS_WS_BYTES = 4096 * 1024 * 2 + 16 * 1024 * 8

FLOP_BN, FLOP_RELU, FLOP_POOL = 2, 1, 8


def _arr(x):
    return np.asarray(x, dtype=np.float64)


def _out(x):
    x = np.asarray(x, dtype=np.float64)
    return float(x) if x.ndim == 0 else x


def _r(nbytes):
    return np.ceil(_arr(nbytes) / ALLOC_GRAN) * ALLOC_GRAN


def layers(S, B):

    S, B = _arr(S), _arr(B)
    L = []
    h, c = S, 3
    for i, (cin, cout, k, s, pool) in enumerate(ARCH, 1):
        ho = h / s
        n_in = B * cin * h * h
        n_out = B * cout * ho * ho
        w = cin * cout * k * k
        L.append(dict(name=f"conv{i}", kind="conv",
                      flops=2.0 * B * cin * cout * k * k * ho * ho,
                      bytes=BYTES * (n_in + w + n_out),
                      n_out=n_out, alloc=True, kernel=True))
        if USE_BN:
            L.append(dict(name=f"bn{i}", kind="bn",
                          flops=FLOP_BN * n_out,
                          bytes=BYTES * (2 * n_out + 4 * cout),
                          n_out=n_out, alloc=True, kernel=True))
        L.append(dict(name=f"relu{i}", kind="relu",           # inplace
                      flops=FLOP_RELU * n_out,
                      bytes=BYTES * 2 * n_out,
                      n_out=n_out, alloc=False, kernel=True))
        h, c = ho, cout
        if pool:
            hp = h / 2
            n_p = B * c * hp * hp
            L.append(dict(name="maxpool", kind="pool",
                          flops=FLOP_POOL * n_p,
                          bytes=BYTES * (n_out + n_p),
                          n_out=n_p, alloc=True, kernel=True))
            h = hp
    n_in = B * c * h * h
    L.append(dict(name="gap", kind="gap", flops=n_in + B * c,
                  bytes=BYTES * (n_in + B * c), n_out=B * c, alloc=True, kernel=True))
    L.append(dict(name="flatten", kind="view", flops=0.0 * B, bytes=0.0 * B,
                  n_out=B * c, alloc=False, kernel=False))
    L.append(dict(name="fc1", kind="linear",
                  flops=2.0 * B * c * HIDDEN + B * HIDDEN,
                  bytes=BYTES * (B * c + c * HIDDEN + HIDDEN + B * HIDDEN),
                  n_out=B * HIDDEN, alloc=True, kernel=True))
    L.append(dict(name="relu_fc", kind="relu", flops=FLOP_RELU * B * HIDDEN,
                  bytes=BYTES * 2 * B * HIDDEN, n_out=B * HIDDEN, alloc=False, kernel=True))
    L.append(dict(name="fc2", kind="linear",
                  flops=2.0 * B * HIDDEN * NUM_CLASSES + B * NUM_CLASSES,
                  bytes=BYTES * (B * HIDDEN + HIDDEN * NUM_CLASSES + NUM_CLASSES + B * NUM_CLASSES),
                  n_out=B * NUM_CLASSES, alloc=True, kernel=True))
    return L


def n_ops():
    return len(layers(64, 1))


def n_kernels():
    return sum(l["kernel"] for l in layers(64, 1))

def flops(image_size, batch):
    return _out(sum(l["flops"] for l in layers(image_size, batch)))


def flops_matmul(image_size, batch):
    S, B = _arr(image_size), _arr(batch)
    tot = 0.0
    for l in layers(S, B):
        if l["kind"] == "conv":
            tot = tot + l["flops"]
    tot = tot + 2.0 * B * (ARCH[-1][1] * HIDDEN + HIDDEN * NUM_CLASSES)
    return _out(tot)


def bytes_moved(image_size, batch):
    return _out(sum(l["bytes"] for l in layers(image_size, batch)))


def _weight_tensor_bytes():
    t = []
    for cin, cout, k, s, pool in ARCH:
        t.append(cin * cout * k * k * BYTES)
        if USE_BN:
            t += [cout * BYTES] * 4 + [8]       
    c = ARCH[-1][1]
    t += [c * HIDDEN * BYTES, HIDDEN * BYTES, HIDDEN * NUM_CLASSES * BYTES, NUM_CLASSES * BYTES]
    return t


N_PARAMS_AND_BUFFERS = sum(b for b in _weight_tensor_bytes())
WEIGHT_BYTES = float(sum(_r(b) for b in _weight_tensor_bytes()))


def input_bytes(image_size, batch):
    S, B = _arr(image_size), _arr(batch)
    return _out(_r(BYTES * B * 3 * S * S))


def memory_live_per_layer(image_size, batch):
    S, B = _arr(image_size), _arr(batch)
    live = []
    cur = None                    
    for l in layers(S, B):
        cur_b = 0.0 if cur is None else _r(BYTES * cur)
        out_b = _r(BYTES * l["n_out"]) if l["alloc"] else 0.0
        live.append((l["name"], cur_b + out_b))
        if l["alloc"]:
            cur = l["n_out"]
    return live


def memory(image_size, batch):
    S, B = _arr(image_size), _arr(batch)
    live = [v for _, v in memory_live_per_layer(S, B)]
    peak_transient = np.maximum.reduce([np.broadcast_to(v, np.broadcast(S, B).shape) for v in live])
    return _out(WEIGHT_BYTES + CUBLAS_WS_BYTES + input_bytes(S, B) + peak_transient)


THETA_KEYS = ["P", "BW", "t_kernel", "t_launch"]


def latency_terms(image_size, batch, theta):
    S, B = _arr(image_size), _arr(batch)
    P, BW = theta["P"], theta["BW"]
    tk, tl = theta["t_kernel"], theta["t_launch"]
    t_comp = t_mem = 0.0
    nk = 0
    for l in layers(S, B):
        if not l["kernel"]:
            continue
        nk += 1
        tc, tm = l["flops"] / P, l["bytes"] / BW
        t_comp = t_comp + np.where(tc >= tm, tc, 0.0)
        t_mem = t_mem + np.where(tc < tm, tm, 0.0)
    gpu = t_comp + t_mem + nk * tk
    cpu = n_ops() * tl + 0.0 * (S + B)
    return dict(cpu=cpu, gpu=gpu, compute=t_comp, memory=t_mem, fixed=nk * tk)


def latency(image_size, batch, theta):
    t = latency_terms(image_size, batch, theta)
    return _out(np.maximum(t["cpu"], t["gpu"]))


def regime(image_size, batch, theta):
    t = latency_terms(image_size, batch, theta)
    launch = t["cpu"] >= t["gpu"]
    return np.where(launch, 0, np.where(t["memory"] >= t["compute"], 1, 2))


def energy(image_size, batch, theta_energy):
    S, B = _arr(image_size), _arr(batch)
    T = latency(S, B, theta_energy["latency"])
    return _out(theta_energy["P_static"] * T
                + theta_energy["e_flop"] * _arr(flops(S, B))
                + theta_energy["e_byte"] * _arr(bytes_moved(S, B)))


if __name__ == "__main__":
    print("params+buffers bytes:", N_PARAMS_AND_BUFFERS, " allocated:", WEIGHT_BYTES)
    print("ops:", n_ops(), " kernels:", n_kernels())
    for S in (32, 224, 512):
        print(f"S={S}: FLOPs/B = {flops(S, 1):.4e}, bytes/B = {bytes_moved(S, 1):.4e}, "
              f"mem(B=1) = {memory(S, 1)/2**20:.1f} MiB, mem(B=256) = {memory(S, 256)/2**30:.2f} GiB")
    for name, v in memory_live_per_layer(224, 1):
        print(f"  {name:8s} {v/2**20:8.2f} MiB")
