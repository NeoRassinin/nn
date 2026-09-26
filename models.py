import torch.nn as nn

from equations import ARCH, USE_BN, HIDDEN, NUM_CLASSES


def build_model(use_bn: bool = USE_BN) -> nn.Sequential:
    layers = []
    for cin, cout, k, stride, pool in ARCH:
        layers.append(nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, bias=False))
        if use_bn:
            layers.append(nn.BatchNorm2d(cout))
        layers.append(nn.ReLU(inplace=True))
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    layers += [
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(ARCH[-1][1], HIDDEN),
        nn.ReLU(inplace=True),
        nn.Linear(HIDDEN, NUM_CLASSES),
    ]
    return nn.Sequential(*layers)


if __name__ == "__main__":
    import torch
    m = build_model().eval()
    print(m)
    n = sum(p.numel() for p in m.parameters())
    print("parameters:", n)
    with torch.inference_mode():
        for S in (32, 64, 224):
            y = m(torch.randn(2, 3, S, S))
            print(S, tuple(y.shape))
