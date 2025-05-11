import torch
from torch.utils import benchmark


def profile_model(fn, desc, label, sub_label, min_run_time=5):
    res = benchmark.Timer(
        stmt='fn()',
        globals={"fn": fn},
        label=label,
        sub_label=sub_label,
        description=f"{desc}"
    ).blocked_autorange(min_run_time=min_run_time)
    return res


def bench_mps_mm():
    devices = ["mps", "cpu"]
    ks = [256, 1024, 2048]
    dtypes = [torch.float32, torch.float16]

    label = "matmul profile"
    results = []
    for k in ks:
        for dtype in dtypes:
            for device in devices:
                sub_label = f"{k}x{k}, {dtype}, {device}"
                a = torch.randn(k, k, device=device, dtype=dtype)
                b = torch.randn(k, k, device=device, dtype=dtype)

                def fn():
                    return a @ b
                desc = f"{device} {dtype} {k}x{k}"
                res = profile_model(fn, desc=desc, label=label, sub_label=sub_label)
                results.append(res)

    compare = benchmark.Compare(results)
    compare.trim_significant_figures()
    compare.colorize()
    compare.print()


if __name__ == "__main__":
    bench_mps_mm()
