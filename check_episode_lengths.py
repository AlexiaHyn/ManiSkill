"""Quick script to report timestep counts per episode in a BinFill h5 file."""
import argparse
import h5py
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5_file", default="record_dataset_BinFill.h5")
    args = p.parse_args()

    with h5py.File(args.h5_file, "r") as f:
        ep_keys = sorted(f.keys())
        lengths = []
        for k in ep_keys:
            n = sum(1 for ts in f[k].keys() if ts.startswith("timestep_"))
            lengths.append((k, n))

    steps = np.array([l for _, l in lengths])
    print(f"Total episodes : {len(steps)}")
    print(f"Min  steps     : {steps.min()}")
    print(f"Max  steps     : {steps.max()}")
    print(f"Mean steps     : {steps.mean():.1f}")
    print(f"Median steps   : {np.median(steps):.1f}")
    print(f"Std  steps     : {steps.std():.1f}")

    print(f"\nHistogram (bucket = 100 steps):")
    bucket = 100
    hi = (int(steps.max()) // bucket + 1) * bucket
    counts, edges = np.histogram(steps, bins=range(0, hi + bucket, bucket))
    for i, c in enumerate(counts):
        bar = "#" * c
        print(f"  {edges[i]:5d} – {edges[i+1]:5d}: {bar} ({c})")

    print(f"\nPer-episode table:")
    print(f"{'Episode':<12}  {'Steps':>6}")
    print("-" * 22)
    for ep_key, n in lengths:
        print(f"{ep_key:<12}  {n:>6}")

if __name__ == "__main__":
    main()
