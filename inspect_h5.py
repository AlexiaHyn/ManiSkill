import h5py
import numpy as np

FILE = "record_dataset_BinFill.h5"

with h5py.File(FILE, "r") as f:
    print("=" * 60)
    print("FULL TREE")
    print("=" * 60)
    f.visititems(lambda name, obj: print(
        name, "->", type(obj).__name__,
        getattr(obj, "shape", ""),
        getattr(obj, "dtype", "")
    ))

    episodes = list(f.keys())
    print(f"\nTotal episodes: {len(episodes)}")
    print(f"First 5: {episodes[:5]}")

    ep = f[episodes[0]]
    print(f"\n{'=' * 60}")
    print(f"EPISODE: {episodes[0]}")
    print("=" * 60)

    # Setup
    if "setup" in ep:
        print("\n--- setup ---")
        for k in ep["setup"]:
            v = ep["setup"][k][()]
            print(f"  {k}: {v}")

    # First timestep
    timesteps = [k for k in ep.keys() if k.startswith("timestep_")]
    timesteps.sort(key=lambda x: int(x.split("_")[1]))
    print(f"\nTimesteps in episode: {len(timesteps)}, showing first: {timesteps[0]}")

    ts = ep[timesteps[0]]

    if "obs" in ts:
        print("\n--- obs ---")
        for k in ts["obs"]:
            v = ts["obs"][k]
            val = v[()]
            preview = val.flat[0] if val.size > 0 else val
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, first_val={preview}")

    if "action" in ts:
        print("\n--- action ---")
        for k in ts["action"]:
            v = ts["action"][k][()]
            print(f"  {k}: {v}")

    if "info" in ts:
        print("\n--- info ---")
        for k in ts["info"]:
            v = ts["info"][k][()]
            print(f"  {k}: {v}")
