import h5py
import numpy as np

FILE = "record_dataset_BinFill.h5"

def print_group(group, indent=4):
    pad = " " * indent
    for k in group:
        item = group[k]
        if isinstance(item, h5py.Dataset):
            val = item[()]
            preview = val.flat[0] if hasattr(val, "flat") and val.size > 0 else val
            print(f"{pad}{k}: shape={item.shape}, dtype={item.dtype}, first_val={preview}")
        elif isinstance(item, h5py.Group):
            print(f"{pad}{k}/  (subgroup)")
            print_group(item, indent + 4)

with h5py.File(FILE, "r") as f:
    episodes = list(f.keys())
    print(f"Total episodes: {len(episodes)}")

    ep = f[episodes[0]]
    print(f"\n{'=' * 60}")
    print(f"EPISODE: {episodes[0]}")
    print("=" * 60)

    # Setup
    if "setup" in ep:
        print("\n--- setup ---")
        print_group(ep["setup"])

    # First 10 timesteps
    timesteps = [k for k in ep.keys() if k.startswith("timestep_")]
    timesteps.sort(key=lambda x: int(x.split("_")[1]))
    print(f"\nTotal timesteps: {len(timesteps)}, printing first 10")

    for ts_key in timesteps[:10]:
        ts = ep[ts_key]
        print(f"\n--- {ts_key} ---")

        for section in ("obs", "action", "info"):
            if section in ts:
                print(f"  {section}:")
                print_group(ts[section], indent=4)
