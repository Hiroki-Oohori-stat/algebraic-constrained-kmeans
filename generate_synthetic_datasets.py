from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_moons


RUN_SEED = 42
BLOBS_RANDOM_STATE = 170

DATASET_VARIANTS = [
    dict(tag="BLOBS-2", kind="blobs", k=2, counts=[5000, 5000], cluster_std=1.0),
    dict(tag="BLOBS-3", kind="blobs", k=3, counts=[3334, 3333, 3333], cluster_std=1.0),
    dict(tag="BLOBS-imbalance", kind="blobs", k=2, counts=[1000, 9000], cluster_std=1.0),
    dict(tag="moon-balance", kind="moons", k=2, counts=[5000, 5000], noise=0.1),
    dict(tag="moon-imbalance", kind="moons", k=2, counts=[1000, 9000], noise=0.1),
    dict(tag="XY-balance", kind="xy", k=2, counts=[5000, 5000]),
    dict(tag="XY-imbalance", kind="xy", k=2, counts=[1000, 9000]),
]


def generate_dataset(variant):
    kind = str(variant["kind"])
    k = int(variant["k"])
    counts = [int(x) for x in variant["counts"]]

    if kind == "blobs":
        rng_cent = np.random.RandomState(BLOBS_RANDOM_STATE)
        centers = rng_cent.uniform(-10.0, 10.0, size=(k, 2))
        X, y = make_blobs(
            n_samples=tuple(counts),
            centers=centers,
            n_features=2,
            cluster_std=variant["cluster_std"],
            shuffle=True,
            random_state=BLOBS_RANDOM_STATE,
        )

    elif kind == "moons":
        X, y = make_moons(
            n_samples=tuple(counts),
            shuffle=True,
            noise=float(variant.get("noise", 0.1)),
            random_state=RUN_SEED,
        )

    elif kind == "xy":
        rng = np.random.RandomState(RUN_SEED)
        A, B = [], []
        n_a, n_b = counts[0], counts[1]

        while (len(A) < n_a) or (len(B) < n_b):
            x = float(rng.rand())
            yv = float(rng.rand())
            if (x < yv) and (len(A) < n_a):
                A.append((x, yv, 0))
            elif (x >= yv) and (len(B) < n_b):
                B.append((x, yv, 1))

        arr = np.array(A + B, dtype=float)
        X = arr[:, :2]
        y = arr[:, 2].astype(int)

    else:
        raise ValueError(f"Unknown dataset kind: {kind}")

    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def save_dataset(variant, output_dir):
    X, y = generate_dataset(variant)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{variant['tag']}.data"
    arr_out = np.column_stack([X, y])
    pd.DataFrame(arr_out).to_csv(output_path, header=False, index=False)
    return output_path


def main():
    output_dir = Path(__file__).resolve().parent / "synthetic_datasets"
    for variant in DATASET_VARIANTS:
        output_path = save_dataset(variant, output_dir)
        print(f"[ok] {variant['tag']}: {output_path}")


if __name__ == "__main__":
    main()
