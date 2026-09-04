from pathlib import Path

import numpy as np


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path("data/ml")
OUTPUT_DIR = Path("data/ml_augmented")

NUM_AUGMENTED_COPIES = 3

RANDOM_SEED = 42


# ============================================================
# AUGMENTATION
# ============================================================

def augment_sample(sample, rng):
    """
    sample shape:
        (20, 126)

    Returns:
        augmented sample with same shape.
    """

    x = sample.copy()

    # --------------------------------------------------------
    # Reshape 126 features into:
    #
    # 42 landmarks × 3 coordinates
    #
    # MediaPipe layout:
    # x, y, z
    # --------------------------------------------------------

    x = x.reshape(20, 42, 3)

    # --------------------------------------------------------
    # Small spatial scaling
    # --------------------------------------------------------

    scale = rng.uniform(0.97, 1.03)

    x[:, :, :2] *= scale

    # --------------------------------------------------------
    # Small translation
    # --------------------------------------------------------

    translation = rng.normal(
        loc=0.0,
        scale=0.005,
        size=(1, 1, 2)
    )

    x[:, :, :2] += translation

    # --------------------------------------------------------
    # Very small landmark noise
    # --------------------------------------------------------

    noise = rng.normal(
        loc=0.0,
        scale=0.002,
        size=x[:, :, :2].shape
    )

    x[:, :, :2] += noise

    # --------------------------------------------------------
    # Keep Z noise smaller
    # --------------------------------------------------------

    z_noise = rng.normal(
        loc=0.0,
        scale=0.001,
        size=x[:, :, 2:3].shape
    )

    x[:, :, 2:3] += z_noise

    return x.reshape(2520)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("              LANDMARK AUGMENTATION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load original training data
    # --------------------------------------------------------

    X_train = np.load(
        INPUT_DIR / "X_train.npy"
    )

    y_train = np.load(
        INPUT_DIR / "y_train.npy"
    )

    print(
        f"\nOriginal X_train: {X_train.shape}"
    )

    print(
        f"Original y_train: {y_train.shape}"
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if X_train.shape[1] != 2520:
        raise RuntimeError(
            f"Expected 2520 features, "
            f"got {X_train.shape[1]}"
        )

    if len(X_train) != len(y_train):
        raise RuntimeError(
            "X_train and y_train sizes do not match."
        )

    # --------------------------------------------------------
    # Random generator
    # --------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    augmented_X = []
    augmented_y = []

    # --------------------------------------------------------
    # Generate augmented samples
    # --------------------------------------------------------

    for i in range(len(X_train)):

        sample = X_train[i]

        for _ in range(
            NUM_AUGMENTED_COPIES
        ):

            new_sample = augment_sample(
                sample,
                rng
            )

            augmented_X.append(
                new_sample
            )

            augmented_y.append(
                y_train[i]
            )

    augmented_X = np.asarray(
        augmented_X,
        dtype=np.float32
    )

    augmented_y = np.asarray(
        augmented_y,
        dtype=np.int64
    )

    # --------------------------------------------------------
    # Combine original + augmented
    # --------------------------------------------------------

    X_final = np.concatenate(
        [
            X_train,
            augmented_X
        ],
        axis=0
    )

    y_final = np.concatenate(
        [
            y_train,
            augmented_y
        ],
        axis=0
    )

    # --------------------------------------------------------
    # Shuffle training data
    # --------------------------------------------------------

    indices = rng.permutation(
        len(X_final)
    )

    X_final = X_final[
        indices
    ]

    y_final = y_final[
        indices
    ]

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.save(
        OUTPUT_DIR / "X_train.npy",
        X_final
    )

    np.save(
        OUTPUT_DIR / "y_train.npy",
        y_final
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\nAugmentation complete.")

    print(
        f"Original samples:   {len(X_train)}"
    )

    print(
        f"Augmented samples:   {len(augmented_X)}"
    )

    print(
        f"Final train samples: {len(X_final)}"
    )

    print(
        f"Features:            {X_final.shape[1]}"
    )

    print(
        "\nSaved:"
    )

    print(
        "  data/ml_augmented/X_train.npy"
    )

    print(
        "  data/ml_augmented/y_train.npy"
    )

    print("\n" + "=" * 70)
    print("              AUGMENTATION COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()