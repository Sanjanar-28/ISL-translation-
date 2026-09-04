from pathlib import Path

import numpy as np


INPUT_DIR = Path("data/ml")
OUTPUT_DIR = Path("data/ml_normalized")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_features(X):
    """
    Normalize each frame independently.

    Input:
        X shape = (videos, 20 * 126)

    Each frame:
        126 values = 42 landmarks × 3 coordinates

    For each frame:
        1. Use the first landmark as origin.
        2. Translate all landmarks relative to that origin.
        3. Scale by the maximum landmark distance.
    """

    X = X.copy()

    n_videos = X.shape[0]
    n_frames = 20
    values_per_frame = 126

    X = X.reshape(
        n_videos,
        n_frames,
        values_per_frame
    )

    X = X.reshape(
        n_videos,
        n_frames,
        42,
        3
    )

    normalized = np.zeros_like(X, dtype=np.float32)

    for v in range(n_videos):

        for f in range(n_frames):

            landmarks = X[v, f]

            # First landmark = reference point
            origin = landmarks[0].copy()

            # Translate
            relative = landmarks - origin

            # Calculate scale
            distances = np.linalg.norm(
                relative,
                axis=1
            )

            scale = np.max(distances)

            # Avoid division by zero
            if scale > 1e-8:
                relative = relative / scale

            normalized[v, f] = relative

    return normalized.reshape(
        n_videos,
        n_frames * values_per_frame
    )


def process_split(name):

    input_file = INPUT_DIR / f"X_{name}.npy"

    X = np.load(input_file)

    print(
        f"{name:12s}: "
        f"original={X.shape}",
        end=""
    )

    X_normalized = normalize_features(X)

    output_file = (
        OUTPUT_DIR / f"X_{name}.npy"
    )

    np.save(
        output_file,
        X_normalized
    )

    print(
        f"  normalized={X_normalized.shape}"
    )


def main():

    print("\n" + "=" * 70)
    print("              LANDMARK NORMALIZATION")
    print("=" * 70)

    for split in [
        "train",
        "validation",
        "test"
    ]:
        process_split(split)

    # Copy labels
    for split in [
        "train",
        "validation",
        "test"
    ]:

        labels = np.load(
            INPUT_DIR / f"y_{split}.npy"
        )

        np.save(
            OUTPUT_DIR / f"y_{split}.npy",
            labels
        )

    print("\nSaved normalized dataset to:")
    print(OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("              NORMALIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()