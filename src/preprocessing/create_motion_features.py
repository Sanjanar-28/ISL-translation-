from pathlib import Path

import numpy as np


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path("data/ml")
OUTPUT_DIR = Path("data/ml_motion")

N_FRAMES = 30
FEATURES_PER_FRAME = 2520 // N_FRAMES


# ============================================================
# FEATURE CREATION
# ============================================================

def create_motion_features(X):
    """
    Convert landmark sequences into frame-to-frame motion features.

    Input:
        X -> (samples, 2520)

    Original landmark representation:
        30 frames × 84 landmark coordinates
        2520 = 30 × 84

    Motion representation:
        Difference between consecutive frames.

    There are 29 frame differences, so the final motion vector is
        29 × 84 = 2436

    To maintain the expected 5040-dimensional representation,
    both positive and absolute motion are combined.
    """

    print("Reshaping sequences...")

    n_samples = X.shape[0]

    X_seq = X.reshape(
        n_samples,
        N_FRAMES,
        FEATURES_PER_FRAME
    )

    print(f"Sequence shape: {X_seq.shape}")

    print("Calculating frame-to-frame motion...")

    motion = np.diff(
        X_seq,
        axis=1
    )

    print(f"Motion shape: {motion.shape}")

    # --------------------------------------------------------
    # Pad the first frame with zeros
    # This restores the sequence to 30 frames.
    # --------------------------------------------------------

    zero_frame = np.zeros(
        (
            n_samples,
            1,
            FEATURES_PER_FRAME
        ),
        dtype=np.float32
    )

    motion_padded = np.concatenate(
        [
            zero_frame,
            motion
        ],
        axis=1
    )

    print(
        f"Padded motion shape: "
        f"{motion_padded.shape}"
    )

    # --------------------------------------------------------
    # Flatten
    # 30 × 84 = 2520
    # --------------------------------------------------------

    signed_motion = motion_padded.reshape(
        n_samples,
        -1
    )

    # --------------------------------------------------------
    # Absolute motion
    # --------------------------------------------------------

    absolute_motion = np.abs(
        motion_padded
    ).reshape(
        n_samples,
        -1
    )

    # --------------------------------------------------------
    # Combine
    #
    # 2520 signed + 2520 absolute = 5040
    # --------------------------------------------------------

    motion_features = np.concatenate(
        [
            signed_motion,
            absolute_motion
        ],
        axis=1
    )

    return motion_features.astype(
        np.float32
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("        ALIGNED MOTION FEATURE CREATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check input files
    # --------------------------------------------------------

    required_files = [
        INPUT_DIR / "X_train.npy",
        INPUT_DIR / "X_validation.npy",
        INPUT_DIR / "X_test.npy",
        INPUT_DIR / "y_train.npy",
        INPUT_DIR / "y_validation.npy",
        INPUT_DIR / "y_test.npy",
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Missing required file:\n{file_path}"
            )

    print("\nAll required files found.")

    # --------------------------------------------------------
    # Load aligned original datasets
    # --------------------------------------------------------

    print("\nLoading aligned original datasets...")

    X_train = np.load(
        INPUT_DIR / "X_train.npy"
    )

    X_validation = np.load(
        INPUT_DIR / "X_validation.npy"
    )

    X_test = np.load(
        INPUT_DIR / "X_test.npy"
    )

    print("\nOriginal dataset shapes:")

    print(
        f"  Train:      "
        f"{X_train.shape}"
    )

    print(
        f"  Validation: "
        f"{X_validation.shape}"
    )

    print(
        f"  Test:       "
        f"{X_test.shape}"
    )

    # --------------------------------------------------------
    # Validate original feature size
    # --------------------------------------------------------

    for name, X in [
        ("Train", X_train),
        ("Validation", X_validation),
        ("Test", X_test),
    ]:

        if X.shape[1] != 2520:

            raise ValueError(
                f"{name} expected 2520 features, "
                f"found {X.shape[1]}"
            )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Create motion features
    # --------------------------------------------------------

    print("\nCreating TRAIN motion features...")

    X_train_motion = create_motion_features(
        X_train
    )

    print("\nCreating VALIDATION motion features...")

    X_validation_motion = create_motion_features(
        X_validation
    )

    print("\nCreating TEST motion features...")

    X_test_motion = create_motion_features(
        X_test
    )

    # --------------------------------------------------------
    # Validate shapes
    # --------------------------------------------------------

    expected_feature_count = 5040

    for name, X in [
        ("Train", X_train_motion),
        ("Validation", X_validation_motion),
        ("Test", X_test_motion),
    ]:

        if X.shape[1] != expected_feature_count:

            raise RuntimeError(
                f"{name} motion feature count mismatch.\n"
                f"Expected: {expected_feature_count}\n"
                f"Found: {X.shape[1]}"
            )

    # --------------------------------------------------------
    # Save features
    # --------------------------------------------------------

    print("\nSaving aligned motion datasets...")

    np.save(
        OUTPUT_DIR / "X_train.npy",
        X_train_motion
    )

    np.save(
        OUTPUT_DIR / "X_validation.npy",
        X_validation_motion
    )

    np.save(
        OUTPUT_DIR / "X_test.npy",
        X_test_motion
    )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("        ALIGNED MOTION FEATURES COMPLETE")
    print("=" * 70)

    print("\nMotion dataset:")

    print(
        f"  Train:      "
        f"{X_train_motion.shape}"
    )

    print(
        f"  Validation: "
        f"{X_validation_motion.shape}"
    )

    print(
        f"  Test:       "
        f"{X_test_motion.shape}"
    )

    print(
        "\nSample alignment:"
    )

    print(
        f"  Train:      "
        f"{X_train.shape[0]} -> "
        f"{X_train_motion.shape[0]}"
    )

    print(
        f"  Validation: "
        f"{X_validation.shape[0]} -> "
        f"{X_validation_motion.shape[0]}"
    )

    print(
        f"  Test:       "
        f"{X_test.shape[0]} -> "
        f"{X_test_motion.shape[0]}"
    )

    print("\nSaved files:")

    print(
        "  data/ml_motion/X_train.npy"
    )

    print(
        "  data/ml_motion/X_validation.npy"
    )

    print(
        "  data/ml_motion/X_test.npy"
    )

    print("\n" + "=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()