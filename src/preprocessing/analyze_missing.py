from pathlib import Path

import pandas as pd


# ==================================================
# CONFIGURATION
# ==================================================

INPUT_PATH = Path(
    "data/test_landmarks_normalized.csv"
)


# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv(INPUT_PATH)


# ==================================================
# COUNT HAND STATES
# ==================================================

both_hands = (
    df["left_detected"] &
    df["right_detected"]
)

left_only = (
    df["left_detected"] &
    ~df["right_detected"]
)

right_only = (
    ~df["left_detected"] &
    df["right_detected"]
)

no_hands = (
    ~df["left_detected"] &
    ~df["right_detected"]
)


# ==================================================
# REPORT
# ==================================================

print("\n")
print("=" * 60)
print("             HAND DETECTION ANALYSIS")
print("=" * 60)

print(
    f"\nTotal frames: "
    f"{len(df)}"
)

print(
    f"\nBoth hands detected: "
    f"{both_hands.sum()}"
)

print(
    f"Left hand only: "
    f"{left_only.sum()}"
)

print(
    f"Right hand only: "
    f"{right_only.sum()}"
)

print(
    f"No hands detected: "
    f"{no_hands.sum()}"
)


print("\nPercentages:")

total = len(df)

if total > 0:

    print(
        f"  Both hands: "
        f"{both_hands.sum() / total * 100:.2f}%"
    )

    print(
        f"  Left only: "
        f"{left_only.sum() / total * 100:.2f}%"
    )

    print(
        f"  Right only: "
        f"{right_only.sum() / total * 100:.2f}%"
    )

    print(
        f"  No hands: "
        f"{no_hands.sum() / total * 100:.2f}%"
    )

print("\n" + "=" * 60)