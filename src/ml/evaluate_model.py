from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)


DATA_DIR = Path("data/ml")


def main():

    print("\n" + "=" * 70)
    print("              RANDOM FOREST EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    model = joblib.load(
        DATA_DIR / "tuned_random_forest.joblib"
    )

    X_test = np.load(
        DATA_DIR / "X_test.npy"
    )

    y_test = np.load(
        DATA_DIR / "y_test.npy"
    )

    mapping = pd.read_csv(
        DATA_DIR / "label_mapping.csv"
    )

    mapping = mapping.sort_values(
        "label_id"
    )

    class_names = mapping["label"].tolist()

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"\nTest accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Classification report with real labels
    # --------------------------------------------------------

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=list(range(len(class_names)))
    )

    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    )

    cm_df.to_csv(
        DATA_DIR / "confusion_matrix.csv"
    )

    print(
        f"\nConfusion matrix saved to:"
        f"\n{DATA_DIR / 'confusion_matrix.csv'}"
    )

    # --------------------------------------------------------
    # Most confused classes
    # --------------------------------------------------------

    print("\nMost confused classes:")

    confusion_pairs = []

    for i in range(len(class_names)):

        for j in range(len(class_names)):

            if i != j and cm[i, j] > 0:

                confusion_pairs.append(
                    (
                        cm[i, j],
                        class_names[i],
                        class_names[j]
                    )
                )

    confusion_pairs.sort(
        reverse=True
    )

    for count, actual, predicted in confusion_pairs[:15]:

        print(
            f"  {actual:15s} -> "
            f"{predicted:15s}: {count}"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()