from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)


DATA_DIR = Path("data/ml_normalized")


def main():

    print("\n" + "=" * 70)
    print("        NORMALIZED RANDOM FOREST EVALUATION")
    print("=" * 70)

    model = joblib.load(
        DATA_DIR / "normalized_random_forest.joblib"
    )

    X_test = np.load(
        DATA_DIR / "X_test.npy"
    )

    y_test = np.load(
        DATA_DIR / "y_test.npy"
    )

    # Load class mapping
    mapping = pd.read_csv(
        "data/ml/label_mapping.csv"
    )

    mapping = mapping.sort_values("label_id")

    class_names = mapping["label"].tolist()

    # Predictions
    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"\nTest accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    # Classification report
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

    # Confusion matrix
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

    output_file = (
        DATA_DIR /
        "normalized_confusion_matrix.csv"
    )

    cm_df.to_csv(output_file)

    print(
        f"\nConfusion matrix saved to:"
        f"\n{output_file}"
    )

    # Find mistakes
    confusion_pairs = []

    for actual in range(len(class_names)):

        for predicted in range(len(class_names)):

            if actual != predicted and cm[actual, predicted] > 0:

                confusion_pairs.append(
                    (
                        cm[actual, predicted],
                        class_names[actual],
                        class_names[predicted]
                    )
                )

    confusion_pairs.sort(reverse=True)

    print("\nMost confused classes:")

    for count, actual, predicted in confusion_pairs[:20]:

        print(
            f"  {actual:15s} -> "
            f"{predicted:15s}: {count}"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()