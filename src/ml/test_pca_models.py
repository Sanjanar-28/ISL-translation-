from pathlib import Path

import numpy as np

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


DATA_DIR = Path("data/ml_normalized")


def main():

    print("\n" + "=" * 70)
    print("                    PCA MODEL EXPERIMENT")
    print("=" * 70)

    X_train = np.load(DATA_DIR / "X_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")

    X_val = np.load(DATA_DIR / "X_validation.npy")
    y_val = np.load(DATA_DIR / "y_validation.npy")

    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    print(f"\nOriginal features: {X_train.shape[1]}")

    for n_components in [128, 256, 512, 768, 1024]:

        print("\n" + "-" * 70)
        print(f"PCA components: {n_components}")

        pca = PCA(
            n_components=n_components,
            random_state=42,
        )

        Xtr = pca.fit_transform(X_train)
        Xv = pca.transform(X_val)
        Xt = pca.transform(X_test)

        print(
            f"Explained variance: "
            f"{pca.explained_variance_ratio_.sum():.4f}"
        )

        model = RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )

        model.fit(Xtr, y_train)

        val_pred = model.predict(Xv)
        test_pred = model.predict(Xt)

        val_acc = accuracy_score(y_val, val_pred)
        test_acc = accuracy_score(y_test, test_pred)

        print(f"Validation accuracy: {val_acc * 100:.2f}%")
        print(f"Test accuracy:       {test_acc * 100:.2f}%")


if __name__ == "__main__":
    main()