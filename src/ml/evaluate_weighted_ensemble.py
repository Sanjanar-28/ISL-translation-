import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

PREDICTIONS_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "ensemble",
    "weighted_ensemble_test_predictions.csv",
)

LABELS_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "ml_ensemble",
    "ensemble_predictions.csv",
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "ensemble",
)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(PREDICTIONS_PATH)
labels_df = pd.read_csv(LABELS_PATH)

y_true = df["actual_class_id"]
y_pred = df["weighted_prediction"]

# ============================================================
# CREATE CLASS ID -> LABEL MAPPING
# ============================================================

label_mapping = (
    labels_df[
        ["actual_class_id", "actual_label"]
    ]
    .drop_duplicates()
    .sort_values("actual_class_id")
)

labels = label_mapping["actual_class_id"].tolist()
target_names = label_mapping["actual_label"].tolist()

# ============================================================
# FINAL METRICS
# ============================================================

accuracy = accuracy_score(y_true, y_pred)

weighted_precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

weighted_recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

macro_precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

macro_recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

print("\n" + "=" * 70)
print("          FINAL WEIGHTED 3-MODEL ENSEMBLE EVALUATION")
print("=" * 70)

print(f"\nTest samples: {len(y_true)}")

print("\nOVERALL RESULTS")
print("-" * 70)
print(f"Accuracy:             {accuracy * 100:.2f}%")
print(f"Weighted Precision:   {weighted_precision * 100:.2f}%")
print(f"Weighted Recall:      {weighted_recall * 100:.2f}%")
print(f"Weighted F1-score:    {weighted_f1 * 100:.2f}%")

print("\nMACRO AVERAGE RESULTS")
print("-" * 70)
print(f"Macro Precision:      {macro_precision * 100:.2f}%")
print(f"Macro Recall:         {macro_recall * 100:.2f}%")
print(f"Macro F1-score:       {macro_f1 * 100:.2f}%")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    labels=labels,
    target_names=target_names,
    zero_division=0,
)

print("\nCLASSIFICATION REPORT")
print("=" * 70)
print(report)

report_path = os.path.join(
    OUTPUT_DIR,
    "final_weighted_ensemble_classification_report.txt",
)

with open(report_path, "w", encoding="utf-8") as file:
    file.write(report)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels,
)

cm_df = pd.DataFrame(
    cm,
    index=target_names,
    columns=target_names,
)

cm_path = os.path.join(
    OUTPUT_DIR,
    "final_weighted_ensemble_confusion_matrix.csv",
)

cm_df.to_csv(cm_path)

print("\n" + "=" * 70)
print("SAVED FILES")
print("=" * 70)
print(f"Classification report:\n  {report_path}")
print(f"\nConfusion matrix:\n  {cm_path}")

print("\n" + "=" * 70)
print("       FINAL WEIGHTED ENSEMBLE EVALUATION COMPLETE")
print("=" * 70)