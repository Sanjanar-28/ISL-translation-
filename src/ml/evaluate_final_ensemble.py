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
    "ml_ensemble",
    "ensemble_predictions.csv",
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "ml_ensemble",
)

# ============================================================
# LOAD PREDICTIONS
# ============================================================

df = pd.read_csv(PREDICTIONS_PATH)

y_true = df["actual_class_id"]
y_pred = df["ensemble_all_prediction"]

print("\n" + "=" * 70)
print("              FINAL 3-MODEL ENSEMBLE EVALUATION")
print("=" * 70)

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(y_true, y_pred)

precision_weighted = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

recall_weighted = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

f1_weighted = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

precision_macro = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

recall_macro = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

f1_macro = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

print(f"\nTest samples: {len(y_true)}")

print("\nOVERALL RESULTS")
print("-" * 70)

print(f"Accuracy:             {accuracy * 100:.2f}%")
print(f"Weighted Precision:   {precision_weighted * 100:.2f}%")
print(f"Weighted Recall:      {recall_weighted * 100:.2f}%")
print(f"Weighted F1-score:    {f1_weighted * 100:.2f}%")

print("\nMACRO AVERAGE RESULTS")
print("-" * 70)

print(f"Macro Precision:      {precision_macro * 100:.2f}%")
print(f"Macro Recall:         {recall_macro * 100:.2f}%")
print(f"Macro F1-score:       {f1_macro * 100:.2f}%")

# ============================================================
# LABEL MAPPING
# ============================================================

label_mapping = (
    df[
        [
            "actual_class_id",
            "actual_label",
        ]
    ]
    .drop_duplicates()
    .sort_values("actual_class_id")
)

labels = label_mapping["actual_class_id"].tolist()
target_names = label_mapping["actual_label"].tolist()

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
    "final_ensemble_classification_report.txt",
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
    "final_ensemble_confusion_matrix.csv",
)

cm_df.to_csv(cm_path)

print("\n" + "=" * 70)
print("SAVED FILES")
print("=" * 70)

print(f"Classification report:")
print(f"  {report_path}")

print(f"\nConfusion matrix:")
print(f"  {cm_path}")

print("\n" + "=" * 70)
print("          FINAL ENSEMBLE EVALUATION COMPLETE")
print("=" * 70)