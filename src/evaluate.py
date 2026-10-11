"""Score the frozen models on the held-out test ids and write feature importance."""

from __future__ import annotations

import json

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data import (
    BASELINE_PATH,
    FEATURES,
    METADATA_PATH,
    MODEL_PATH,
    SPLIT_PATH,
    TARGET,
    clean,
    load_raw,
    model_matrix,
    write_json,
)

# Missed defaults are treated as five times as costly as a false alarm.
FN_COST = 5.0
FP_COST = 1.0


def choose_threshold(y_true, proba, fn_cost: float = FN_COST, fp_cost: float = FP_COST) -> float:
    """Pick an operating point on one labeled fold. Do not pass the test fold."""
    labels = np.asarray(y_true).astype(int)
    scores = np.asarray(proba, dtype=float)
    best_key = None
    best_threshold = 0.5
    for threshold in (round(value, 2) for value in np.linspace(0.05, 0.95, 91)):
        predicted = scores >= threshold
        false_positives = int(np.count_nonzero(predicted & (labels == 0)))
        false_negatives = int(np.count_nonzero(~predicted & (labels == 1)))
        true_positives = int(np.count_nonzero(predicted & (labels == 1)))
        cost = fn_cost * false_negatives + fp_cost * false_positives
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0
        key = (cost, -recall, threshold)
        if best_key is None or key < best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def classification_scores(y_true, proba, threshold: float) -> dict:
    labels = np.asarray(y_true).astype(int)
    scores = np.asarray(proba, dtype=float)
    predicted = (scores >= threshold).astype(int)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        labels, predicted, labels=[0, 1]
    ).ravel()
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": {
            "tn": int(true_negative),
            "fp": int(false_positive),
            "fn": int(false_negative),
            "tp": int(true_positive),
        },
        "cost_5fn_1fp": float(FN_COST * int(false_negative) + FP_COST * int(false_positive)),
    }


def feature_importance(booster: lgb.Booster) -> list[dict]:
    names = list(booster.feature_name())
    if names != FEATURES:
        raise SystemExit(f"Booster features {names} do not match the training contract.")
    gain = booster.feature_importance(importance_type="gain").astype(float)
    total = float(gain.sum()) or 1.0
    rows = [
        {"feature": name, "gain": float(value), "share": float(value / total)}
        for name, value in zip(names, gain)
    ]
    rows.sort(key=lambda row: row["gain"], reverse=True)
    return rows


def _fold(cleaned, ids: list[int]):
    table = cleaned.set_index("id", drop=False)
    fold = table.loc[ids].reset_index(drop=True)
    if len(fold) != len(ids):
        raise SystemExit("Split ids do not match the cleaned table.")
    return fold


def _print_block(title: str, scores: dict) -> None:
    matrix = scores["confusion_matrix"]
    print(
        f"{title}: ROC-AUC {scores['roc_auc']:.4f}  PR-AUC {scores['pr_auc']:.4f}  "
        f"F1 {scores['f1']:.4f}  precision {scores['precision']:.4f}  "
        f"recall {scores['recall']:.4f}  threshold {scores['threshold']:.2f}"
    )
    print(
        "  confusion matrix rows are actual [non-default, default], "
        "columns are predicted [non-default, default]"
    )
    print(f"  [[{matrix['tn']:5d} {matrix['fp']:5d}]")
    print(f"   [{matrix['fn']:5d} {matrix['tp']:5d}]]")


def main() -> None:
    for path, hint in (
        (MODEL_PATH, "Run python -m src.train"),
        (METADATA_PATH, "Run python -m src.train"),
        (BASELINE_PATH, "Run python -m src.train"),
        (SPLIT_PATH, "Run python -m src.train"),
    ):
        if not path.is_file():
            raise SystemExit(f"Missing {path}. {hint}.")

    metadata = json.loads(METADATA_PATH.read_text())
    split = json.loads(SPLIT_PATH.read_text())
    cleaned = clean(load_raw())
    test = _fold(cleaned, split["test"])
    labels = test[TARGET].to_numpy(dtype=int)
    features = model_matrix(test)

    booster = lgb.Booster(model_file=str(MODEL_PATH))
    baseline = joblib.load(BASELINE_PATH)
    lightgbm_scores = classification_scores(labels, booster.predict(features), float(metadata["threshold"]))
    baseline_scores = classification_scores(
        labels,
        baseline.predict_proba(features)[:, 1],
        float(metadata["validation"]["logistic_regression"]["threshold"]),
    )
    if lightgbm_scores["roc_auc"] <= 0.5:
        raise SystemExit("Test ROC-AUC is not above 0.5. The positive class may be flipped.")

    importance = feature_importance(booster)
    metadata["test"] = {
        "default_rate": float(labels.mean()),
        "logistic_regression": baseline_scores,
        "lightgbm": lightgbm_scores,
    }
    metadata["feature_importance"] = importance
    write_json(METADATA_PATH, metadata)

    print(f"test rows {len(test)}  default rate {labels.mean():.4f}")
    _print_block("test logistic", baseline_scores)
    _print_block("test lightgbm", lightgbm_scores)
    print("feature importance by gain share")
    for row in importance[:10]:
        print(f"  {row['feature']:12s} {row['share']:.3f}")


if __name__ == "__main__":
    main()
