"""Fit the logistic baseline and a LightGBM model. The test fold is not scored here."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import (
    BASELINE_PATH,
    FEATURES,
    METADATA_PATH,
    MODEL_PATH,
    SEED,
    SPLIT_PATH,
    TARGET,
    clean,
    load_raw,
    make_splits,
    model_matrix,
    quality_report,
    write_json,
)
from src.evaluate import classification_scores, choose_threshold

# Simpler candidates are listed first. A tie on validation ROC-AUC keeps the simpler one.
CANDIDATES = [
    {"num_leaves": 15, "min_child_samples": 40, "learning_rate": 0.05, "n_estimators": 400},
    {"num_leaves": 31, "min_child_samples": 50, "learning_rate": 0.05, "n_estimators": 400},
    {"num_leaves": 63, "min_child_samples": 100, "learning_rate": 0.05, "n_estimators": 400},
]


def _fit_baseline(x_train, y_train):
    categorical = ["education", "marriage"]
    numeric = [column for column in FEATURES if column not in categorical]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
            ("num", StandardScaler(), numeric),
        ]
    )
    # l1_ratio=0 is L2 in scikit-learn 1.8+. class_weight balances the log loss.
    classifier = LogisticRegression(
        C=1.0,
        l1_ratio=0.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=SEED,
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("model", classifier)])
    pipeline.fit(x_train, y_train)
    if list(pipeline.named_steps["model"].classes_) != [0, 1]:
        raise SystemExit(f"Unexpected baseline classes: {pipeline.named_steps['model'].classes_}")
    return pipeline


def _fit_lgbm(x_train, y_train, x_val, y_val, scale_pos_weight: float, params: dict):
    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        scale_pos_weight=scale_pos_weight,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=1,
        deterministic=True,
        force_col_wise=True,
        importance_type="gain",
        verbosity=-1,
        **params,
    )
    model.fit(
        x_train,
        y_train,
        eval_X=x_val,
        eval_y=y_val,
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=40, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    if list(model.classes_) != [0, 1]:
        raise SystemExit(f"Unexpected LightGBM classes: {model.classes_}")
    return model


def _print_scores(title: str, scores: dict) -> None:
    matrix = scores["confusion_matrix"]
    print(
        f"{title}: ROC-AUC {scores['roc_auc']:.4f}  PR-AUC {scores['pr_auc']:.4f}  "
        f"F1 {scores['f1']:.4f}  precision {scores['precision']:.4f}  "
        f"recall {scores['recall']:.4f}  threshold {scores['threshold']:.2f}"
    )
    print(
        f"  confusion tn={matrix['tn']} fp={matrix['fp']} "
        f"fn={matrix['fn']} tp={matrix['tp']}  cost={scores['cost_5fn_1fp']:.0f}"
    )


def main() -> None:
    raw = load_raw()
    quality = quality_report(raw)
    train_df, val_df, test_df = make_splits(clean(raw))

    write_json(
        SPLIT_PATH,
        {
            "train": [int(value) for value in train_df["id"].tolist()],
            "validation": [int(value) for value in val_df["id"].tolist()],
            "test": [int(value) for value in test_df["id"].tolist()],
        },
    )

    x_train = model_matrix(train_df)
    y_train = train_df[TARGET].to_numpy(dtype=int)
    x_val = model_matrix(val_df)
    y_val = val_df[TARGET].to_numpy(dtype=int)

    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    # Training-fold counts only. Validation and test do not enter this weight.
    scale_pos_weight = negatives / positives

    baseline = _fit_baseline(x_train, y_train)
    baseline_val = baseline.predict_proba(x_val)[:, 1]
    baseline_threshold = choose_threshold(y_val, baseline_val)
    baseline_scores = classification_scores(y_val, baseline_val, baseline_threshold)

    candidate_rows = []
    winner = None
    for params in CANDIDATES:
        model = _fit_lgbm(x_train, y_train, x_val, y_val, scale_pos_weight, params)
        proba = model.predict_proba(x_val)[:, 1]
        roc_auc = float(roc_auc_score(y_val, proba))
        iteration = int(model.best_iteration_ or params["n_estimators"])
        candidate_rows.append(
            {
                "params": params,
                "validation_roc_auc": roc_auc,
                "best_iteration": iteration,
            }
        )
        print(
            f"candidate leaves={params['num_leaves']} "
            f"min_child={params['min_child_samples']} "
            f"val ROC-AUC {roc_auc:.4f}  trees {iteration}"
        )
        if winner is None or roc_auc > winner["roc_auc"]:
            winner = {"roc_auc": roc_auc, "model": model, "proba": proba, "params": params, "iteration": iteration}

    assert winner is not None
    lgbm_threshold = choose_threshold(y_val, winner["proba"])
    lgbm_scores = classification_scores(y_val, winner["proba"], lgbm_threshold)

    if lgbm_scores["roc_auc"] < baseline_scores["roc_auc"]:
        raise SystemExit(
            "LightGBM validation ROC-AUC is below the logistic baseline. "
            "Refusing to save it as the selected model."
        )

    iteration = winner["iteration"]
    winner["model"].booster_.save_model(str(MODEL_PATH), num_iteration=iteration)
    saved = lgb.Booster(model_file=str(MODEL_PATH))
    saved_val = saved.predict(x_val)
    if not np.allclose(saved_val, winner["proba"], rtol=0, atol=1e-6):
        max_gap = float(np.max(np.abs(saved_val - winner["proba"])))
        raise SystemExit(f"Saved booster does not match in-memory validation scores (max abs gap {max_gap}).")
    if list(saved.feature_name()) != FEATURES:
        raise SystemExit(f"Saved feature names {saved.feature_name()} do not match FEATURES.")

    import joblib

    joblib.dump(baseline, BASELINE_PATH)

    metadata = {
        "model_name": "lightgbm",
        "model_file": MODEL_PATH.name,
        "target": TARGET,
        "positive_class": 1,
        "features": FEATURES,
        "excluded_features": ["id", "sex"],
        "threshold": lgbm_threshold,
        "threshold_rule": "On validation, minimize 5*FN + 1*FP. Ties: higher recall, then lower threshold.",
        "imbalance": {
            "lightgbm": "scale_pos_weight = n_negative / n_positive on the training fold",
            "logistic_regression": "class_weight=balanced",
            "threshold": "same 5*FN + 1*FP rule, fit on validation probabilities",
            "scale_pos_weight": scale_pos_weight,
            "train_positives": positives,
            "train_negatives": negatives,
            "fn_cost": 5,
            "fp_cost": 1,
        },
        "cleaning": {
            "education_keep": [1, 2, 3, 4],
            "education_other_mapped_to": 4,
            "marriage_keep": [1, 2, 3],
            "marriage_other_mapped_to": 3,
            "rows_dropped": 0,
        },
        "pay_status_note": "pay_0 is September 2005. The source file has no PAY_1 column. pay_2..pay_6 run from August back to April.",
        "split": {
            "seed": SEED,
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_default_rate": float(y_train.mean()),
            "validation_default_rate": float(y_val.mean()),
        },
        "lightgbm_version": lgb.__version__,
        "selected_params": winner["params"],
        "best_iteration": iteration,
        "validation": {
            "logistic_regression": baseline_scores,
            "lightgbm": lgbm_scores,
            "candidates": candidate_rows,
        },
        "data_quality": quality,
        "selection_metric": "validation ROC-AUC",
    }
    write_json(METADATA_PATH, metadata)

    print(f"scale_pos_weight {scale_pos_weight:.4f}  train default rate {y_train.mean():.4f}")
    _print_scores("validation logistic", baseline_scores)
    _print_scores("validation lightgbm", lgbm_scores)
    print(f"saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
