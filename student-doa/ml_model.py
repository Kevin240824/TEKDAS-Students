"""Train and evaluate a product-grouped rating classification model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data_prep import (
    BASE_DIR,
    CLASS_ORDER,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    prepare_dataset,
)


MODEL_PATH = BASE_DIR / "amazon_rating_pipeline.joblib"
METRICS_PATH = BASE_DIR / "model_metrics.json"
FEATURE_IMPORTANCE_PATH = BASE_DIR / "feature_importance.csv"
PREDICTIONS_PATH = BASE_DIR / "test_predictions.csv"
NUMERIC_FEATURES = [
    "discounted_price",
    "actual_price",
    "discount_percentage",
    "rating_count",
]
CATEGORICAL_FEATURES = ["main_category"]


def split_dataset(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a near-80/20 stratified split without sharing product IDs."""
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )
    groups = dataset["product_id"].fillna("__missing_product_id__")
    try:
        train_positions, test_positions = next(
            splitter.split(
                dataset[FEATURE_COLUMNS],
                dataset[TARGET_COLUMN],
                groups=groups,
            )
        )
    except ValueError as error:
        raise ValueError(
            "Tidak dapat membuat split stratified per product_id. "
            "Pastikan setiap kelas memiliki cukup grup produk."
        ) from error

    train = dataset.iloc[train_positions].copy()
    test = dataset.iloc[test_positions].copy()
    if not train.index.is_unique or not test.index.is_unique:
        raise ValueError("Indeks baris harus unik agar split dapat divalidasi.")
    if not train.index.intersection(test.index).empty:
        raise AssertionError("Indeks train dan test tumpang tindih.")
    if not set(train["product_id"].dropna()).isdisjoint(
        set(test["product_id"].dropna())
    ):
        raise AssertionError("product_id ditemukan pada train dan test sekaligus.")
    if set(train[TARGET_COLUMN]) != set(CLASS_ORDER):
        raise ValueError("Train split tidak memuat ketiga kelas rating.")
    return train, test


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("category", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_model(dataset: pd.DataFrame) -> dict[str, Any]:
    """Fit the pipeline on train rows and return test metrics and artifacts."""
    train, test = split_dataset(dataset)
    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN].astype(str)
    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN].astype(str)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    predicted = pipeline.predict(X_test)
    probability_array = pipeline.predict_proba(X_test)
    model_classes = pipeline.named_steps["classifier"].classes_.tolist()
    probabilities = pd.DataFrame(
        probability_array,
        index=test.index,
        columns=[f"probability_{label}" for label in model_classes],
    )
    if not np.allclose(probability_array.sum(axis=1), 1.0):
        raise AssertionError("Probabilitas prediksi tidak berjumlah 1.")

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_predicted = dummy.predict(X_test)

    report = classification_report(
        y_test,
        predicted,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    feature_names = pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out()
    importances = pipeline.named_steps[
        "classifier"
    ].feature_importances_
    feature_importance = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    prediction_frame = test[
        ["product_id", "product_name", "main_category", "rating", TARGET_COLUMN]
    ].copy()
    prediction_frame["predicted_class"] = predicted
    for column in probabilities.columns:
        prediction_frame[column] = probabilities[column]

    metrics = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_ratio": float(len(train) / len(dataset)),
        "test_ratio": float(len(test) / len(dataset)),
        "accuracy": float(accuracy_score(y_test, predicted)),
        "macro_f1": float(f1_score(y_test, predicted, average="macro")),
        "dummy_accuracy": float(accuracy_score(y_test, dummy_predicted)),
        "dummy_macro_f1": float(
            f1_score(y_test, dummy_predicted, average="macro", zero_division=0)
        ),
        "classes": CLASS_ORDER,
        "class_report": {
            label: {
                metric: float(value)
                for metric, value in report[label].items()
            }
            for label in CLASS_ORDER
        },
        "confusion_matrix": confusion_matrix(
            y_test,
            predicted,
            labels=CLASS_ORDER,
        ).tolist(),
        "predicted_classes": sorted(set(map(str, predicted))),
        "train_product_ids": int(train["product_id"].nunique()),
        "test_product_ids": int(test["product_id"].nunique()),
    }
    return {
        "pipeline": pipeline,
        "train": train,
        "test": test,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "predictions": prediction_frame,
        "probabilities": probabilities,
    }


def save_artifacts(results: dict[str, Any]) -> None:
    joblib.dump(results["pipeline"], MODEL_PATH)
    METRICS_PATH.write_text(
        json.dumps(results["metrics"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    results["feature_importance"].to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    results["predictions"].to_csv(PREDICTIONS_PATH, index=False)


def main() -> None:
    dataset, _ = prepare_dataset()
    results = train_model(dataset)
    save_artifacts(results)
    metrics = results["metrics"]
    print(
        f"Train/test: {metrics['train_rows']}/{metrics['test_rows']} baris "
        f"({metrics['train_ratio']:.1%}/{metrics['test_ratio']:.1%})"
    )
    print(
        f"Accuracy: {metrics['accuracy']:.3f} | "
        f"Macro F1: {metrics['macro_f1']:.3f} | "
        f"Dummy accuracy: {metrics['dummy_accuracy']:.3f} | "
        f"Dummy macro F1: {metrics['dummy_macro_f1']:.3f}"
    )
    print("Classification report:")
    print(
        classification_report(
            results["test"][TARGET_COLUMN].astype(str),
            results["pipeline"].predict(
                results["test"][FEATURE_COLUMNS]
            ),
            labels=CLASS_ORDER,
            zero_division=0,
        )
    )
    print(f"Model dan metrik disimpan di {BASE_DIR}")


if __name__ == "__main__":
    main()
