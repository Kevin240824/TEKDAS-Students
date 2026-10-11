"""Clean and validate the Amazon product-review dataset."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "amazon.csv"
CLEAN_DATA_PATH = BASE_DIR / "amazon_clean.csv"
QUALITY_REPORT_PATH = BASE_DIR / "data_quality_report.json"
NUMERIC_COLUMNS = [
    "discounted_price",
    "actual_price",
    "discount_percentage",
    "rating_count",
]
REQUIRED_COLUMNS = {
    "product_id",
    "product_name",
    "category",
    "discounted_price",
    "actual_price",
    "discount_percentage",
    "rating",
    "rating_count",
    "about_product",
}
FEATURE_COLUMNS = [
    "discounted_price",
    "actual_price",
    "discount_percentage",
    "rating_count",
    "main_category",
]
TARGET_COLUMN = "Rating_Class"
CLASS_ORDER = ["Low", "Medium", "High"]


def _parse_numeric(values: pd.Series) -> pd.Series:
    cleaned = values.astype("string").str.replace(r"[^0-9.\-]", "", regex=True)
    cleaned = cleaned.replace("", pd.NA)
    return pd.to_numeric(cleaned, errors="coerce")


def prepare_dataset(
    source_path: str | Path = DATA_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load, clean, and label rows, returning the frame and a quality report."""
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {source_path}")

    frame = pd.read_csv(source_path, encoding="utf-8")
    missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            "Kolom wajib tidak ditemukan: " + ", ".join(sorted(missing_columns))
        )

    input_rows = len(frame)
    missing_before = frame.isna().sum().to_dict()
    duplicate_rows = int(frame.duplicated().sum())

    for column in NUMERIC_COLUMNS:
        frame[column] = _parse_numeric(frame[column])

    frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce")
    invalid_rating_mask = frame["rating"].isna()
    invalid_rating_rows = int(invalid_rating_mask.sum())
    frame = frame.loc[~invalid_rating_mask].copy()
    if frame.empty:
        raise ValueError("Tidak ada baris dengan rating valid setelah pembersihan.")

    rating_count_median = frame["rating_count"].median()
    if pd.isna(rating_count_median):
        raise ValueError("Semua nilai rating_count kosong; median tidak dapat dihitung.")
    missing_rating_count = int(frame["rating_count"].isna().sum())
    frame["rating_count"] = frame["rating_count"].fillna(rating_count_median)

    frame["main_category"] = (
        frame["category"]
        .fillna("Tidak diketahui")
        .astype(str)
        .str.split("|", n=1)
        .str[0]
        .replace("", "Tidak diketahui")
    )
    frame[TARGET_COLUMN] = pd.cut(
        frame["rating"],
        bins=[float("-inf"), 4.0, 4.4, float("inf")],
        labels=CLASS_ORDER,
        right=False,
    )
    if frame[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Ada rating di luar rentang kelas Low/Medium/High yang ditentukan."
        )

    numeric_parse_failures = {
        column: int(frame[column].isna().sum())
        for column in NUMERIC_COLUMNS
    }
    quality_report = {
        "source": source_path.name,
        "input_rows": input_rows,
        "rows_dropped_invalid_rating": invalid_rating_rows,
        "clean_rows": len(frame),
        "columns": len(frame.columns),
        "duplicate_rows": duplicate_rows,
        "missing_values_before_cleaning": {
            column: int(count) for column, count in missing_before.items()
        },
        "missing_rating_count_filled_with_median": missing_rating_count,
        "rating_count_median": float(rating_count_median),
        "numeric_values_missing_after_conversion": numeric_parse_failures,
        "missing_values_after_cleaning": {
            column: int(count)
            for column, count in frame.isna().sum().items()
        },
        "rating_class_counts": {
            str(label): int(count)
            for label, count in frame[TARGET_COLUMN].value_counts().items()
        },
    }
    return frame, quality_report


def main() -> None:
    dataset, report = prepare_dataset()
    dataset.to_csv(CLEAN_DATA_PATH, index=False)
    QUALITY_REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Dataset bersih disimpan: {CLEAN_DATA_PATH.name}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
