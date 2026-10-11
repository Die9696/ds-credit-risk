"""Load the UCI Default of Credit Card Clients file and build the modeling table.

Source: Yeh, I-Cheng (2009). UCI Machine Learning Repository.
DOI: https://doi.org/10.24432/C55S3H
License: CC BY 4.0.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = ROOT / "models"

UCI_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
XLS_NAME = "default of credit card clients.xls"
ZIP_PATH = RAW_DIR / "default_of_credit_card_clients.zip"
XLS_PATH = RAW_DIR / XLS_NAME

# Pins the file the serialized model and the README numbers were built from.
ZIP_SHA256 = "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602"
XLS_SHA256 = "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933"

MODEL_PATH = MODELS_DIR / "model.txt"
METADATA_PATH = MODELS_DIR / "metadata.json"
BASELINE_PATH = MODELS_DIR / "baseline.joblib"
SPLIT_PATH = MODELS_DIR / "split_ids.json"

TARGET = "default_payment_next_month"
SEED = 42
TRAIN_ROWS = 21_000
VAL_ROWS = 4_500
TEST_ROWS = 4_500

# Sex is on the file and is not a model input.
FEATURES = [
    "limit_bal",
    "education",
    "marriage",
    "age",
    "pay_0",
    "pay_2",
    "pay_3",
    "pay_4",
    "pay_5",
    "pay_6",
    "bill_amt1",
    "bill_amt2",
    "bill_amt3",
    "bill_amt4",
    "bill_amt5",
    "bill_amt6",
    "pay_amt1",
    "pay_amt2",
    "pay_amt3",
    "pay_amt4",
    "pay_amt5",
    "pay_amt6",
]

PAY_STATUS = ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]
BILL_AMT = ["bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6"]
PAY_AMT = ["pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6"]

# Codes documented by UCI. Anything else is collapsed to "other".
EDUCATION_KEEP = (1, 2, 3, 4)
EDUCATION_OTHER = 4
MARRIAGE_KEEP = (1, 2, 3)
MARRIAGE_OTHER = 3

_SOURCE_COLUMNS = [
    "ID",
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
    "default payment next month",
]

_RENAME = {
    "ID": "id",
    "LIMIT_BAL": "limit_bal",
    "SEX": "sex",
    "EDUCATION": "education",
    "MARRIAGE": "marriage",
    "AGE": "age",
    "PAY_0": "pay_0",
    "PAY_2": "pay_2",
    "PAY_3": "pay_3",
    "PAY_4": "pay_4",
    "PAY_5": "pay_5",
    "PAY_6": "pay_6",
    "BILL_AMT1": "bill_amt1",
    "BILL_AMT2": "bill_amt2",
    "BILL_AMT3": "bill_amt3",
    "BILL_AMT4": "bill_amt4",
    "BILL_AMT5": "bill_amt5",
    "BILL_AMT6": "bill_amt6",
    "PAY_AMT1": "pay_amt1",
    "PAY_AMT2": "pay_amt2",
    "PAY_AMT3": "pay_amt3",
    "PAY_AMT4": "pay_amt4",
    "PAY_AMT5": "pay_amt5",
    "PAY_AMT6": "pay_amt6",
    "default payment next month": TARGET,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_xls() -> Path:
    """Download the official UCI zip and return the extracted workbook path."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if XLS_PATH.exists() and _sha256(XLS_PATH) == XLS_SHA256:
        return XLS_PATH

    print(f"Downloading {UCI_URL}")
    request = urllib.request.Request(UCI_URL, headers={"User-Agent": "ds-credit-risk/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"Could not download the UCI dataset from {UCI_URL}: {exc}. "
            "The dataset is CC BY 4.0 (UCI id 350). "
            "A public alternative with the same schema is the UCI file mirrored by OpenML "
            "once you confirm the license on that copy."
        ) from exc

    digest = hashlib.sha256(payload).hexdigest()
    if digest != ZIP_SHA256:
        raise SystemExit(
            f"SHA256 of the downloaded zip is {digest}, expected {ZIP_SHA256}. "
            "The UCI archive no longer matches the file this model was trained on."
        )

    ZIP_PATH.write_bytes(payload)
    with zipfile.ZipFile(ZIP_PATH) as archive:
        if XLS_NAME not in archive.namelist():
            raise SystemExit(f"The zip does not contain {XLS_NAME!r}.")
        archive.extract(XLS_NAME, RAW_DIR)

    found = _sha256(XLS_PATH)
    if found != XLS_SHA256:
        raise SystemExit(f"SHA256 of the extracted workbook is {found}, expected {XLS_SHA256}.")
    return XLS_PATH


def load_raw() -> pd.DataFrame:
    """Return the published table with snake_case columns, before code cleanup.

    The workbook has two header rows. Row 0 is X1..Y and row 1 is the names we use.
    PAY_0 is the September 2005 status. The file has no PAY_1 column.
    """
    import xlrd

    major = int(str(xlrd.__VERSION__).split(".", maxsplit=1)[0])
    if major >= 2:
        raise SystemExit("xlrd>=2 cannot read legacy .xls. Install xlrd==1.2.0.")

    book = xlrd.open_workbook(ensure_xls())
    sheet = book.sheet_by_index(0)
    headers = [str(sheet.cell_value(1, col)).strip() for col in range(sheet.ncols)]
    if headers != _SOURCE_COLUMNS:
        raise SystemExit(f"Unexpected columns in the workbook: {headers}")
    if sheet.nrows != 30_002:
        raise SystemExit(f"Expected 30000 data rows, found {sheet.nrows - 2}.")

    rows = [
        [sheet.cell_value(row, col) for col in range(sheet.ncols)]
        for row in range(2, sheet.nrows)
    ]
    frame = pd.DataFrame(rows, columns=headers).rename(columns=_RENAME)
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    missing = frame.isna().sum()
    if int(missing.sum()) > 0:
        present = missing[missing > 0].to_dict()
        raise SystemExit(f"Unexpected missing values after load: {present}")

    frame["id"] = frame["id"].astype(int)
    if frame["id"].duplicated().any():
        raise SystemExit("The workbook has duplicate client ids.")
    if int(frame["id"].min()) != 1 or int(frame["id"].max()) != 30_000:
        raise SystemExit("Client ids are not the expected 1..30000 range.")

    frame = frame.sort_values("id").reset_index(drop=True)
    labels = set(frame[TARGET].astype(int).unique())
    if labels != {0, 1}:
        raise SystemExit(f"Target labels are {labels}, expected {{0, 1}}.")
    return frame


def _counts(series: pd.Series) -> dict[str, int]:
    values = series.astype(int).value_counts().sort_index()
    return {str(int(key)): int(value) for key, value in values.items()}


def quality_report(raw: pd.DataFrame) -> dict:
    """Summarize the published file before education and marriage codes are collapsed."""
    education = raw["education"].astype(int)
    marriage = raw["marriage"].astype(int)
    bills = raw[BILL_AMT].to_numpy()
    payments = raw[PAY_AMT].to_numpy()
    default_count = int((raw[TARGET] == 1).sum())
    rows = int(len(raw))
    return {
        "rows": rows,
        "columns_including_id_and_target": int(raw.shape[1]),
        "missing_cells": int(raw.isna().sum().sum()),
        "duplicate_ids": int(raw["id"].duplicated().sum()),
        "default_count": default_count,
        "non_default_count": rows - default_count,
        "default_rate": default_count / rows,
        "education_counts": _counts(education),
        "marriage_counts": _counts(marriage),
        "undocumented_education_rows": int((~education.isin(EDUCATION_KEEP)).sum()),
        "undocumented_marriage_rows": int((~marriage.isin(MARRIAGE_KEEP)).sum()),
        "negative_bill_rows": int((bills < 0).any(axis=1).sum()),
        "negative_bill_cells": int((bills < 0).sum()),
        "pay_amount_max": float(payments.max()),
        "pay_amount_p99": float(np.quantile(payments, 0.99)),
        "limit_bal_min": float(raw["limit_bal"].min()),
        "limit_bal_max": float(raw["limit_bal"].max()),
        "age_min": int(raw["age"].min()),
        "age_max": int(raw["age"].max()),
        "pay_0_min": int(raw["pay_0"].min()),
        "pay_0_max": int(raw["pay_0"].max()),
    }


def _map_codes(values: pd.Series, keep: tuple[int, ...], other: int) -> pd.Series:
    codes = values.astype(int)
    return codes.where(codes.isin(list(keep)), other).astype(int)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse undocumented education and marriage codes. Do not drop rows."""
    frame = raw.copy()
    frame["education"] = _map_codes(frame["education"], EDUCATION_KEEP, EDUCATION_OTHER)
    frame["marriage"] = _map_codes(frame["marriage"], MARRIAGE_KEEP, MARRIAGE_OTHER)
    frame[TARGET] = frame[TARGET].astype(int)
    frame["age"] = frame["age"].astype(int)
    for column in PAY_STATUS:
        frame[column] = frame[column].astype(int)
    return frame


def model_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix in the order the saved model expects.

    Education and marriage codes are mapped again so a raw request and a
    cleaned training frame take the same path.
    """
    mapped = frame.copy()
    mapped["education"] = _map_codes(mapped["education"], EDUCATION_KEEP, EDUCATION_OTHER)
    mapped["marriage"] = _map_codes(mapped["marriage"], MARRIAGE_KEEP, MARRIAGE_OTHER)
    return mapped.loc[:, FEATURES].astype(np.float64)


def make_splits(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 21_000 / 4_500 / 4_500 split. The test fold is not for fitting."""
    from sklearn.model_selection import train_test_split

    if len(frame) != 30_000:
        raise SystemExit(f"Expected 30000 rows, found {len(frame)}.")

    train_val, test = train_test_split(
        frame,
        test_size=TEST_ROWS,
        random_state=SEED,
        stratify=frame[TARGET],
    )
    train, validation = train_test_split(
        train_val,
        test_size=VAL_ROWS,
        random_state=SEED,
        stratify=train_val[TARGET],
    )
    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    test = test.reset_index(drop=True)
    if (len(train), len(validation), len(test)) != (TRAIN_ROWS, VAL_ROWS, TEST_ROWS):
        raise SystemExit("Split sizes are not 21000/4500/4500.")

    ids = [train["id"], validation["id"], test["id"]]
    merged = pd.concat(ids)
    if merged.duplicated().any() or merged.nunique() != 30_000:
        raise SystemExit("Train, validation, and test ids overlap or do not cover the file.")
    return train, validation, test


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    raw = load_raw()
    report = quality_report(raw)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
