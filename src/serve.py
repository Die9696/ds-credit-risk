"""HTTP API: one applicant in, probability of default out."""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from src.data import FEATURES, METADATA_PATH, MODEL_PATH, model_matrix

def _load():
    if not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
        raise RuntimeError("models/model.txt and models/metadata.json are required. Run python -m src.train.")
    metadata = json.loads(METADATA_PATH.read_text())
    if metadata.get("features") != FEATURES:
        raise RuntimeError("models/metadata.json features do not match src/data.py.")
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    if list(booster.feature_name()) != FEATURES:
        raise RuntimeError("Saved LightGBM feature names do not match src/data.py.")
    return booster, metadata


BOOSTER, METADATA = _load()
THRESHOLD = float(METADATA["threshold"])

app = FastAPI(
    title="Credit default probability",
    version="1.0.0",
    description="Probability that a credit-card client defaults next month.",
)


class Applicant(BaseModel):
    """Raw applicant fields. Undocumented education and marriage codes are mapped in the model matrix."""

    model_config = ConfigDict(extra="forbid")

    limit_bal: float = Field(..., ge=0)
    education: int
    marriage: int
    age: int = Field(..., ge=18, le=120)
    pay_0: int
    pay_2: int
    pay_3: int
    pay_4: int
    pay_5: int
    pay_6: int
    bill_amt1: float
    bill_amt2: float
    bill_amt3: float
    bill_amt4: float
    bill_amt5: float
    bill_amt6: float
    pay_amt1: float
    pay_amt2: float
    pay_amt3: float
    pay_amt4: float
    pay_amt5: float
    pay_amt6: float


class Prediction(BaseModel):
    probability_of_default: float
    predicted_default: int
    threshold: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": METADATA["model_name"], "threshold": THRESHOLD}


@app.post("/predict", response_model=Prediction)
def predict(applicant: Applicant) -> Prediction:
    frame = model_matrix(pd.DataFrame([applicant.model_dump()]))
    probability = round(float(np.clip(BOOSTER.predict(frame)[0], 0.0, 1.0)), 6)
    return Prediction(
        probability_of_default=probability,
        predicted_default=int(probability >= THRESHOLD),
        threshold=THRESHOLD,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
