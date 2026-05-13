from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from src.feature_engineering import engineer_features, FEATURE_COLS

# ── Model paths ────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# ── Load models at startup — once, not per request ────────────────────────────
xgb_model  = joblib.load(MODEL_DIR / "xgb_avito_best_model.joblib")
platt_xgb  = joblib.load(MODEL_DIR / "platt_xgb.joblib")
lgb_model  = joblib.load(MODEL_DIR / "lgb_avito_model.joblib")
platt_lgb  = joblib.load(MODEL_DIR / "platt_lgb.joblib")

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Ad Click CTR Prediction API",
    description=(
        "Predicts click-through probability for Avito-style contextual ads. "
        "Uses a Platt-calibrated XGBoost + LightGBM ensemble trained on "
        "2.4M contextual impressions. Ensemble AUC=0.7613, log-loss=0.034040."
    ),
    version="1.0.0",
)

# ── Input schema ───────────────────────────────────────────────────────────────
class AdRequest(BaseModel):
    # Core ad features
    HistCTR: float = Field(
        default=0.006142,
        ge=0.0, le=1.0,
        description="Historical CTR of this ad (0–1). Defaults to global mean."
    )
    Position: int = Field(
        default=1,
        ge=1, le=7,
        description="Ad position on the page (1=top)."
    )
    IsUserLoggedOn: int = Field(
        default=0,
        ge=0, le=1,
        description="1 if user is logged in, 0 if anonymous."
    )

    # Content features
    Price: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Ad price in local currency. None if not listed."
    )
    Title: Optional[str] = Field(
        default=None,
        description="Ad title text. Used for word count feature."
    )
    category_level: int = Field(
        default=1,
        ge=1, le=3,
        description="Category depth in hierarchy (1=top, 3=leaf)."
    )
    category_match: int = Field(
        default=0,
        ge=0, le=1,
        description="1 if ad category matches search category."
    )

    # Session context
    session_size: int = Field(
        default=1,
        ge=1, le=2,
        description="Number of contextual ads in this search session (1 or 2)."
    )
    SearchDate: Optional[str] = Field(
        default=None,
        description="ISO datetime of the search. Defaults to current time."
    )

    # User history (optional — falls back to global priors if not provided)
    user_impression_count: int = Field(
        default=0,
        ge=0,
        description="Cumulative impressions seen by this user."
    )
    user_click_count: int = Field(
        default=0,
        ge=0,
        description="Cumulative clicks by this user."
    )
    uid_category_count: int = Field(
        default=0,
        ge=0,
        description="Times this user has searched this category."
    )

    # Entity rate encodings (optional — falls back to smoothed(0,0) prior)
    ad_ctr: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    category_ctr: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    location_ctr: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    position_ctr: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    device_ctr: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    class Config:
        json_schema_extra = {
            "example": {
                "HistCTR": 0.008,
                "Position": 1,
                "IsUserLoggedOn": 1,
                "Price": 4500,
                "Title": "Продам ноутбук Lenovo ThinkPad",
                "category_level": 2,
                "category_match": 1,
                "session_size": 2,
                "user_impression_count": 12,
                "user_click_count": 2,
                "uid_category_count": 3,
                "ad_ctr": 0.009,
                "category_ctr": 0.007
            }
        }

# ── Output schema ──────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    p_click: float = Field(description="Calibrated ensemble click probability (0–1)")
    p_click_pct: float = Field(description="Click probability as percentage")
    xgb_calibrated: float = Field(description="XGBoost calibrated probability")
    lgb_calibrated: float = Field(description="LightGBM calibrated probability")
    baseline_ctr: float = Field(
        default=0.006142,
        description="Global baseline CTR for reference (0.6142%)"
    )
    above_baseline: bool = Field(
        description="True if predicted CTR exceeds baseline"
    )

class BatchRequest(BaseModel):
    records: List[AdRequest] = Field(
        description="List of ad impression records to score."
    )

class BatchResponse(BaseModel):
    predictions: List[float] = Field(
        description="Calibrated ensemble click probabilities, one per input record."
    )
    mean_p_click_pct: float = Field(
        description="Mean predicted CTR across the batch as percentage."
    )
    count: int

# ── Helper ─────────────────────────────────────────────────────────────────────
def score_features(X: pd.DataFrame) -> dict:
    """Run the full ensemble pipeline on a feature DataFrame."""
    xgb_raw = xgb_model.predict_proba(X)[:, 1]
    lgb_raw = lgb_model.predict_proba(X)[:, 1]
    xgb_cal = platt_xgb.predict_proba(xgb_raw.reshape(-1, 1))[:, 1]
    lgb_cal = platt_lgb.predict_proba(lgb_raw.reshape(-1, 1))[:, 1]
    ensemble = 0.5 * xgb_cal + 0.5 * lgb_cal
    return {
        'xgb_calibrated': xgb_cal,
        'lgb_calibrated': lgb_cal,
        'ensemble': ensemble
    }

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Liveness check — confirms models are loaded and API is running."""
    return {
        "status": "ok",
        "models_loaded": ["xgb_avito", "platt_xgb", "lgb_avito", "platt_lgb"],
        "ensemble_auc": 0.7613,
        "ensemble_log_loss": 0.034040
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: AdRequest):
    """
    Score a single ad impression.
    Returns calibrated ensemble click probability and per-model breakdown.
    """
    try:
        X = engineer_features(request.dict())
        scores = score_features(X)
        ensemble = float(scores['ensemble'][0])
        xgb_cal = float(scores['xgb_calibrated'][0])
        lgb_cal = float(scores['lgb_calibrated'][0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictionResponse(
        p_click=round(ensemble, 6),
        p_click_pct=round(ensemble * 100, 4),
        xgb_calibrated=round(xgb_cal, 6),
        lgb_calibrated=round(lgb_cal, 6),
        above_baseline=ensemble > 0.006142
    )

@app.post("/batch_predict", response_model=BatchResponse)
def batch_predict(request: BatchRequest):
    """
    Score a batch of ad impressions.
    Returns one probability per input record plus batch mean CTR.
    """
    if len(request.records) > 10000:
        raise HTTPException(
            status_code=400,
            detail="Batch size limit is 10,000 records."
        )
    try:
        rows = [engineer_features(r.dict()) for r in request.records]
        X_batch = pd.concat(rows, ignore_index=True)
        scores = score_features(X_batch)
        preds = scores['ensemble'].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return BatchResponse(
        predictions=[round(p, 6) for p in preds],
        mean_p_click_pct=round(float(np.mean(preds)) * 100, 4),
        count=len(preds)
    )
