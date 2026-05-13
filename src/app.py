import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from src.feature_engineering import engineer_features, FEATURE_COLS, GLOBAL_CTR


@st.cache_resource
def load_models():
    base = Path(__file__).resolve().parent.parent / "models"
    return {
        'xgb':       joblib.load(base / "xgb_avito_best_model.joblib"),
        'platt_xgb': joblib.load(base / "platt_xgb.joblib"),
        'lgb':       joblib.load(base / "lgb_avito_model.joblib"),
        'platt_lgb': joblib.load(base / "platt_lgb.joblib"),
    }


def predict(record: dict, models: dict) -> dict:
    X = engineer_features(record)
    xgb_raw  = models['xgb'].predict_proba(X)[:, 1][0]
    lgb_raw  = models['lgb'].predict_proba(X)[:, 1][0]
    xgb_cal  = models['platt_xgb'].predict_proba([[xgb_raw]])[:, 1][0]
    lgb_cal  = models['platt_lgb'].predict_proba([[lgb_raw]])[:, 1][0]
    ensemble = 0.5 * xgb_cal + 0.5 * lgb_cal
    return {
        'ensemble':  round(ensemble, 6),
        'xgb_cal':   round(xgb_cal, 6),
        'lgb_cal':   round(lgb_cal, 6),
    }


st.set_page_config(
    page_title="Ad Click CTR Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Ad Click CTR Predictor")
st.caption(
    "Platt-calibrated XGBoost + LightGBM ensemble trained on 2.4M Avito "
    "contextual impressions. Ensemble AUC=0.7613, log-loss=0.034040."
)

col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("Ad Details")

    hist_ctr = st.slider(
        "Historical CTR (HistCTR)",
        min_value=0.001, max_value=0.05,
        value=0.008, step=0.001, format="%.3f",
        help="Historical click rate for this ad. Global mean = 0.6142%"
    )
    position = st.selectbox(
        "Ad Position", options=[1, 2, 3, 4, 5, 6, 7],
        index=0,
        help="Position on page (1=top). Note: model learned an inverted "
             "gradient vs empirical data due to session_size confound."
    )
    session_size = st.selectbox(
        "Session Size", options=[1, 2],
        index=1,
        help="Number of contextual ads in this search (1 or 2 in Avito data)"
    )
    is_logged_on = st.radio(
        "User Login Status",
        options=[0, 1],
        format_func=lambda x: "Logged out" if x == 0 else "Logged in",
        horizontal=True
    )
    category_match = st.radio(
        "Category Match",
        options=[0, 1],
        format_func=lambda x: "Mismatch" if x == 0 else "Match",
        horizontal=True,
        help="NB04: mismatch CTR (0.3167%) > match (0.2921%) — counterintuitive"
    )

    st.subheader("Content")
    price = st.number_input(
        "Price", min_value=0, max_value=1000000,
        value=4500, step=100,
        help="Ad price. 99.9% of Avito contextual ads have a price."
    )
    title = st.text_input(
        "Ad Title", value="Продам ноутбук Lenovo ThinkPad",
        help="Used for title_word_count feature."
    )
    category_level = st.selectbox(
        "Category Level", options=[1, 2, 3], index=1,
        help="Depth in category hierarchy (1=top, 3=leaf)"
    )

    st.subheader("User History")
    user_imp = st.number_input(
        "User Impression Count", min_value=0, max_value=1000,
        value=12, step=1,
        help="Cumulative ads seen by this user. 27.1% have zero history."
    )
    user_clicks = st.number_input(
        "User Click Count", min_value=0, max_value=100,
        value=2, step=1
    )
    uid_cat = st.number_input(
        "User–Category Count", min_value=0, max_value=100,
        value=3, step=1,
        help="Times this user has searched this category"
    )

    st.subheader("Rate Features")
    st.caption("Leave at dataset means if unknown.")
    ad_ctr       = st.slider("Ad CTR",       0.001, 0.030, 0.009, 0.001, "%.3f")
    category_ctr = st.slider("Category CTR", 0.001, 0.030, 0.007, 0.001, "%.3f")
    location_ctr = st.slider("Location CTR", 0.001, 0.030, 0.006, 0.001, "%.3f")

with col_output:
    st.subheader("Prediction")

    models = load_models()
    record = {
        'HistCTR':               hist_ctr,
        'Position':              position,
        'session_size':          session_size,
        'IsUserLoggedOn':        is_logged_on,
        'category_match':        category_match,
        'Price':                 price if price > 0 else None,
        'Title':                 title,
        'category_level':        category_level,
        'user_impression_count': user_imp,
        'user_click_count':      user_clicks,
        'uid_category_count':    uid_cat,
        'ad_ctr':                ad_ctr,
        'category_ctr':          category_ctr,
        'location_ctr':          location_ctr,
        'position_ctr':          0.006,
        'device_ctr':            0.006,
        'SearchDate':            '2015-06-01 14:00:00',
    }
    result = predict(record, models)
    ensemble_pct = result['ensemble'] * 100

    delta_pct = ((result['ensemble'] - GLOBAL_CTR) / GLOBAL_CTR) * 100
    st.metric(
        label="Predicted CTR (Ensemble)",
        value=f"{ensemble_pct:.4f}%",
        delta=f"{delta_pct:+.1f}% vs baseline ({GLOBAL_CTR*100:.4f}%)"
    )

    if result['ensemble'] > GLOBAL_CTR:
        st.success("Above baseline CTR ✓")
    else:
        st.warning("Below baseline CTR")

    st.subheader("Model Breakdown")
    breakdown = pd.DataFrame({
        'Model':    ['XGBoost (calibrated)', 'LightGBM (calibrated)', 'Ensemble'],
        'P(click)': [result['xgb_cal'], result['lgb_cal'], result['ensemble']],
        'CTR %':    [
            f"{result['xgb_cal']*100:.4f}%",
            f"{result['lgb_cal']*100:.4f}%",
            f"{ensemble_pct:.4f}%"
        ]
    })
    st.dataframe(breakdown, hide_index=True, use_container_width=True)

    st.subheader("Position Sensitivity")
    st.caption(
        "Model predicts an inverted position gradient vs empirical data — "
        "documented in NB08. Higher positions score higher due to "
        "session_size confound in training data."
    )
    pos_results = []
    for pos in range(1, 8):
        r = dict(record)
        r['Position'] = pos
        res = predict(r, models)
        pos_results.append({'Position': pos, 'CTR %': res['ensemble'] * 100})

    pos_df = pd.DataFrame(pos_results)
    st.line_chart(pos_df.set_index('Position'))

    st.subheader("HistCTR Sensitivity")
    st.caption("Non-monotone response — documented in NB08.")
    hctr_vals = [0.001, 0.003, 0.006, 0.010, 0.020]
    hctr_results = []
    for h in hctr_vals:
        r = dict(record)
        r['HistCTR'] = h
        res = predict(r, models)
        hctr_results.append({'HistCTR': h, 'CTR %': res['ensemble'] * 100})

    hctr_df = pd.DataFrame(hctr_results)
    st.line_chart(hctr_df.set_index('HistCTR'))

    st.subheader("Model Facts")
    st.markdown("""
| Metric | Value |
|---|---|
| Ensemble AUC | 0.7613 |
| Ensemble log-loss | 0.034040 |
| Baseline log-loss (HistCTR alone) | 0.039682 |
| Calibration ratio (batch) | 1.015× |
| Training rows | 2,422,983 |
| Features | 21 |
""")
