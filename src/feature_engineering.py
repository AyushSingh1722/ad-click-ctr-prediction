import pandas as pd
import numpy as np
from typing import Optional

FEATURE_COLS = [
    'HistCTR',
    'Position', 'position_in_session', 'ads_before', 'session_size',
    'hour_of_day', 'day_of_week',
    'user_impression_count', 'user_historical_ctr', 'uid_category_count',
    'ad_ctr', 'category_ctr', 'location_ctr', 'position_ctr', 'device_ctr',
    'price_log', 'has_price', 'title_word_count',
    'category_level', 'category_match', 'IsUserLoggedOn'
]

GLOBAL_CTR = 0.006142   # contextual CTR from NB04
ALPHA = 0.05            # smoothing alpha from NB05
BETA = 75               # smoothing beta from NB05

def smoothed_ctr(clicks: float = 0, impressions: float = 0) -> float:
    """Laplace-smoothed CTR — same formula as NB05 rate encoding."""
    return (clicks + ALPHA * BETA) / (impressions + BETA)

def engineer_features(record: dict) -> pd.DataFrame:
    """
    Takes a raw Avito impression record dict and returns a single-row
    DataFrame with all 21 features in FEATURE_COLS order.
    Mirrors the logic in NB08 engineer_features() exactly.
    Falls back to global priors for missing user/entity history.
    """
    # Parse SearchDate — default to now if not provided or explicitly None
    _sd = record.get('SearchDate')
    search_date = pd.to_datetime(_sd) if _sd is not None else pd.Timestamp.now()

    # Temporal features
    hour_of_day = search_date.hour
    day_of_week = search_date.dayofweek

    # Session features
    position = float(record.get('Position', 1))
    session_size = float(record.get('session_size', 1))
    position_in_session = position / session_size
    ads_before = max(position - 1, 0)

    # User behaviour — fallback to global prior
    user_imp_count = float(record.get('user_impression_count', 0))
    user_click_count = float(record.get('user_click_count', 0))
    user_hist_ctr = (
        user_click_count / user_imp_count
        if user_imp_count > 0 else GLOBAL_CTR
    )
    uid_cat_count = float(record.get('uid_category_count', 0))

    # Rate encoding — fallback to smoothed(0,0) for unseen entities or None
    _prior = smoothed_ctr()
    def _ctr(key: str) -> float:
        v = record.get(key)
        return float(v) if v is not None else _prior
    ad_ctr = _ctr('ad_ctr')
    category_ctr = _ctr('category_ctr')
    location_ctr = _ctr('location_ctr')
    position_ctr = _ctr('position_ctr')
    device_ctr = _ctr('device_ctr')

    # Content features
    price = record.get('Price', None)
    try:
        price_val = float(price) if price is not None else None
    except (ValueError, TypeError):
        price_val = None
    price_log = np.log1p(price_val) if price_val and price_val > 0 else 0.0
    has_price = 1 if price_val and price_val > 0 else 0

    title = record.get('Title', '')
    title_wc = len(str(title).split()) if title else 0

    category_level = float(record.get('category_level', 1))
    category_match = int(record.get('category_match', 0))
    is_logged_on = int(record.get('IsUserLoggedOn', 0))
    hist_ctr = float(record.get('HistCTR', GLOBAL_CTR))

    features = {
        'HistCTR': hist_ctr,
        'Position': position,
        'position_in_session': position_in_session,
        'ads_before': ads_before,
        'session_size': session_size,
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week,
        'user_impression_count': user_imp_count,
        'user_historical_ctr': user_hist_ctr,
        'uid_category_count': uid_cat_count,
        'ad_ctr': ad_ctr,
        'category_ctr': category_ctr,
        'location_ctr': location_ctr,
        'position_ctr': position_ctr,
        'device_ctr': device_ctr,
        'price_log': price_log,
        'has_price': has_price,
        'title_word_count': title_wc,
        'category_level': category_level,
        'category_match': category_match,
        'IsUserLoggedOn': is_logged_on
    }

    return pd.DataFrame([features])[FEATURE_COLS].fillna(0)
