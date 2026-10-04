"""Everything the app needs to turn a date into an order quantity.

Kept out of streamlit_app.py on purpose. A function in here can be imported and
tested without starting a web server, which is what the case study notebook does.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / 'models' / 'model.joblib'
META_PATH = HERE / 'models' / 'model_meta.json'
DATA_PATH = HERE / 'data' / 'store14_bakery_sales.csv'

# From the Week 23 backtest, not from a guess. Ordering above the forecast costs
# 62p a wasted unit; running out costs GBP 1.78 of lost margin.
SERVICE_MULTIPLIER = {0.50: 1.012, 0.60: 1.025, 0.70: 1.053, 0.74: 1.061,
                      0.80: 1.073, 0.90: 1.104}
DEFAULT_SERVICE_LEVEL = 0.74
BREAK_DATE = date(2024, 7, 1)


class ContractError(ValueError):
    """Raised when a request does not match what the model expects."""


def load_bundle(path: Path | str = MODEL_PATH):
    """Return (model, meta). The meta is the feature contract, not decoration."""
    bundle = joblib.load(path)
    model, meta = bundle['model'], bundle['meta']
    for key in ('features', 'log_target'):
        if key not in meta:
            raise ContractError(f'the bundle has no {key!r}, so it cannot be served safely')
    return model, meta


def load_history(path: Path | str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['date'])
    return df.sort_values('date').reset_index(drop=True)


def environment_note(meta: dict) -> str:
    """Say plainly whether this environment matches the one that trained the model."""
    trained = (meta.get('trained_with') or {}).get('scikit_learn')
    if trained is None:
        return 'The bundle does not record which scikit-learn trained it.'
    if trained == sklearn.__version__:
        return f'scikit-learn {sklearn.__version__} matches the version that trained the model.'
    return (f'This environment has scikit-learn {sklearn.__version__}; the model was trained '
            f'with {trained}. Pin the training version in requirements.txt.')


def build_row(target: date, lags: list[float], level_28d: float,
              promo: int = 0, closed: int = 0) -> dict:
    """One day of features, in the shape the model was fitted on."""
    if len(lags) != 4:
        raise ContractError('four lags are required: 14, 21, 28 and 35 days back')
    if any(float(v) < 0 for v in lags) or float(level_28d) < 0:
        raise ContractError('sales history cannot be negative')
    if promo not in (0, 1) or closed not in (0, 1):
        raise ContractError('promo and closed are 0 or 1')

    row = {'promo': int(promo), 'lag14': float(lags[0]), 'lag21': float(lags[1]),
           'lag28': float(lags[2]), 'lag35': float(lags[3]),
           'same_weekday_mean': float(np.mean(lags)), 'level_28d': float(level_28d),
           'post_competitor': int(target >= BREAK_DATE), 't': 0, '_closed': int(closed)}
    for w in range(1, 7):
        row[f'weekday_{w}'] = int(target.weekday() == w)
    for m in range(2, 13):
        row[f'month_{m}'] = int(target.month == m)
    return row


def forecast_units(model, meta: dict, rows: list[dict]) -> np.ndarray:
    """Forecast one or more days. The column order comes from the contract, never
    from the order somebody happened to build the dictionary in."""
    frame = pd.DataFrame(rows)
    closed = frame.pop('_closed') if '_closed' in frame else pd.Series(0, index=frame.index)

    missing = [c for c in meta['features'] if c not in frame.columns]
    if missing:
        raise ContractError(f'missing features: {missing[:5]}')
    X = frame[meta['features']]                     # the line that prevents silent nonsense

    raw = model.predict(X)
    units = np.exp(raw) if meta['log_target'] else raw
    units = np.where(closed.to_numpy() == 1, 0.0, units)
    return np.round(np.clip(units, 0, None), 1)


def order_quantity(forecast: float | np.ndarray, service_level: float = DEFAULT_SERVICE_LEVEL):
    """Turn a forecast into the number the counter actually orders."""
    if service_level not in SERVICE_MULTIPLIER:
        nearest = min(SERVICE_MULTIPLIER, key=lambda k: abs(k - service_level))
        multiplier = SERVICE_MULTIPLIER[nearest]
    else:
        multiplier = SERVICE_MULTIPLIER[service_level]
    return np.ceil(np.asarray(forecast, dtype=float) * multiplier)


def plan_fortnight(history: pd.DataFrame, model, meta: dict, start: date,
                   promo_days: set[str] | None = None,
                   closed_days: set[str] | None = None,
                   service_level: float = DEFAULT_SERVICE_LEVEL) -> pd.DataFrame:
    """The whole decision: fourteen days of forecast, and what to order for each."""
    promo_days = promo_days or set()
    closed_days = closed_days or set()

    units = history.units_sold
    lags = [float(units.iloc[-14]), float(units.iloc[-21]),
            float(units.iloc[-28]), float(units.iloc[-35])]
    level = float(units.tail(28).mean())

    days = [start + timedelta(days=i) for i in range(14)]
    rows = [build_row(d, lags, level,
                      promo=int(d.isoformat() in promo_days),
                      closed=int(d.isoformat() in closed_days)) for d in days]
    forecast = forecast_units(model, meta, rows)
    order = order_quantity(forecast, service_level)

    return pd.DataFrame({
        'date': [d.isoformat() for d in days],
        'weekday': [d.strftime('%a') for d in days],
        'promotion': ['yes' if d.isoformat() in promo_days else '' for d in days],
        'forecast_units': forecast,
        'order_units': order,
    })


def model_summary(meta: dict) -> dict:
    """What the app shows about the model it is serving."""
    return {
        'model': f"{meta.get('model', 'unknown')} (alpha {meta.get('alpha', 'n/a')})",
        'features': len(meta.get('features', [])),
        'trained_at': meta.get('trained_at', 'unknown'),
        'test_mae': round(meta.get('metrics', {}).get('mae', float('nan')), 2),
        'test_mape': round(meta.get('metrics', {}).get('mape', float('nan')), 2),
    }


def read_meta_file(path: Path | str = META_PATH) -> dict:
    return json.loads(Path(path).read_text())
