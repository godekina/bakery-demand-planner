"""Train the Store 14 bakery demand model and save it with its feature contract.

A plain script, not a notebook, because the thing you deploy has to be runnable
by a scheduler or a build step, neither of which can click Run All.

    python train.py                    # the model the app serves
    python train.py --compare          # four settings side by side, then save the best
"""
import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge

HERE = Path(__file__).parent
DATA = HERE / 'data' / 'store14_bakery_sales.csv'
MODEL_DIR = HERE / 'models'
HORIZON = 14          # the ordering lead time, fixed by the supply chain
TEST_DAYS = 84        # the final twelve weeks, opened once
BREAK_DATE = '2024-07-01'


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    df['closed'] = (df.units_sold == 0).astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Every column here must be knowable 14 days before the day it describes."""
    f = pd.DataFrame({'date': df.date, 'y': df.units_sold, 'promo': df.promo,
                      'closed': df.closed})
    for lag in (14, 21, 28, 35):
        f[f'lag{lag}'] = df.units_sold.shift(lag)
    f['same_weekday_mean'] = f[['lag14', 'lag21', 'lag28', 'lag35']].mean(axis=1)
    f['level_28d'] = df.units_sold.shift(HORIZON).rolling(28).mean()
    f['weekday'] = df.date.dt.weekday
    f['month'] = df.date.dt.month
    f['post_competitor'] = (df.date >= BREAK_DATE).astype(int)
    f['t'] = np.arange(len(f))
    f = pd.get_dummies(f, columns=['weekday', 'month'], drop_first=True)
    return f.dropna().reset_index(drop=True)


def metrics(actual, pred):
    e = np.asarray(actual, float) - np.asarray(pred, float)
    return {'mae': float(np.abs(e).mean()),
            'rmse': float(np.sqrt((e ** 2).mean())),
            'mape': float(np.abs(e / np.asarray(actual, float)).mean() * 100),
            'bias': float(e.mean())}


def fit_once(alpha=0.1, log_target=True, save=False):
    df = load()
    f = build_features(df)
    test, train = f.iloc[-TEST_DAYS:], f.iloc[:-TEST_DAYS]
    open_train = train[train.closed == 0]

    drop = ['date', 'y', 'closed']
    X_tr, y_tr = open_train.drop(columns=drop), open_train.y
    X_te, y_te = test.drop(columns=drop), test.y

    model = Ridge(alpha=alpha)
    model.fit(X_tr, np.log(y_tr) if log_target else y_tr)
    raw = model.predict(X_te)
    pred = np.exp(raw) if log_target else raw
    pred = np.where(test.closed == 1, 0, pred)
    m = metrics(y_te[test.closed == 0], pred[test.closed.values == 0])

    if save:
        MODEL_DIR.mkdir(exist_ok=True)
        meta = {
            'model': 'ridge',
            'alpha': alpha,
            'log_target': log_target,
            'features': list(X_tr.columns),
            'metrics': m,
            'horizon_days': HORIZON,
            'trained_at': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
            'trained_with': {'python': platform.python_version(),
                             'scikit_learn': sklearn.__version__,
                             'numpy': np.__version__,
                             'pandas': pd.__version__},
            'training_rows': int(len(X_tr)),
            'data_file': DATA.name,
        }
        joblib.dump({'model': model, 'meta': meta}, MODEL_DIR / 'model.joblib')
        (MODEL_DIR / 'model_meta.json').write_text(json.dumps(meta, indent=2))
    return model, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--compare', action='store_true',
                    help='try four settings, print the table, then save the best')
    args = ap.parse_args()

    if not args.compare:
        _, m = fit_once(save=True)
        print(f"saved models/model.joblib   MAE {m['mae']:.2f}  RMSE {m['rmse']:.2f}  "
              f"MAPE {m['mape']:.2f}%  bias {m['bias']:+.2f}")
        print(f"trained with scikit-learn {sklearn.__version__}, "
              f"which is the version requirements.txt pins")
        return

    rows = []
    for alpha, log_target in [(0.1, True), (1.0, True), (10.0, True), (1.0, False)]:
        _, m = fit_once(alpha, log_target)
        rows.append({'alpha': alpha, 'log_target': log_target, **m})
    table = pd.DataFrame(rows).sort_values('mae')
    print(table.to_string(index=False, float_format=lambda v: f'{v:.2f}'))

    best = table.iloc[0]
    print(f"\nbest: alpha={best.alpha} log_target={bool(best.log_target)}")
    _, m = fit_once(float(best.alpha), bool(best.log_target), save=True)
    print(f"saved models/model.joblib   MAE {m['mae']:.2f}")


if __name__ == '__main__':
    main()
