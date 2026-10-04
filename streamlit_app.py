"""Store 14 bakery demand planner.

Run it locally with:

    streamlit run streamlit_app.py

This file is the entrypoint Streamlit Community Cloud looks for. It does the
screen, and nothing else. The forecasting lives in model_utils.py so that it can
be tested without a browser.
"""
from datetime import date, timedelta

import pandas as pd
import streamlit as st

import model_utils as mu

st.set_page_config(page_title='Bakery Demand Planner', page_icon='🥖', layout='wide')


@st.cache_resource
def get_model():
    """Loaded once per session, not once per click."""
    return mu.load_bundle()


@st.cache_data
def get_history():
    return mu.load_history()


st.title('Store 14 bakery demand planner')
st.caption('Plan the next fortnight of orders. Order from the order column, not the forecast.')

try:
    model, meta = get_model()
except Exception as exc:                      # a missing or unreadable artefact
    st.error('The model could not be loaded, so no numbers are shown. '
             'Run `python train.py` to rebuild models/model.joblib.')
    st.caption(f'Details: {exc}')
    st.stop()

history = get_history()
summary = mu.model_summary(meta)

with st.sidebar:
    st.header('Plan')
    start = st.date_input('First day to order for', value=date(2025, 1, 1))
    service_level = st.select_slider(
        'Service level', options=sorted(mu.SERVICE_MULTIPLIER), value=mu.DEFAULT_SERVICE_LEVEL,
        help='Higher means fewer stockouts and more waste. 74% comes from the cost ratio.')
    days = [(start + timedelta(days=i)).isoformat() for i in range(14)]
    promo_days = set(st.multiselect('Days with a published promotion', days))
    closed_days = set(st.multiselect('Days the counter is closed', days))

    st.divider()
    st.caption(f"Model: {summary['model']}, {summary['features']} features")
    st.caption(f"Trained: {summary['trained_at']}")
    st.caption(f"Test error: {summary['test_mae']} units ({summary['test_mape']}%)")

plan = mu.plan_fortnight(history, model, meta, start, promo_days, closed_days, service_level)

total_forecast = float(plan.forecast_units.sum())
total_order = float(plan.order_units.sum())

left, middle, right = st.columns(3)
left.metric('Fortnight forecast', f'{total_forecast:,.0f} units')
middle.metric('Fortnight order', f'{total_order:,.0f} units',
              f'{total_order - total_forecast:,.0f} safety stock')
right.metric('Service level', f'{service_level:.0%}')

st.subheader('What to order, day by day')
st.dataframe(plan, width='stretch', hide_index=True)

st.subheader('Recent sales, for context')
recent = history.tail(60)[['date', 'units_sold']].set_index('date')
st.line_chart(recent, height=220)

note = mu.environment_note(meta)
if 'matches' not in note:
    st.warning(note)

with st.expander('What this app does not do'):
    st.markdown(
        '- **No login.** Anyone with the link can see it, so it holds no personal data.\n'
        '- **No record of what it predicted.** Nobody can check later whether the forecasts '
        'were any good, which is the first thing to add.\n'
        '- **No retraining.** The model is a file in the repository, frozen at the date above.\n'
        '- **No alerting.** If the model drifts, the numbers simply get quietly worse.'
    )
