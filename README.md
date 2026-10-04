# Store 14 Bakery Demand Planner

The Week 23 forecasting model, turned into something the counter manager can open
in a browser. Same data, same model, same 14 day horizon, same 74% service level.
What is new is that somebody who is not you can use it.

```
streamlit_app.py      the screen. Streamlit Community Cloud runs this file
model_utils.py        loading, features, forecasting, order quantity
train.py              rebuilds models/model.joblib from the data
models/               the model bundle and its feature contract
data/                 the sales history the app reads
requirements.txt      what the app needs, and why some pins are exact
```

## 1. Run it on your own machine

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

A browser tab opens at `http://localhost:8501`. Pick a start date, tick any days
with a promotion, and read the **order** column rather than the forecast column.

## 2. Retrain the model, if you want to

```bash
python train.py              # saves models/model.joblib and models/model_meta.json
python train.py --compare    # four settings side by side, then saves the best
```

`train.py` prints the scikit-learn version it used. If that differs from the pin
in `requirements.txt`, change the pin to match. The saved model is a pickle, and a
pickle belongs to the library version that made it.

## 3. Put it on the internet, free

Streamlit Community Cloud runs an app straight from a public GitHub repository.

**Create the repository.** From this folder:

```bash
git init
git add .
git commit -m "Bakery demand planner"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/bakery-demand-planner.git
git push -u origin main
```

Create the empty repository on github.com first, and make it **public**.

**Deploy it.**

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with your GitHub account.
2. Choose **Create app**, then **Deploy a public app from GitHub**.
3. Repository: `YOUR-USERNAME/bakery-demand-planner`. Branch: `main`.
   Main file path: `streamlit_app.py`.
4. Open **Advanced settings** and set the Python version to the one you developed
   on. `python --version` tells you what that is.
5. Click **Deploy**. The first build takes a few minutes while it installs the
   requirements. The logs are on the right, and they are where any failure shows up.

You get a URL ending in `.streamlit.app`. Send it to somebody.

**Changing it later.** Push to `main` and the deployed app updates itself. There
is no second deployment step.

## 4. When it does not work

| What you see | What it means | What to do |
| --- | --- | --- |
| Build fails installing requirements | The pinned versions do not exist for the Python version chosen | Use the Python version you developed on, and keep loose pins for everything except scikit-learn |
| App loads, then shows a model error | `models/model.joblib` was not committed | Check it is in the repository on github.com, not ignored |
| Warning about scikit-learn versions | The cloud installed a different version from the one that trained the model | Set the pin in `requirements.txt` to the version `train.py` printed |
| "This app has gone to sleep" | No traffic for 12 hours | Click the button to wake it. This is normal on the free tier |
| Everything is slow with a big file | Community Cloud gives about 2.7 GB of memory | Keep the data small, and cache loading with `st.cache_data` |

## What this project deliberately does not do

No login, so it holds no personal data. No record of what it predicted, so nobody
can check later whether the forecasts were any good. No retraining, so the model is
frozen at the date in `models/model_meta.json`. No alerting, so if it drifts the
numbers quietly get worse.

Those are real gaps, and naming them is part of the work. The appendix in the
course pack explains the tools that close them, and when they start to be worth it.
