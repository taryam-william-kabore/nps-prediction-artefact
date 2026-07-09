# Customer NPS Prediction for a Telecom Operator

A machine-learning system that predicts a customer's **NPS category** (Detractor / Passive / Promoter) from account and service data, built for the Artefact CI Junior Data Scientist take-home challenge.

## Problem

Telecom operators run regular NPS surveys, but only ~15% of customers respond. The business cannot target retention actions effectively when it knows the NPS of only a small fraction of its base. This project scores the silent ~85% so the retention team can prioritise likely detractors before they churn.

## Data

This project uses the **IBM Cognos Telco Customer Churn (11.1.3+)** dataset, which contains a real, human-provided **Satisfaction Score (1–5)** — unlike the widely-circulated 21-column Kaggle version, which lacks it.

*Note on my process: I initially used the Kaggle `blastchar` version and derived a proxy target from churn/tenure. On re-reading the brief (Section 4.1), I switched to the Cognos version and rebuilt the NPS target from the real Satisfaction Score, which removed a leakage problem in my first approach.*

The NPS target is mapped from the Satisfaction Score as specified in the brief:

| Satisfaction | NPS class |
|---|---|
| 5 | Promoter |
| 4 | Passive |
| ≤ 3 | Detractor |

This yields a realistically imbalanced target: **58% Detractor / 25% Passive / 16% Promoter** (Net NPS −42).

## Approach

Framed as a **3-class classification** problem.

- **Leakage handling:** all post-outcome columns (`Churn Label`, `Churn Value`, `Churn Score`, `Churn Reason`, `Customer Status`) and the target source (`Satisfaction Score`) are excluded from features. Geographic columns (`City`, `Zip Code`, `Latitude`, `Longitude`) are dropped as socio-economic proxies for fairness reasons.
- **Baseline:** Logistic Regression with balanced class weights (Macro F1 ≈ 0.41).
- **Main model:** XGBoost with balanced sample weights (Macro F1 ≈ 0.41).
- **Interpretability:** SHAP identifies online security, monthly charges, tenure, and contract type as the main drivers.
- **Fairness:** per-group Detractor recall audited across gender, senior-citizen, partner, and dependents segments.

### An honest note on performance

XGBoost matches but does not beat the logistic baseline (~0.41 Macro F1). This indicates the limit is the **features, not the model** — account data carries only weak signal about felt satisfaction. My earlier churn-derived approach produced an inflated ~0.74, which was a leakage artefact. The ~0.41 figure is the honest, leak-free result and the one I trust. This motivates the customer-verbatim extension (free text is where richer satisfaction signal lives).

## Project Structure

```
├── data/                          IBM Cognos Telco dataset (xlsx)
├── models/                        trained model, preprocessor, label encoder
├── assets/                        SHAP + distribution figures
├── nps_prediction.ipynb           full pipeline: raw data -> trained model
├── app.py                         Streamlit prediction app
├── dashboard.py                   Streamlit analytics dashboard
├── requirements.txt
└── runtime.txt
```

## Running Locally

Install dependencies:
```
pip install -r requirements.txt
```
Run the prediction app:
```
streamlit run app.py
```
Run the analytics dashboard (separate terminal):
```
streamlit run dashboard.py --server.port 8502
```

## Features

- **Prediction app** — score any customer profile; returns an NPS category with class probabilities and actionable retention recommendations. Predictions are logged to a local SQLite database. At inference the app loads the same fitted preprocessor used in training, ensuring train/serve consistency.
- **Dashboard** — analytics over all logged predictions: KPIs, distribution charts, and an exportable prediction log.
- **Dark / light mode** on both interfaces.

## Model Performance

| Metric | Logistic Regression | XGBoost |
|---|---|---|
| Macro F1 | 0.41 | 0.41 |
| Accuracy | 0.46 | 0.46 |

## Limitations

- The target is derived from a 1–5 satisfaction score rather than true 0–10 NPS survey responses, adding noise (especially for the Passive class).
- Account data weakly predicts satisfaction; the Passive class is hardest to separate.
- A fairness gap exists on the Dependents segment (lower Detractor recall) that would need mitigation before production.
- Geography was dropped for fairness at some accuracy cost — a defensible but reversible choice.

## Use of AI tools

In line with the challenge's guidance, I used an LLM as a pair-programming assistant to help scaffold code, debug, and structure the notebook and apps. All modelling decisions — target construction, leakage handling, feature selection, evaluation, and interpretation — are my own, and I have verified every result.

## Author

**Taryam William Rodrigue Kabore** — Junior Data Scientist candidate, Artefact CI
