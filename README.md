# Customer NPS Prediction for a Telecom Operator

A machine learning system that predicts customer NPS category (Detractor / Passive / Promoter) from account and service data, built for the Artefact CI Junior Data Scientist take-home challenge.

## Problem

Telecom operators run regular NPS surveys, but only ~15% of customers respond. The business cannot target retention actions effectively when it only knows the NPS of a small fraction of its base. This project scores the silent 85% so the retention team can prioritise likely detractors before they churn.

## Approach

The challenge is framed as a 3-class classification problem. Since the available IBM Telco dataset does not include a Satisfaction Score, the NPS target is derived from behavioural signals (churn and tenure), with the churn column excluded from model features to avoid leakage.

- **Baseline:** Logistic Regression (Macro F1 = 0.721)
- **Final model:** XGBoost Classifier (Macro F1 = 0.737)
- **Interpretability:** SHAP values identify tenure, contract type, and internet service as the main drivers of detraction
- **Fairness:** Model audited for disparities across gender and senior-citizen segments

## Project Structure
## Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the prediction app:

```bash
streamlit run app.py
```

Run the analytics dashboard (in a separate terminal):

```bash
streamlit run dashboard.py --server.port 8502
```

## Features

- **Prediction app** — score any customer profile and get an NPS category with class probabilities and actionable retention recommendations. Every prediction is logged to a local SQLite database.
- **Dashboard** — live analytics over all logged predictions, with KPIs, distribution charts, and a full prediction log exportable to CSV.
- **Dark / light mode** — both interfaces support a theme toggle.

## Model Performance

| Metric | Logistic Regression | XGBoost |
|--------|--------------------|---------|
| Macro F1 | 0.721 | 0.737 |
| Accuracy | 75% | 78% |

## Limitations

- The NPS label is derived from behavioural proxies rather than actual survey responses, which introduces noise (particularly for the Passive class).
- The dataset is US-based; a pan-African operator would require retraining on local data.
- Hyperparameters were lightly tuned — a full cross-validated search would likely improve performance.

## Author

Taryam William Rodrigue Kabore
Junior Data Scientist candidate — Artefact CI
