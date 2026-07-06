
# NPS Prediction App
# Taryam William Rodrigue Kabore | Artefact Junior Data Scientist Challenge 2026
#
# This app lets a retention analyst score any customer and get
# an NPS prediction (Detractor / Passive / Promoter) in real time.
# Every prediction is logged to a local SQLite database so we can
# track usage and feed the dashboard.
#
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import os
from datetime import datetime
import plotly.graph_objects as go

# ── file paths ────────────────────────────────────────────────────────────────
# using __file__ so the app works regardless of where it is launched from
BASE   = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")
DB     = os.path.join(BASE, "predictions.db")

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NPS Predictor | Artefact CI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── dark / light mode ─────────────────────────────────────────────────────────
# i store the mode in session_state so it persists across reruns
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

dark = st.session_state.dark_mode

# these variables make it easy to swap every colour at once when the mode changes
if dark:
    BG       = "linear-gradient(135deg, #0f0c29, #302b63, #24243e)"
    CARD_BG  = "rgba(255,255,255,0.07)"
    CARD_BR  = "rgba(255,255,255,0.12)"
    TEXT     = "#ffffff"
    SUBTEXT  = "rgba(255,255,255,0.55)"
    SIDEBAR  = "rgba(255,255,255,0.04)"
    ACCENT   = "#a78bfa"
    PLOT_BG  = "rgba(0,0,0,0)"
    FONT_CLR = "white"
else:
    BG       = "linear-gradient(135deg, #f0f4ff, #e8eaf6, #f5f5f5)"
    CARD_BG  = "rgba(255,255,255,0.9)"
    CARD_BR  = "rgba(0,0,0,0.08)"
    TEXT     = "#1a1a2e"
    SUBTEXT  = "rgba(0,0,0,0.5)"
    SIDEBAR  = "rgba(0,0,0,0.03)"
    ACCENT   = "#5c35d1"
    PLOT_BG  = "rgba(255,255,255,0)"
    FONT_CLR = "#1a1a2e"

# injecting CSS directly — streamlit does not expose enough styling hooks otherwise
st.markdown(f"""
<style>
    .stApp {{
        background: {BG};
        color: {TEXT};
    }}
    section[data-testid="stSidebar"] {{
        background: {SIDEBAR};
        border-right: 1px solid {CARD_BR};
    }}
    /* generic card used for KPIs and spend preview */
    .card {{
        background: {CARD_BG};
        border: 1px solid {CARD_BR};
        border-radius: 14px;
        padding: 20px;
        text-align: center;
    }}
    .card-value {{
        font-size: 2rem;
        font-weight: 700;
        margin: 6px 0 2px;
    }}
    .card-label {{
        font-size: 0.75rem;
        color: {SUBTEXT};
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }}
    /* section dividers */
    .section-label {{
        font-size: 0.85rem;
        font-weight: 600;
        color: {ACCENT};
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 28px 0 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid {CARD_BR};
    }}
    /* result box changes colour depending on predicted class */
    .result-box {{
        border-radius: 14px;
        padding: 28px;
        text-align: center;
        border: 2px solid;
    }}
    /* recommendation cards below the result */
    .rec-card {{
        background: {CARD_BG};
        border: 1px solid {CARD_BR};
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        height: 100%;
    }}
    /* predict button */
    .stButton > button {{
        width: 100%;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 13px;
        font-size: 0.95rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }}
    .stButton > button:hover {{ opacity: 0.88; }}
    /* hide streamlit default chrome */
    /* force widget labels (slider, selectbox) to follow theme */
    .stSlider label, .stSelectbox label, [data-testid="stWidgetLabel"] {{
        color: {TEXT} !important;
    }}
    .stSlider label p, .stSelectbox label p {{
        color: {TEXT} !important;
    }}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

# ── database helpers ──────────────────────────────────────────────────────────
# keeping all db logic in small functions makes it easier to test and change later

def init_db():
    """create the predictions table if it does not exist yet"""
    conn = sqlite3.connect(DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT,
            tenure          INTEGER,
            monthly_charges REAL,
            total_charges   REAL,
            contract        TEXT,
            internet        TEXT,
            senior          INTEGER,
            gender          TEXT,
            paperless       INTEGER,
            payment         TEXT,
            prediction      TEXT,
            prob_detractor  REAL,
            prob_passive    REAL,
            prob_promoter   REAL,
            confidence      REAL
        )
    """)
    conn.commit()
    conn.close()

def save_prediction(d):
    """log one prediction row to the database"""
    conn = sqlite3.connect(DB)
    conn.execute("""
        INSERT INTO predictions (
            timestamp, tenure, monthly_charges, total_charges,
            contract, internet, senior, gender, paperless, payment,
            prediction, prob_detractor, prob_passive, prob_promoter, confidence
        ) VALUES (
            :timestamp, :tenure, :monthly_charges, :total_charges,
            :contract, :internet, :senior, :gender, :paperless, :payment,
            :prediction, :prob_detractor, :prob_passive, :prob_promoter, :confidence
        )
    """, d)
    conn.commit()
    conn.close()

def load_recent(n=8):
    """pull the n most recent predictions for the history table"""
    if not os.path.exists(DB):
        return pd.DataFrame()
    conn = sqlite3.connect(DB)
    df = pd.read_sql(
        f"SELECT * FROM predictions ORDER BY id DESC LIMIT {n}", conn
    )
    conn.close()
    return df

def count_all():
    """total number of predictions ever made"""
    if not os.path.exists(DB):
        return 0
    conn = sqlite3.connect(DB)
    n = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    conn.close()
    return n

init_db()

# ── feature list ──────────────────────────────────────────────────────────────
# this must match exactly what was used during model training
# if the order changes the model will silently produce wrong predictions
FEATURE_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "PaperlessBilling", "MonthlyCharges", "TotalCharges", "charges_per_month",
    "Contract_Month-to-month", "Contract_One year", "Contract_Two year",
    "InternetService_DSL", "InternetService_Fiber optic", "InternetService_No",
    "PaymentMethod_Bank transfer (automatic)",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check", "PaymentMethod_Mailed check"
]

# ── model loading ─────────────────────────────────────────────────────────────
# cache_resource means streamlit only loads the model once per session
# not on every rerun — important for performance
@st.cache_resource
def load_model():
    mdl = joblib.load(os.path.join(MODELS, "nps_xgb_model.pkl"))
    le  = joblib.load(os.path.join(MODELS, "label_encoder.pkl"))
    return mdl, le

try:
    model, le = load_model()
    model_ok  = True
except Exception as e:
    model_ok  = False
    model_err = str(e)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"<h3 style='color:{ACCENT};margin-bottom:4px;'>NPS Predictor</h3>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p style='font-size:0.8rem;color:{SUBTEXT};'>"
        f"Artefact CI — Data Science Challenge</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # theme toggle — simple but effective
    mode_label = "Switch to Light Mode" if dark else "Switch to Dark Mode"
    if st.button(mode_label):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # show running total of predictions
    total_preds = count_all()
    st.markdown(f"""
    <div class="card">
        <div class="card-label">Total Predictions</div>
        <div class="card-value" style="color:{ACCENT};">{total_preds}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:0.82rem;line-height:2;color:{TEXT};'>
        <b>Model</b><br>XGBoost Classifier<br>
        <b>Macro F1</b><br>0.737<br>
        <b>Classes</b><br>Detractor · Passive · Promoter<br>
        <b>Author</b><br>Taryam W. R. Kabore
    </div>""", unsafe_allow_html=True)

# ── page header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<h1 style='text-align:center;font-size:2.2rem;font-weight:800;
    background:linear-gradient(90deg,#a78bfa,#60a5fa);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    margin-bottom:4px;'>
    Customer NPS Prediction
</h1>
<p style='text-align:center;color:{SUBTEXT};margin-bottom:32px;'>
    Score silent customers — prioritise detractors — drive retention actions
</p>""", unsafe_allow_html=True)

if not model_ok:
    st.error(f"Model loading failed: {model_err}")
    st.info("Make sure nps_xgb_model.pkl and label_encoder.pkl are in the models/ folder.")
    st.stop()

# ── input form ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Customer Profile</div>',
            unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"<p style='font-weight:600;color:{TEXT};'>Account</p>",
                unsafe_allow_html=True)
    tenure   = st.slider("Tenure (months)", 0, 72, 12,
                         help="How long this customer has been with us")
    monthly  = st.slider("Monthly Charges ($)", 18, 120, 65)
    contract = st.selectbox("Contract Type",
                            ["Month-to-month", "One year", "Two year"])

with col2:
    st.markdown(f"<p style='font-weight:600;color:{TEXT};'>Services</p>",
                unsafe_allow_html=True)
    internet  = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment   = st.selectbox("Payment Method",
                             ["Electronic check", "Mailed check",
                              "Bank transfer (automatic)",
                              "Credit card (automatic)"])

with col3:
    st.markdown(f"<p style='font-weight:600;color:{TEXT};'>Demographics</p>",
                unsafe_allow_html=True)
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    st.markdown("<br>", unsafe_allow_html=True)
    est_total = tenure * monthly
    st.markdown(f"""
    <div class="card">
        <div class="card-label">Estimated Total Spend</div>
        <div class="card-value" style="color:#60a5fa;">${est_total:,.0f}</div>
        <div style="font-size:0.78rem;opacity:0.5;">
            {tenure} months x ${monthly}/mo
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
run_btn = st.button("Run NPS Prediction", use_container_width=True)

# ── prediction logic ──────────────────────────────────────────────────────────
if run_btn:

    # build the feature row — everything not provided by the user defaults to 0
    # this mirrors the preprocessing done in the notebook
    total_charges = tenure * monthly
    cpp = total_charges / (tenure + 1)   # avg historical spend per month

    row = {c: 0 for c in FEATURE_COLS}
    row["gender"]            = 1 if gender == "Male" else 0
    row["SeniorCitizen"]     = 1 if senior == "Yes" else 0
    row["tenure"]            = tenure
    row["PhoneService"]      = 1   # assuming phone service by default
    row["PaperlessBilling"]  = 1 if paperless == "Yes" else 0
    row["MonthlyCharges"]    = monthly
    row["TotalCharges"]      = total_charges
    row["charges_per_month"] = cpp
    row[f"Contract_{contract}"]        = 1
    row[f"InternetService_{internet}"] = 1
    row[f"PaymentMethod_{payment}"]    = 1

    X          = pd.DataFrame([row])[FEATURE_COLS]
    pred_enc   = model.predict(X)[0]
    proba      = model.predict_proba(X)[0]
    label      = le.inverse_transform([pred_enc])[0]
    confidence = float(proba.max() * 100)

    # map probabilities to class names
    classes = list(le.classes_)
    p_det   = float(proba[classes.index("Detractor")] * 100)
    p_pas   = float(proba[classes.index("Passive")]   * 100)
    p_pro   = float(proba[classes.index("Promoter")]  * 100)

    # log to database before rendering — so it is saved even if the UI crashes
    save_prediction({
        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tenure":          tenure,
        "monthly_charges": monthly,
        "total_charges":   total_charges,
        "contract":        contract,
        "internet":        internet,
        "senior":          1 if senior == "Yes" else 0,
        "gender":          gender,
        "paperless":       1 if paperless == "Yes" else 0,
        "payment":         payment,
        "prediction":      label,
        "prob_detractor":  p_det,
        "prob_passive":    p_pas,
        "prob_promoter":   p_pro,
        "confidence":      confidence,
    })

    # ── result display ────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Prediction Result</div>',
                unsafe_allow_html=True)

    r1, r2 = st.columns([1, 1])

    COLOR_MAP = {
        "Detractor": ("#ff4b4b", "High churn risk — priority outreach required"),
        "Passive":   ("#ffa500", "Neutral — monitor and nurture proactively"),
        "Promoter":  ("#00c853", "Loyal customer — leverage for referral campaigns"),
    }
    color, message = COLOR_MAP[label]

    with r1:
        st.markdown(f"""
        <div class="result-box" style="border-color:{color};background:{color}18;">
            <div style="font-size:0.75rem;font-weight:600;letter-spacing:2px;
                 text-transform:uppercase;color:{color};margin-bottom:8px;">
                Predicted Category
            </div>
            <div style="font-size:2.2rem;font-weight:800;color:{color};">
                {label.upper()}
            </div>
            <div style="font-size:0.9rem;opacity:0.8;margin:10px 0 8px;">
                {message}
            </div>
            <div style="font-size:0.82rem;color:{color};font-weight:600;">
                Confidence: {confidence:.1f}%
            </div>
        </div>""", unsafe_allow_html=True)

    with r2:
        # bar chart with one bar per class — easier to read than a pie chart here
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Detractor", "Passive", "Promoter"],
            y=[p_det, p_pas, p_pro],
            marker_color=["#ff4b4b", "#ffa500", "#00c853"],
            text=[f"{v:.1f}%" for v in [p_det, p_pas, p_pro]],
            textposition="outside",
            textfont=dict(color=FONT_CLR if dark else "#1a1a2e", size=12),
        ))
        fig.update_layout(
            title=dict(text="Class Probabilities",
                       font=dict(color=FONT_CLR, size=13)),
            paper_bgcolor=PLOT_BG,
            plot_bgcolor=PLOT_BG,
            font=dict(color=FONT_CLR),
            yaxis=dict(range=[0, 115], showgrid=False,
                       ticksuffix="%", color=FONT_CLR),
            xaxis=dict(color=FONT_CLR),
            height=270,
            margin=dict(t=40, b=10, l=10, r=10),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── retention recommendations ─────────────────────────────────────────────
    # these are actionable — not generic — and change based on the prediction
    st.markdown('<div class="section-label">Retention Recommendation</div>',
                unsafe_allow_html=True)

    REC_MAP = {
        "Detractor": [
            ("Priority Call",
             "Contact within 24 hours. Identify the root cause of dissatisfaction."),
            ("Contract Upgrade",
             "Offer a one-year contract incentive to reduce cancellation risk."),
            ("Escalate",
             "Flag to senior retention team if monthly charges exceed $80."),
        ],
        "Passive": [
            ("Email Campaign",
             "Send a personalised offer within the week."),
            ("Service Upgrade",
             "Propose a value-added service to increase engagement."),
            ("Re-score in 30d",
             "Monitor and re-run prediction after 30 days."),
        ],
        "Promoter": [
            ("Referral Program",
             "Invite to refer friends — highest-value advocacy channel."),
            ("Loyalty Reward",
             "Offer a loyalty benefit to reinforce long-term commitment."),
            ("Case Study",
             "Consider for a testimonial or co-marketing opportunity."),
        ],
    }

    rec_cols = st.columns(3)
    for col, (title, desc) in zip(rec_cols, REC_MAP[label]):
        with col:
            st.markdown(f"""
            <div class="rec-card">
                <div style="font-size:0.7rem;font-weight:700;color:{ACCENT};
                     text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">
                    {title}
                </div>
                <div style="font-size:0.82rem;opacity:0.75;line-height:1.6;">
                    {desc}
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.success("Prediction saved to database.")

# ── recent predictions table ──────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Recent Predictions</div>',
            unsafe_allow_html=True)

recent = load_recent(8)
if recent.empty:
    st.info("No predictions yet — run the first one above.")
else:
    display = recent[[
        "timestamp", "tenure", "monthly_charges", "contract",
        "internet", "prediction", "confidence"
    ]].copy()
    display.columns = [
        "Timestamp", "Tenure", "Monthly $", "Contract",
        "Internet", "Prediction", "Confidence %"
    ]
    display["Confidence %"] = display["Confidence %"].round(1)
    display["Monthly $"] = display["Monthly $"].round(2)
    display["Tenure"] = display["Tenure"].astype(int)
    display["Monthly $"] = display["Monthly $"].round(2)
    display["Tenure"] = display["Tenure"].astype(int)
    display["Monthly $"] = display["Monthly $"].round(2)

    def color_pred(val):
        m = {
            "Detractor": "color:#ff4b4b;font-weight:700",
            "Passive":   "color:#ffa500;font-weight:700",
            "Promoter":  "color:#00c853;font-weight:700",
        }
        return m.get(val, "")

    st.dataframe(
        display.style.map(color_pred, subset=["Prediction"]),
        use_container_width=True,
        hide_index=True
    )

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown(
    f"<p style='text-align:center;opacity:0.3;font-size:0.75rem;"
    f"margin-top:40px;color:{TEXT};'>"
    f"NPS Predictor — Taryam W. R. Kabore — Artefact CI Challenge 2026</p>",
    unsafe_allow_html=True
)
