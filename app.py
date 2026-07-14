
# NPS Prediction App
# Taryam William Rodrigue Kabore | Artefact Junior Data Scientist Challenge 2026
#
# This app lets a retention analyst score any customer and get
# an NPS prediction (Detractor / Passive / Promoter) in real time.
# Every prediction is logged to a local SQLite database so we can
# track usage and feed the dashboard.
#
# The model is trained on the IBM Cognos Telco dataset (11.1.3+), with the
# NPS target built from the real Satisfaction Score. At prediction time I load
# the SAME preprocessor that was fitted during training and pass raw inputs
# through it — this guarantees the app encodes features exactly like training
# did, instead of me re-implementing the encoding by hand (which is fragile).
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
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

dark = st.session_state.dark_mode

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

st.markdown(f"""
<style>
    .stApp {{ background: {BG}; color: {TEXT}; }}
    section[data-testid="stSidebar"] {{
        background: {SIDEBAR};
        border-right: 1px solid {CARD_BR};
        min-width: 300px !important;
        max-width: 300px !important;
        transform: none !important;
        visibility: visible !important;
        margin-left: 0px !important;
    }}
    section[data-testid="stSidebar"][aria-expanded="false"] {{
        transform: none !important;
        visibility: visible !important;
        margin-left: 0px !important;
    }}
    /* keep the header/collapse control usable so the sidebar toggle works */
    header[data-testid="stHeader"] {{ background: transparent; }}
    .card {{
        background: {CARD_BG};
        border: 1px solid {CARD_BR};
        border-radius: 14px;
        padding: 20px;
        text-align: center;
    }}
    .card-value {{ font-size: 2rem; font-weight: 700; margin: 6px 0 2px; }}
    .card-label {{
        font-size: 0.75rem; color: {SUBTEXT};
        text-transform: uppercase; letter-spacing: 1.5px;
    }}
    .section-label {{
        font-size: 0.85rem; font-weight: 600; color: {ACCENT};
        text-transform: uppercase; letter-spacing: 2px;
        margin: 28px 0 12px; padding-bottom: 6px;
        border-bottom: 1px solid {CARD_BR};
    }}
    .result-box {{
        border-radius: 14px; padding: 28px; text-align: center; border: 2px solid;
    }}
    .rec-card {{
        background: {CARD_BG}; border: 1px solid {CARD_BR};
        border-radius: 12px; padding: 18px; text-align: center; height: 100%;
    }}
    .stButton > button {{
        width: 100%;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; border: none; border-radius: 10px;
        padding: 13px; font-size: 0.95rem; font-weight: 600; letter-spacing: 0.5px;
    }}
    .stButton > button:hover {{ opacity: 0.88; }}
    .stSlider label, .stSelectbox label, [data-testid="stWidgetLabel"] {{ color: {TEXT} !important; }}
    .stSlider label p, .stSelectbox label p {{ color: {TEXT} !important; }}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    [data-testid="stToolbar"] {{visibility: hidden;}}
    [data-testid="stDecoration"] {{display: none;}}
</style>
""", unsafe_allow_html=True)

# ── database helpers ──────────────────────────────────────────────────────────
# NOTE: schema is unchanged from the original app, so the dashboard keeps working.

def init_db():
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
    conn = sqlite3.connect(DB)
    conn.execute("""
        INSERT INTO predictions (
            timestamp, tenure, monthly_charges, total_charges, contract,
            internet, senior, gender, paperless, payment,
            prediction, prob_detractor, prob_passive, prob_promoter, confidence
        ) VALUES (
            :timestamp, :tenure, :monthly_charges, :total_charges, :contract,
            :internet, :senior, :gender, :paperless, :payment,
            :prediction, :prob_detractor, :prob_passive, :prob_promoter, :confidence
        )
    """, d)
    conn.commit()
    conn.close()

def load_recent(n=8):
    if not os.path.exists(DB):
        return pd.DataFrame()
    conn = sqlite3.connect(DB)
    df = pd.read_sql(f"SELECT * FROM predictions ORDER BY id DESC LIMIT {n}", conn)
    conn.close()
    return df

def count_all():
    if not os.path.exists(DB):
        return 0
    conn = sqlite3.connect(DB)
    n = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    conn.close()
    return n

init_db()

# ── model loading ─────────────────────────────────────────────────────────────
# We load THREE artifacts saved by the notebook:
#   - the trained XGBoost model
#   - the fitted preprocessor (does the scaling + one-hot encoding)
#   - the label encoder (maps 0/1/2 back to Detractor/Passive/Promoter)
@st.cache_resource
def load_model():
    mdl = joblib.load(os.path.join(MODELS, "nps_xgb_model.pkl"))
    pre = joblib.load(os.path.join(MODELS, "preprocessor.pkl"))
    le  = joblib.load(os.path.join(MODELS, "label_encoder.pkl"))
    return mdl, pre, le

try:
    model, preprocessor, le = load_model()
    model_ok = True
except Exception as e:
    model_ok = False
    model_err = str(e)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<h3 style='color:{ACCENT};margin-bottom:4px;'>NPS Predictor</h3>",
                unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:0.8rem;color:{SUBTEXT};'>"
                f"Artefact CI — Data Science Challenge</p>", unsafe_allow_html=True)
    st.markdown("---")

    mode_label = "Switch to Light Mode" if dark else "Switch to Dark Mode"
    if st.button(mode_label):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    total_preds = count_all()
    st.markdown(f"""
    <div class="card">
        <div class="card-label">Total Predictions</div>
        <div class="card-value" style="color:{ACCENT};">{total_preds}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    # Honest model stats: real leak-free numbers from the rebuilt notebook.
    st.markdown(f"""
    <div style='font-size:0.82rem;line-height:2;color:{TEXT};'>
        <b>Model</b><br>XGBoost Classifier<br>
        <b>Target</b><br>NPS from Satisfaction Score<br>
        <b>Macro F1</b><br>0.41 (honest, leak-free)<br>
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
    st.info("Make sure nps_xgb_model.pkl, preprocessor.pkl and label_encoder.pkl are in models/.")
    st.stop()

# ── input form ────────────────────────────────────────────────────────────────
# The model needs 19 features. I group them like the customer's real profile:
# Account, Services, Demographics. Service add-ons default to sensible values so
# the analyst only changes what they know.
st.markdown('<div class="section-label">Customer Profile</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"<p style='font-weight:600;color:{TEXT};'>Account</p>", unsafe_allow_html=True)
    tenure   = st.slider("Tenure (months)", 0, 72, 12,
                         help="How long this customer has been with us")
    monthly  = st.slider("Monthly Charges ($)", 18, 120, 65)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment  = st.selectbox("Payment Method",
                            ["Bank Withdrawal", "Credit Card", "Mailed Check"])

with col2:
    st.markdown(f"<p style='font-weight:600;color:{TEXT};'>Services</p>", unsafe_allow_html=True)
    internet   = st.selectbox("Internet Service", ["DSL", "Fiber Optic", "Cable", "No"])
    online_sec = st.selectbox("Online Security", ["Yes", "No"])
    tech_sup   = st.selectbox("Tech Support", ["Yes", "No"])
    online_bkp = st.selectbox("Online Backup", ["Yes", "No"])
    paperless  = st.selectbox("Paperless Billing", ["No", "Yes"])

with col3:
    st.markdown(f"<p style='font-weight:600;color:{TEXT};'>Demographics</p>", unsafe_allow_html=True)
    gender     = st.selectbox("Gender", ["Female", "Male"])
    senior     = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner    = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    est_total  = tenure * monthly
    st.markdown(f"""
    <div class="card">
        <div class="card-label">Estimated Total Spend</div>
        <div class="card-value" style="color:#60a5fa;">${est_total:,.0f}</div>
        <div style="font-size:0.78rem;opacity:0.5;">{tenure} months x ${monthly}/mo</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
run_btn = st.button("Run NPS Prediction", use_container_width=True)

# ── prediction logic ──────────────────────────────────────────────────────────
if run_btn:
    total_charges = tenure * monthly

    # Build a ONE-ROW dataframe using the EXACT raw column names the preprocessor
    # was fitted on. Service add-ons the form doesn't ask about get sensible
    # defaults. The preprocessor then handles all scaling + one-hot encoding.
    row = {
        "Tenure Months":     tenure,
        "Monthly Charges":   monthly,
        "Total Charges":     total_charges,
        "Gender":            gender,
        "Senior Citizen":    "Yes" if senior == "Yes" else "No",
        "Partner":           partner,
        "Dependents":        dependents,
        "Phone Service":     "Yes",
        "Multiple Lines":    "No",
        "Internet Service":  internet,
        "Online Security":   online_sec,
        "Online Backup":     online_bkp,
        "Device Protection": "No",
        "Tech Support":      tech_sup,
        "Streaming TV":      "No",
        "Streaming Movies":  "No",
        "Contract":          contract,
        "Paperless Billing": paperless,
        "Payment Method":    payment,
    }
    X_raw = pd.DataFrame([row])
    X_p   = preprocessor.transform(X_raw)   # same preprocessing as training

    pred_enc   = model.predict(X_p)[0]
    proba      = model.predict_proba(X_p)[0]
    label      = le.inverse_transform([pred_enc])[0]
    confidence = float(proba.max() * 100)

    classes = list(le.classes_)
    p_det = float(proba[classes.index("Detractor")] * 100)
    p_pas = float(proba[classes.index("Passive")]   * 100)
    p_pro = float(proba[classes.index("Promoter")]  * 100)

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

    st.markdown('<div class="section-label">Prediction Result</div>', unsafe_allow_html=True)
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
                Predicted Category</div>
            <div style="font-size:2.2rem;font-weight:800;color:{color};">{label.upper()}</div>
            <div style="font-size:0.9rem;opacity:0.8;margin:10px 0 8px;">{message}</div>
            <div style="font-size:0.82rem;color:{color};font-weight:600;">
                Confidence: {confidence:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    with r2:
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
            title=dict(text="Class Probabilities", font=dict(color=FONT_CLR, size=13)),
            paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG, font=dict(color=FONT_CLR),
            yaxis=dict(range=[0, 115], showgrid=False, ticksuffix="%", color=FONT_CLR),
            xaxis=dict(color=FONT_CLR), height=270,
            margin=dict(t=40, b=10, l=10, r=10), showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-label">Retention Recommendation</div>', unsafe_allow_html=True)
    REC_MAP = {
        "Detractor": [
            ("Priority Call", "Contact within 24 hours. Identify the root cause of dissatisfaction."),
            ("Add Security/Support", "Offer online security and tech support — top SHAP drivers of detraction."),
            ("Contract Upgrade", "Offer a one-year contract incentive to reduce churn risk."),
        ],
        "Passive": [
            ("Email Campaign", "Send a personalised offer within the week."),
            ("Service Upgrade", "Propose a value-added service to increase engagement."),
            ("Re-score in 30d", "Monitor and re-run prediction after 30 days."),
        ],
        "Promoter": [
            ("Referral Program", "Invite to refer friends — highest-value advocacy channel."),
            ("Loyalty Reward", "Offer a loyalty benefit to reinforce long-term commitment."),
            ("Case Study", "Consider for a testimonial or co-marketing opportunity."),
        ],
    }
    rec_cols = st.columns(3)
    for col, (title, desc) in zip(rec_cols, REC_MAP[label]):
        with col:
            st.markdown(f"""
            <div class="rec-card">
                <div style="font-size:0.7rem;font-weight:700;color:{ACCENT};
                     text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">{title}</div>
                <div style="font-size:0.82rem;opacity:0.75;line-height:1.6;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.success("Prediction saved to database.")

# ── recent predictions table ──────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Recent Predictions</div>', unsafe_allow_html=True)

recent = load_recent(8)
if recent.empty:
    st.info("No predictions yet — run the first one above.")
else:
    display = recent[["timestamp","tenure","monthly_charges","contract",
                      "internet","prediction","confidence"]].copy()
    display.columns = ["Timestamp","Tenure","Monthly $","Contract",
                       "Internet","Prediction","Confidence %"]
    display["Confidence %"] = display["Confidence %"].round(1)
    display["Monthly $"] = display["Monthly $"].round(2)
    display["Tenure"] = display["Tenure"].astype(int)

    def color_pred(val):
        m = {"Detractor":"color:#ff4b4b;font-weight:700",
             "Passive":"color:#ffa500;font-weight:700",
             "Promoter":"color:#00c853;font-weight:700"}
        return m.get(val, "")

    st.dataframe(display.style.map(color_pred, subset=["Prediction"]),
                 use_container_width=True, hide_index=True)

st.markdown(
    f"<p style='text-align:center;opacity:0.3;font-size:0.75rem;"
    f"margin-top:40px;color:{TEXT};'>"
    f"NPS Predictor — Taryam W. R. Kabore — Artefact CI Challenge 2026</p>",
    unsafe_allow_html=True
)
