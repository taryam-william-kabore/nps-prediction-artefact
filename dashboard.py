
# NPS Prediction Dashboard
# Taryam William Rodrigue Kabore | Artefact Junior Data Scientist Challenge 2026
#
# Visualises all predictions logged by app.py
# Run: streamlit run dashboard.py --server.port 8502

import streamlit as st
import pandas as pd
import sqlite3
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE = os.path.dirname(os.path.abspath(__file__))
DB   = os.path.join(BASE, "predictions.db")

st.set_page_config(
    page_title="NPS Dashboard | Artefact CI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

dark = st.session_state.dark_mode

if dark:
    BG       = "linear-gradient(135deg, #0f0c29, #302b63, #24243e)"
    CARD_BG  = "rgba(255,255,255,0.07)"
    CARD_BR  = "rgba(255,255,255,0.12)"
    TEXT     = "#ffffff"
    SUBTEXT  = "rgba(255,255,255,0.6)"
    ACCENT   = "#a78bfa"
    PLOT_BG  = "rgba(0,0,0,0)"
    FC       = "#ffffff"
    PIE_LINE = "#0f0c29"
else:
    BG       = "#f4f6fb"
    CARD_BG  = "#ffffff"
    CARD_BR  = "rgba(0,0,0,0.09)"
    TEXT     = "#1a1a2e"
    SUBTEXT  = "#666688"
    ACCENT   = "#5c35d1"
    PLOT_BG  = "rgba(255,255,255,0)"
    FC       = "#1a1a2e"
    PIE_LINE = "#f4f6fb"

st.markdown(f"""
<style>
    .stApp {{ background: {BG}; color: {TEXT}; }}
    .kpi-card {{
        background: {CARD_BG};
        border: 1px solid {CARD_BR};
        border-radius: 16px;
        padding: 20px 14px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }}
    .kpi-value {{ font-size: 2.1rem; font-weight: 800; margin: 6px 0 2px; }}
    .kpi-label {{
        font-size: 0.72rem; color: {SUBTEXT};
        text-transform: uppercase; letter-spacing: 1.5px;
    }}
    .kpi-sub {{ font-size: 0.78rem; color: {SUBTEXT}; margin-top: 4px; }}
    .section-label {{
        font-size: 0.85rem; font-weight: 600; color: {ACCENT};
        text-transform: uppercase; letter-spacing: 2px;
        margin: 32px 0 14px; padding-bottom: 6px;
        border-bottom: 1px solid {CARD_BR};
    }}
    .stButton > button {{
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white !important; border: none;
        border-radius: 8px; padding: 8px 16px;
        font-size: 0.82rem; font-weight: 600;
    }}
    /* fix dataframe text in light mode */
    .stDataFrame {{ color: {TEXT} !important; }}
    [data-testid="stDataFrameResizable"] {{ color: {TEXT} !important; }}
    #MainMenu {{visibility:hidden;}} footer {{visibility:hidden;}} header {{visibility:hidden;}}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=10)
def load_data():
    if not os.path.exists(DB):
        return pd.DataFrame()
    conn = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM predictions ORDER BY id ASC", conn)
    conn.close()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

df = load_data()

# header row with toggle
col_title, col_toggle = st.columns([5, 1])
with col_title:
    st.markdown(f"""
    <h1 style='font-size:2rem;font-weight:800;
        background:linear-gradient(90deg,#a78bfa,#60a5fa,#34d399);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        margin-bottom:2px;'>NPS Prediction Dashboard</h1>
    <p style='color:{SUBTEXT};font-size:0.88rem;'>
        All predictions logged — Live analytics — Artefact Challenge 2026 — Taryam W. R. Kabore
    </p>""", unsafe_allow_html=True)

with col_toggle:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Light Mode" if dark else "Dark Mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

st.markdown(
    f"<hr style='border:none;border-top:1px solid {CARD_BR};margin:12px 0 24px;'>",
    unsafe_allow_html=True
)

if df.empty:
    st.warning("No predictions yet. Run app.py first!")
    st.stop()

total    = len(df)
n_det    = (df["prediction"] == "Detractor").sum()
n_pas    = (df["prediction"] == "Passive").sum()
n_pro    = (df["prediction"] == "Promoter").sum()
avg_conf = df["confidence"].mean()
nps_score = round((n_pro - n_det) / total * 100) if total else 0

COLORS = {"Detractor": "#ff4b4b", "Passive": "#ffa500", "Promoter": "#00c853"}

# KPIs
st.markdown('<div class="section-label">Key Metrics</div>', unsafe_allow_html=True)
k1,k2,k3,k4,k5,k6 = st.columns(6)
kpis = [
    (k1, str(total),          "#a78bfa", "Total Predictions",  "all runs"),
    (k2, str(n_det),          "#ff4b4b", "Detractors",         f"{n_det/total*100:.0f}% of total"),
    (k3, str(n_pas),          "#ffa500", "Passives",           f"{n_pas/total*100:.0f}% of total"),
    (k4, str(n_pro),          "#00c853", "Promoters",          f"{n_pro/total*100:.0f}% of total"),
    (k5, f"{avg_conf:.0f}%",  "#60a5fa", "Avg Confidence",     "model certainty"),
    (k6, f"{nps_score:+.0f}", "#f472b6", "Net NPS Score",      "promoters minus detractors"),
]
for col, val, color, label, sub in kpis:
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="color:{color};">{val}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

# shared chart layout - FC is the font color that changes with mode
def layout(fig, title, height=300):
    fig.update_layout(
        title=dict(text=title, font=dict(color=FC, size=13)),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=FC, size=11),
        legend=dict(font=dict(color=FC, size=11), title=""),
        xaxis=dict(color=FC, showgrid=False,
                   tickfont=dict(color=FC),
                   title_font=dict(color=FC)),
        yaxis=dict(color=FC, showgrid=False,
                   tickfont=dict(color=FC),
                   title_font=dict(color=FC)),
        height=height,
        margin=dict(t=40, b=10, l=10, r=10),
    )
    return fig

# row 1
st.markdown('<div class="section-label">Distribution and Trends</div>',
            unsafe_allow_html=True)
c1, c2, c3 = st.columns([1, 1.4, 1.6])

with c1:
    fig_d = go.Figure(go.Pie(
        labels=list(COLORS.keys()),
        values=[n_det, n_pas, n_pro],
        hole=0.55,
        marker=dict(
            colors=list(COLORS.values()),
            line=dict(color=PIE_LINE, width=2)
        ),
        textinfo="percent",
        textfont=dict(color=FC, size=12),
    ))
    fig_d.add_annotation(
        text=f"<b>{total}</b><br>total",
        x=0.5, y=0.5,
        font=dict(size=15, color=FC),
        showarrow=False
    )
    layout(fig_d, "NPS Split")
    st.plotly_chart(fig_d, use_container_width=True)

with c2:
    ct = df.groupby(["contract","prediction"]).size().reset_index(name="count")
    fig_b = px.bar(ct, x="contract", y="count", color="prediction",
                   color_discrete_map=COLORS, barmode="group")
    layout(fig_b, "Predictions by Contract Type")
    st.plotly_chart(fig_b, use_container_width=True)

with c3:
    tmp = df.copy().reset_index(drop=True)
    tmp["run"] = tmp.index + 1
    fig_l = px.line(tmp, x="run", y="confidence", color="prediction",
                    color_discrete_map=COLORS, markers=True)
    layout(fig_l, "Model Confidence per Run")
    fig_l.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig_l, use_container_width=True)

# row 2
st.markdown('<div class="section-label">Customer Profile Analysis</div>',
            unsafe_allow_html=True)
d1, d2 = st.columns([1.6, 1])

with d1:
    fig_sc = px.scatter(df, x="tenure", y="monthly_charges",
                        color="prediction", size="confidence",
                        color_discrete_map=COLORS,
                        hover_data=["contract","internet","confidence"])
    layout(fig_sc, "Tenure vs Monthly Charges (bubble size = confidence)", height=350)
    fig_sc.update_xaxes(title="Tenure (months)")
    fig_sc.update_yaxes(title="Monthly Charges ($)")
    st.plotly_chart(fig_sc, use_container_width=True)

with d2:
    fig_h = px.histogram(df, x="confidence", color="prediction",
                         nbins=20, color_discrete_map=COLORS,
                         barmode="overlay", opacity=0.75)
    layout(fig_h, "Confidence Distribution (%)", height=350)
    fig_h.update_xaxes(title="Confidence (%)")
    st.plotly_chart(fig_h, use_container_width=True)

# row 3
st.markdown('<div class="section-label">Segment Breakdown</div>',
            unsafe_allow_html=True)
e1, e2 = st.columns(2)

with e1:
    inet = df.groupby(["internet","prediction"]).size().reset_index(name="count")
    fig_i = px.bar(inet, x="internet", y="count", color="prediction",
                   color_discrete_map=COLORS, barmode="stack")
    layout(fig_i, "Predictions by Internet Service")
    st.plotly_chart(fig_i, use_container_width=True)

with e2:
    avgs = (df.groupby("prediction")
              .agg(avg_tenure=("tenure","mean"),
                   avg_monthly=("monthly_charges","mean"))
              .reset_index())
    fig_a = make_subplots(specs=[[{"secondary_y": True}]])
    fig_a.add_trace(go.Bar(
        x=avgs["prediction"], y=avgs["avg_tenure"].round(1),
        name="Avg Tenure (months)",
        marker_color=[COLORS.get(p,"#888") for p in avgs["prediction"]],
        text=avgs["avg_tenure"].round(1), textposition="outside",
        textfont=dict(color=FC),
    ), secondary_y=False)
    fig_a.add_trace(go.Scatter(
        x=avgs["prediction"], y=avgs["avg_monthly"].round(1),
        name="Avg Monthly ($)", mode="lines+markers",
        line=dict(color="#60a5fa", width=3),
        marker=dict(size=9, color="#60a5fa"),
    ), secondary_y=True)
    fig_a.update_layout(
        title=dict(text="Avg Tenure and Monthly Charges by Class",
                   font=dict(color=FC, size=13)),
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=FC),
        legend=dict(font=dict(color=FC)),
        xaxis=dict(color=FC, showgrid=False, tickfont=dict(color=FC)),
        yaxis=dict(color=FC, showgrid=False, title="Tenure (months)",
                   tickfont=dict(color=FC)),
        yaxis2=dict(color="#60a5fa", title="Monthly ($)",
                    tickfont=dict(color="#60a5fa")),
        height=320, margin=dict(t=40,b=10,l=10,r=10),
    )
    st.plotly_chart(fig_a, use_container_width=True)

# full log
st.markdown('<div class="section-label">Full Prediction Log</div>',
            unsafe_allow_html=True)

show = df[[
    "timestamp","tenure","monthly_charges","total_charges",
    "contract","internet","gender","prediction",
    "prob_detractor","prob_passive","prob_promoter","confidence"
]].copy().sort_values("timestamp", ascending=False)

show.columns = [
    "Timestamp","Tenure","Monthly $","Total $",
    "Contract","Internet","Gender","Prediction",
    "P(Det)%","P(Pas)%","P(Pro)%","Conf%"
]

# round all numeric columns properly
for c in ["Monthly $","Total $","P(Det)%","P(Pas)%","P(Pro)%","Conf%"]:
    show[c] = show[c].round(1)
show["Tenure"] = show["Tenure"].astype(int)

def color_pred(val):
    m = {
        "Detractor": "color:#ff4b4b;font-weight:700",
        "Passive":   "color:#ffa500;font-weight:700",
        "Promoter":  "color:#00c853;font-weight:700",
    }
    return m.get(val, f"color:{TEXT}")

st.dataframe(
    show.style.map(color_pred, subset=["Prediction"]),
    use_container_width=True,
    hide_index=True
)

csv = show.to_csv(index=False).encode("utf-8")
st.download_button(
    "Export to CSV", csv,
    file_name="nps_predictions_export.csv",
    mime="text/csv"
)

st.markdown(
    f"<p style='text-align:center;opacity:0.35;font-size:0.75rem;"
    f"margin-top:40px;color:{TEXT};'>"
    f"NPS Dashboard — Taryam W. R. Kabore — Artefact CI Challenge 2026</p>",
    unsafe_allow_html=True
)
