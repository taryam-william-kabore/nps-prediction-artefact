# ============================================================
# NPS Prediction Dashboard
# Taryam William Rodrigue Kabore
# Artefact CI - Junior Data Scientist Challenge 2026
# ============================================================
#
# What this file does:
#   This is the "analytics" side of my project. app.py makes a single
#   prediction for one customer at a time and saves it. This dashboard
#   reads ALL the saved predictions and turns them into charts so you can
#   see the big picture (how many detractors, which contract types are
#   risky, etc.) instead of looking at one prediction at a time.
#
# How to run it locally:
#   streamlit run dashboard.py --server.port 8502
#   (I use a different port because app.py already uses the default 8501,
#    so I can have both open at the same time.)

import streamlit as st
import pandas as pd
import sqlite3
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# I build the paths from this file's own location instead of hard-coding
# something like "C:/Users/.../predictions.db". That way the same code works
# on my laptop AND on the Streamlit cloud server (which is Linux, different
# folders). os.path.dirname(__file__) = "the folder this script lives in".
BASE = os.path.dirname(os.path.abspath(__file__))
DB   = os.path.join(BASE, "predictions.db")          # the live prediction log
SAMPLE_CSV = os.path.join(BASE, "sample_predictions.csv")  # backup demo data (explained below)

st.set_page_config(
    page_title="NPS Dashboard | Artefact CI",
    layout="wide",                      # wide = use the full screen for charts
    initial_sidebar_state="collapsed"
)

# --- Theme handling (dark / light mode) ---
# I keep the current mode in session_state so it survives when Streamlit
# re-runs the script (Streamlit re-runs the whole file top-to-bottom on every
# click, so a normal variable would reset - session_state is how you remember
# things between re-runs).
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True   # start in dark mode by default

dark = st.session_state.dark_mode

# Instead of writing two separate stylesheets, I just swap the colour values
# depending on the mode, then reuse the same variable names everywhere below.
# FC = "font colour" - the one that has to flip between white and dark so the
# chart text stays readable in both modes (this was the fiddly part).
if dark:
    BG       = "linear-gradient(135deg, #0f0c29, #302b63, #24243e)"
    CARD_BG  = "rgba(255,255,255,0.07)"
    CARD_BR  = "rgba(255,255,255,0.12)"
    TEXT     = "#ffffff"
    SUBTEXT  = "rgba(255,255,255,0.6)"
    ACCENT   = "#a78bfa"
    PLOT_BG  = "rgba(0,0,0,0)"          # transparent so the page gradient shows through
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

# All my custom CSS in one block. Streamlit doesn't give nice "cards" out of
# the box, so I style my own with a semi-transparent background + border to get
# that glassy look. The f-string lets me drop the theme colours straight in.
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
    .sample-banner {{
        background: rgba(167,139,250,0.12);
        border: 1px solid rgba(167,139,250,0.35);
        border-radius: 10px;
        padding: 10px 16px;
        font-size: 0.82rem;
        color: {SUBTEXT};
        margin-bottom: 8px;
    }}
    .stButton > button {{
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white !important; border: none;
        border-radius: 8px; padding: 8px 16px;
        font-size: 0.82rem; font-weight: 600;
    }}
    /* Streamlit's dataframe text stayed dark-on-dark in dark mode until I
       forced the colour here - took me a while to find the right selector. */
    .stDataFrame {{ color: {TEXT} !important; }}
    [data-testid="stDataFrameResizable"] {{ color: {TEXT} !important; }}
    /* hide Streamlit's default menu/footer/header for a cleaner look */
    #MainMenu {{visibility:hidden;}} footer {{visibility:hidden;}} header {{visibility:hidden;}}
</style>
""", unsafe_allow_html=True)


# @st.cache_data means: don't re-read the database on every single re-run,
# reuse the result for 10 seconds (ttl=10). Without this the dashboard would
# hit the DB constantly and feel slow. 10s is short enough that new predictions
# still show up quickly.
@st.cache_data(ttl=10)
def load_data():
    """
    Load the predictions to display, and tell the caller whether we're showing
    real data or sample data (that's the second return value, is_sample).

    Why the two-step logic below:
    On my laptop, app.py and this dashboard share the same folder, so they share
    predictions.db - the dashboard sees every real prediction I make. But on
    Streamlit Cloud each app runs in its OWN separate container with its OWN
    filesystem, so the dashboard literally cannot see the predictor's database.
    That's not a bug, it's just how their free hosting works.

    So: try the real database first. If it's there and has rows, use it.
    Otherwise fall back to a small sample CSV I committed to the repo, purely so
    the charts have something to show online. I label it clearly as demo data
    further down so nobody mistakes it for real logged predictions.
    """
    # 1) Try the real logged predictions first
    if os.path.exists(DB):
        try:
            conn = sqlite3.connect(DB)
            df = pd.read_sql("SELECT * FROM predictions ORDER BY id ASC", conn)
            conn.close()
            if not df.empty:
                # timestamp comes out of SQLite as plain text; convert it to a
                # real datetime so the time-based sorting/charts work properly.
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                return df, False   # False = this is REAL data
        except Exception:
            # if anything goes wrong reading the DB, don't crash the page -
            # just fall through to the sample data below.
            pass

    # 2) Fall back to the committed sample file (the cloud / first-run case)
    if os.path.exists(SAMPLE_CSV):
        df = pd.read_csv(SAMPLE_CSV)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df, True            # True = this is SAMPLE/demo data

    # 3) Nothing at all available
    return pd.DataFrame(), False

df, is_sample = load_data()

# --- Header: title on the left, dark/light toggle button on the right ---
# st.columns([5, 1]) splits the row so the title gets 5x the width of the button.
col_title, col_toggle = st.columns([5, 1])
with col_title:
    # I use a gradient-filled heading (the -webkit-background-clip trick paints
    # the gradient onto the text itself instead of behind it).
    st.markdown(f"""
    <h1 style='font-size:2rem;font-weight:800;
        background:linear-gradient(90deg,#a78bfa,#60a5fa,#34d399);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        margin-bottom:2px;'>NPS Prediction Dashboard</h1>
    <p style='color:{SUBTEXT};font-size:0.88rem;'>
        All predictions logged — Live analytics — Artefact Challenge 2026 — Taryam W. R. Kabore
    </p>""", unsafe_allow_html=True)

with col_toggle:
    st.markdown("<br>", unsafe_allow_html=True)   # tiny spacer to line the button up
    # Clicking flips the mode and re-runs the script so the new colours apply.
    if st.button("Light Mode" if dark else "Dark Mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

st.markdown(
    f"<hr style='border:none;border-top:1px solid {CARD_BR};margin:12px 0 24px;'>",
    unsafe_allow_html=True
)

# If there's genuinely no data (no DB and no sample file), stop here with a
# friendly message instead of crashing on the empty charts below.
if df.empty:
    st.warning("No predictions yet. Run app.py to log some predictions.")
    st.stop()

# Be transparent: if we're on the sample fallback, say so out loud.
if is_sample:
    st.markdown(
        "<div class='sample-banner'>Showing <b>demonstration data</b> "
        "(50 representative predictions). In the deployed predictor app, and when "
        "running locally, real predictions are logged live to SQLite and appear here "
        "automatically.</div>",
        unsafe_allow_html=True
    )

# --- Headline numbers (the KPI cards) ---
# Count how many of each class, the average confidence, and the overall NPS.
total    = len(df)
n_det    = (df["prediction"] == "Detractor").sum()
n_pas    = (df["prediction"] == "Passive").sum()
n_pro    = (df["prediction"] == "Promoter").sum()
avg_conf = df["confidence"].mean()
# Net Promoter Score formula: (% promoters - % detractors). Passives don't count
# towards NPS by definition, which is why they're not in this calculation.
nps_score = round((n_pro - n_det) / total * 100) if total else 0

# One colour per class, reused in every chart so the meaning stays consistent
# (red = bad/detractor, orange = neutral/passive, green = good/promoter).
COLORS = {"Detractor": "#ff4b4b", "Passive": "#ffa500", "Promoter": "#00c853"}

st.markdown('<div class="section-label">Key Metrics</div>', unsafe_allow_html=True)
k1,k2,k3,k4,k5,k6 = st.columns(6)   # 6 cards side by side
# I put the card data in a list of tuples then loop, instead of copy-pasting the
# same HTML block 6 times. Each tuple = (column, big number, colour, label, subtitle).
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


# Every Plotly chart needs the same styling (transparent background, correct
# font colour for the current theme, etc). Rather than repeat ~12 lines per
# chart, I wrote this helper once and call it on each figure. DRY principle.
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

# ============================================================
# ROW 1 - overall distribution + trends
# ============================================================
st.markdown('<div class="section-label">Distribution and Trends</div>',
            unsafe_allow_html=True)
c1, c2, c3 = st.columns([1, 1.4, 1.6])   # uneven widths so each chart gets room

with c1:
    # Donut chart of the class split. A donut (hole=0.55) lets me put the total
    # count in the middle, which I think reads nicer than a plain pie.
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
    # Grouped bar: for each contract type, how many of each class.
    # This is the most business-relevant chart - it shows month-to-month
    # customers are the detractor-heavy group, which is the key retention insight.
    ct = df.groupby(["contract","prediction"]).size().reset_index(name="count")
    fig_b = px.bar(ct, x="contract", y="count", color="prediction",
                   color_discrete_map=COLORS, barmode="group")
    layout(fig_b, "Predictions by Contract Type")
    st.plotly_chart(fig_b, use_container_width=True)

with c3:
    # Confidence over time / per run. I add a simple "run" counter (1,2,3...)
    # as the x-axis so you can see if the model is more/less confident on
    # certain predictions.
    tmp = df.copy().reset_index(drop=True)
    tmp["run"] = tmp.index + 1
    fig_l = px.line(tmp, x="run", y="confidence", color="prediction",
                    color_discrete_map=COLORS, markers=True)
    layout(fig_l, "Model Confidence per Run")
    fig_l.update_yaxes(ticksuffix="%", color=FC, tickfont=dict(color=FC), title_font=dict(color=FC))
    st.plotly_chart(fig_l, use_container_width=True)

# ============================================================
# ROW 2 - digging into the customer profiles
# ============================================================
st.markdown('<div class="section-label">Customer Profile Analysis</div>',
            unsafe_allow_html=True)
d1, d2 = st.columns([1.6, 1])

with d1:
    # Scatter of tenure vs monthly charges, coloured by class, bubble size =
    # confidence. This is my favourite chart because you can literally SEE the
    # pattern: detractors cluster at low tenure (left side), promoters at high
    # tenure (right side). It visually confirms tenure is the main driver.
    fig_sc = px.scatter(df, x="tenure", y="monthly_charges",
                        color="prediction", size="confidence",
                        color_discrete_map=COLORS,
                        hover_data=["contract","internet","confidence"])
    layout(fig_sc, "Tenure vs Monthly Charges (bubble size = confidence)", height=350)
    fig_sc.update_xaxes(title="Tenure (months)", color=FC, tickfont=dict(color=FC), title_font=dict(color=FC))
    fig_sc.update_yaxes(title="Monthly Charges ($)", color=FC, tickfont=dict(color=FC), title_font=dict(color=FC))
    st.plotly_chart(fig_sc, use_container_width=True)

with d2:
    # Histogram of confidence values. Overlaid + semi-transparent so I can see
    # where each class's confidence tends to sit.
    fig_h = px.histogram(df, x="confidence", color="prediction",
                         nbins=20, color_discrete_map=COLORS,
                         barmode="overlay", opacity=0.75)
    layout(fig_h, "Confidence Distribution (%)", height=350)
    fig_h.update_xaxes(title="Confidence (%)", color=FC, tickfont=dict(color=FC), title_font=dict(color=FC))
    st.plotly_chart(fig_h, use_container_width=True)

# ============================================================
# ROW 3 - segment breakdowns
# ============================================================
st.markdown('<div class="section-label">Segment Breakdown</div>',
            unsafe_allow_html=True)
e1, e2 = st.columns(2)

with e1:
    # Stacked bar by internet service type. Stacked (not grouped) here because I
    # care about the total per internet type as well as the split within it.
    inet = df.groupby(["internet","prediction"]).size().reset_index(name="count")
    fig_i = px.bar(inet, x="internet", y="count", color="prediction",
                   color_discrete_map=COLORS, barmode="stack")
    layout(fig_i, "Predictions by Internet Service")
    st.plotly_chart(fig_i, use_container_width=True)

with e2:
    # This one needs TWO y-axes because tenure (months, ~0-70) and monthly
    # charges ($, ~20-120) are on totally different scales - plotting them on
    # one axis would squash the smaller one. make_subplots(secondary_y=True)
    # gives me a left axis for tenure (bars) and a right axis for charges (line).
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
    # I couldn't use my layout() helper here because of the second y-axis, so I
    # set the layout manually for this one chart.
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

# ============================================================
# Full prediction log (the raw table) + CSV export
# ============================================================
st.markdown('<div class="section-label">Full Prediction Log</div>',
            unsafe_allow_html=True)

# Pick the columns I want to show and rename them to short, readable headers.
# Newest first (ascending=False on timestamp).
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

# Round the number columns so the table isn't full of long decimals, and make
# tenure a whole number (you can't have 12.0 months, it's just 12).
for c in ["Monthly $","Total $","P(Det)%","P(Pas)%","P(Pro)%","Conf%"]:
    show[c] = show[c].round(1)
show["Tenure"] = show["Tenure"].astype(int)

# Colour the Prediction column text by class so the table is scannable at a glance.
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

# Let the user download the table as a CSV - useful if the retention team wants
# to open the list in Excel and start working through the detractors.
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
