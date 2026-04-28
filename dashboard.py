"""
dashboard.py — Posture Analytics Dashboard (Streamlit)
Run: streamlit run dashboard.py
Opens at http://localhost:8501
"""

import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import os
from config import DB_PATH

st.set_page_config(
    page_title="Posture Dashboard",
    page_icon="🧍",
    layout="wide",
)

st.title("AI Posture Detection — Dashboard")
st.caption("Review your posture history and track improvement over time.")

# ── Load data ──────────────────────────────────────────────────────────────

@st.cache_resource
def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH, check_same_thread=False)

con = get_connection()

try:
    df = pd.read_sql("SELECT * FROM sessions ORDER BY id DESC", con)
except Exception:
    df = pd.DataFrame()

# ── Empty state ────────────────────────────────────────────────────────────

if df.empty:
    st.info("No sessions recorded yet. Run  `python main.py`  to start monitoring.")
    st.stop()

# ── Metric cards ───────────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total sessions",      len(df))
col2.metric("Avg posture score",   f"{df['good_pct'].mean():.0f}%")
col3.metric("Best session",        f"{df['good_pct'].max():.0f}%")
total_min = df["duration_sec"].sum() // 60
col4.metric("Total monitored",     f"{total_min} min")

st.divider()

# ── Charts ─────────────────────────────────────────────────────────────────

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Posture score over time")
    df_chart = df.sort_values("id")
    fig = px.line(
        df_chart, x="date", y="good_pct",
        markers=True,
        labels={"good_pct": "Good posture %", "date": "Date"},
    )
    fig.add_hline(y=70, line_dash="dash", line_color="green",
                  annotation_text="Target 70%")
    fig.update_layout(
        yaxis_range=[0, 100],
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Most common issues")
    issue_counts = df["main_issue"].value_counts().reset_index()
    issue_counts.columns = ["Issue", "Count"]
    if not issue_counts.empty:
        fig2 = px.bar(
            issue_counts, x="Count", y="Issue",
            orientation="h",
            labels={"Issue": "", "Count": "Sessions"},
        )
        fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No issues logged yet.")

st.divider()

# ── Posture quality distribution ───────────────────────────────────────────

st.subheader("Session quality distribution")

def score_label(pct):
    if pct >= 90: return "Excellent"
    if pct >= 70: return "Good"
    if pct >= 50: return "Poor"
    return "Critical"

df["quality"] = df["good_pct"].apply(score_label)
q_counts = df["quality"].value_counts().reset_index()
q_counts.columns = ["Quality", "Count"]
color_map = {
    "Excellent": "#3BB273",
    "Good":      "#F0A500",
    "Poor":      "#E05C2A",
    "Critical":  "#D63031",
}
fig3 = px.pie(q_counts, names="Quality", values="Count",
              color="Quality", color_discrete_map=color_map,
              hole=0.45)
fig3.update_layout(margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── Session history table ──────────────────────────────────────────────────

st.subheader("Session history")
display_df = df[["date", "start_time", "duration_sec", "good_pct", "main_issue"]].copy()
display_df.columns = ["Date", "Time", "Duration (s)", "Good posture %", "Main issue"]
display_df["Good posture %"] = display_df["Good posture %"].apply(lambda x: f"{x:.0f}%")
display_df["Duration (s)"]   = display_df["Duration (s)"].apply(lambda x: f"{x}s")
st.dataframe(display_df, use_container_width=True, hide_index=True)

# ── Refresh button ─────────────────────────────────────────────────────────
if st.button("Refresh data"):
    st.cache_resource.clear()
    st.rerun()
