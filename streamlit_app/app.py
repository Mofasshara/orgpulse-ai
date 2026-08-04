"""OrgPulse AI — PharmaOps dashboard (presentation plane).

Reads the committed run artifacts written by the Hermes compute plane
(``runs/latest``) and renders them. Imports NO ``orgpulse`` package code — only
``streamlit``, ``pandas``, ``plotly`` and the self-contained ``data`` module — so
Streamlit Cloud stays free of the Prophet/LangGraph stack (split-plane design).

Run locally:  ``streamlit run streamlit_app/app.py``
Override run source: env ``ORGPULSE_RUNS`` (defaults to ``./runs``).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from streamlit_app import data

# --- severity palette (works in light and dark) ---
SEV = {
    "SAFE": ("#16a34a", "🟢", "Safe"),
    "WARNING": ("#d97706", "🟠", "Warning"),
    "CRITICAL": ("#dc2626", "🔴", "Critical"),
    "DATA_QUALITY": ("#64748b", "⚪", "Data quality"),
}
HERO = "daily_enrollment"

RUNS_ROOT = Path(os.environ.get("ORGPULSE_RUNS", "runs"))

st.set_page_config(page_title="OrgPulse AI — PharmaOps", page_icon="🧬", layout="wide")


def _sev(name: str):
    return SEV.get(name, SEV["SAFE"])


def _severity_chip(name: str) -> str:
    color, icon, label = _sev(name)
    return (
        f"<span style='background:{color};color:white;padding:2px 10px;"
        f"border-radius:12px;font-size:0.85em;font-weight:600'>{icon}&nbsp;{label}</span>"
    )


# ── Load ────────────────────────────────────────────────────────────────────
latest = RUNS_ROOT / "latest"
if not (latest / "findings.json").exists():
    st.title("🧬 OrgPulse AI — PharmaOps")
    st.warning(
        f"No run artifacts found under `{latest}`.\n\n"
        "Generate one with:\n\n```bash\nuv run orgpulse tick --tick-number 0\n```"
    )
    st.stop()

findings = data.load_findings(latest)
history = data.load_history(RUNS_ROOT)
audit_ok = data.verify_audit(latest / "audit.jsonl")
audit_entries = data.load_audit(latest / "audit.jsonl")

# ── Header ──────────────────────────────────────────────────────────────────
st.title("🧬 OrgPulse AI — PharmaOps")
st.caption(
    "Autonomous organizational-intelligence agent for regulated pharma operations. "
    "Decision support only — every recommendation is gated on human review (EU AI Act Art. 14)."
)

overall = findings.get("overall_severity", "SAFE")
color, icon, label = _sev(overall)
top = st.columns([2, 1, 1, 1])
top[0].markdown(f"### {icon} Overall status: {label}")
top[1].metric("Run", findings.get("run_id", "—").split("-")[1] if "-" in findings.get("run_id", "") else "—")
top[2].metric("Generated", findings.get("generated_at", "—")[:10])
top[3].metric("Audit chain", "verified ✅" if audit_ok else "TAMPER ❌")

if findings.get("alert"):
    st.error(f"🚨 Alert raised — overall severity **{label}**. Human review required before any action.", icon="🚨")
elif overall == "SAFE":
    st.success("All monitored KPIs within expected control limits.", icon="✅")

st.divider()

# ── KPI cards ───────────────────────────────────────────────────────────────
st.subheader("Monitored KPIs")
kfs = findings.get("kpi_findings", [])
cols = st.columns(len(kfs) or 1)
for col, kf in zip(cols, kfs):
    c, ic, lab = _sev(kf["severity"])
    val = kf.get("latest_value")
    val_str = f"{val:.3g}" if isinstance(val, (int, float)) else "—"
    with col:
        st.markdown(
            f"<div style='border-left:4px solid {c};padding:6px 12px'>"
            f"<div style='font-weight:600'>{kf['kpi']}</div>"
            f"<div style='font-size:1.6em'>{val_str}</div>"
            f"{_severity_chip(kf['severity'])}</div>",
            unsafe_allow_html=True,
        )
        st.caption(kf.get("detail", ""))

st.divider()

# ── Forecast fan (hero KPI) ─────────────────────────────────────────────────
left, right = st.columns([3, 2])
with left:
    st.subheader(f"Forecast — {HERO}")
    fc = findings.get("forecast", {}).get(HERO)
    if fc:
        anchor = next((k.get("latest_value") for k in kfs if k["kpi"] == HERO), None)
        xs, yhat, lower, upper = [0], [anchor], [anchor], [anchor]
        for h in ("h30", "h60", "h90"):
            xs.append(int(h[1:]))
            yhat.append(fc[h]["yhat"])
            lower.append(fc[h]["lower"])
            upper.append(fc[h]["upper"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs + xs[::-1], y=upper + lower[::-1], fill="toself",
                                 fillcolor="rgba(37,99,235,0.15)", line=dict(color="rgba(0,0,0,0)"),
                                 name="90% interval", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=yhat, mode="lines+markers", line=dict(color="#2563eb", width=3),
                                 name="forecast"))
        fig.update_layout(
            height=340, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="days ahead", yaxis_title=HERO,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.0),
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Prophet forecast with a calibrated 90% interval. Backtest is honest: this model does "
            "**not** beat a naïve baseline (MASE ≈ 1.4) — the value is the interval + the audit trail, "
            "not point accuracy."
        )
    else:
        st.info("No forecast in this run.")

with right:
    st.subheader("Analyst interpretation")
    st.write(findings.get("analysis", "—"))

st.divider()

# ── Recommendations (human-gated) ───────────────────────────────────────────
st.subheader("Recommendations — pending human review")
recs = findings.get("recommendations", [])
if not recs:
    st.info("No recommendations generated for this run.")
for r in recs:
    with st.container(border=True):
        head = st.columns([3, 1])
        head[0].markdown(f"**{r.get('capa_frame','—')} · {r['kpi']}**")
        head[1].markdown(
            f"<span style='background:#7c3aed;color:white;padding:2px 8px;border-radius:10px;"
            f"font-size:0.8em'>⏳ {r.get('review_status','PENDING_HUMAN_REVIEW')}</span>",
            unsafe_allow_html=True,
        )
        st.write(r["text"])
        st.caption("⚖️ " + r.get("disclaimer", ""))

st.divider()

# ── Governance: audit trail + EU AI Act ─────────────────────────────────────
gcol1, gcol2 = st.columns(2)
with gcol1:
    st.subheader("Audit trail (EU AI Act Art. 12)")
    st.markdown(
        ("✅ **Hash chain verified** — every logged decision is intact and reconstructable."
         if audit_ok else
         "❌ **Tamper detected** — the audit chain does not verify.")
    )
    if audit_entries:
        df = pd.DataFrame(
            [{"seq": e["seq"], "kind": e["kind"], "trace_id": e["trace_id"],
              "timestamp": e["timestamp"], "entry_hash": e["entry_hash"][:12] + "…"}
             for e in audit_entries]
        )
        st.dataframe(df, hide_index=True, width="stretch", height=220)

with gcol2:
    st.subheader("EU AI Act posture")
    st.markdown(
        "- **Risk tier:** High-risk (Annex III) — self-classified\n"
        "- **Art. 9** Risk management · **Art. 11** Technical documentation\n"
        "- **Art. 12** Logging → the hash-chained audit trail (left)\n"
        "- **Art. 13** Transparency → this dashboard + disclaimers\n"
        "- **Art. 14** Human oversight → every recommendation is `PENDING_HUMAN_REVIEW`\n\n"
        "See `docs/EU_AI_ACT_SELF_CLASSIFICATION.md` for the full mapping."
    )

# ── Run history ─────────────────────────────────────────────────────────────
if history:
    st.divider()
    st.subheader("Run history")
    hist_df = pd.DataFrame(history)[["generated_at", "overall_severity", "alert", "run_id"]]
    st.dataframe(hist_df, hide_index=True, width="stretch")

st.caption("OrgPulse AI — PharmaOps · synthetic data · decision-support prototype · not a medical device.")
