import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import random

# ---------------------------
# IMPORT DETERMINISTIC ENGINE
# ---------------------------
import engine as eng

st.set_page_config(page_title="ATC Clarity Console", layout="wide")

# ---------------------------
# SESSION STATE
# ---------------------------
if "history" not in st.session_state:
st.session_state.history = [] # conflict counts

if "interventions" not in st.session_state:
st.session_state.interventions = []

if "planes" not in st.session_state:
st.session_state.planes = []
st.session_state.last_refresh = None


# ---------------------------
# SIDEBAR CONTROLS
# ---------------------------
with st.sidebar:
st.subheader("Telemetry Controls")
refresh = st.button("Refresh Telemetry")
plane_count = st.slider("Number of aircraft", min_value=10, max_value=60, value=20, step=5)

# ---------------------------
# TELEMETRY GENERATION
# ---------------------------
if refresh or not st.session_state.planes:
seed = len(st.session_state.history)
st.session_state.planes = eng.generate_aircraft(plane_count, seed)
st.session_state.last_refresh = datetime.now()

planes = st.session_state.planes

# ---------------------------
# MODEL COMPUTATION PIPELINE
# ---------------------------
conflicts = eng.detect_conflicts(planes)
work = eng.compute_workload(planes, conflicts)
comms = eng.compute_communications()

# History update
st.session_state.history.append(len(conflicts))
conflict_counts = st.session_state.history

pred_conf = eng.predict_conflicts(conflict_counts)
clarity = eng.compute_clarity(conflicts, pred_conf, work["idx"], comms["fraction"])

# Bayesian layer
priors = eng.PRIORS
evidence = eng.compute_evidence(
clarity,
len(conflicts),
pred_conf,
work["idx"],
comms["fraction"],
)
posterior = eng.bayesian_confidence(priors, evidence)
best_state = max(posterior, key=posterior.get)


# ---------------------------
# LAYOUT TABS
# ---------------------------
tab_status, tab_traffic, tab_interventions, tab_model, tab_whatif = st.tabs(
["Status & Confidence", "Traffic Picture", "Interventions Log", "Model Internals", "What-If Lab"]
)

# ---------------------------
# TAB 1 — STATUS
# ---------------------------
with tab_status:
col1, col2, col3 = st.columns(3)

with col1:
st.metric("Clarity", f"{clarity:.1f} %")
st.metric("Active Conflicts", len(conflicts))
st.metric("Predicted Conflicts (5 min)", pred_conf)

with col2:
st.metric("Traffic Load (planes)", work["count"])
st.metric("Workload Index", f"{work['idx']:.2f}")
st.metric("Comms Fraction", f"{comms['fraction']:.2f}")

with col3:
st.subheader("Bayesian Confidence")
bayes_df = pd.DataFrame(
{
"State": list(posterior.keys()),
"Confidence (%)": [round(v * 100, 1) for v in posterior.values()],
}
).set_index("State")
st.bar_chart(bayes_df)
st.markdown(
f"**Most likely condition:** `{best_state}` "
f"({posterior[best_state] * 100:.1f} %)"
)

st.markdown("---")
st.subheader("Conflict Trend")
trend_df = pd.DataFrame(
{"timestep": range(len(conflict_counts)), "conflicts": conflict_counts}
).set_index("timestep")
st.line_chart(trend_df)


# ---------------------------
# TAB 2 — TRAFFIC
# ---------------------------
with tab_traffic:
st.subheader("Active Aircraft Telemetry")

planes_df = pd.DataFrame(planes)
conflict_ids = set()
for c in conflicts:
conflict_ids.add(c["plane_a"])
conflict_ids.add(c["plane_b"])
planes_df["in_conflict"] = planes_df["id"].isin(conflict_ids)
st.dataframe(planes_df)

st.markdown("#### Spatial View (Not for nav)")
st.map(planes_df.rename(columns={"lat": "latitude", "lon": "longitude"}))

st.markdown("---")
st.subheader("Current Conflicts")
if conflicts:
st.dataframe(pd.DataFrame(conflicts))
else:
st.info("No current conflicts detected.")


# ---------------------------
# TAB 3 — INTERVENTIONS
# ---------------------------
with tab_interventions:
st.subheader("Human-Gated Interventions")
st.write("System never acts autonomously. Operator approval required.")

action = st.radio(
"Select action:",
[
"Hold all departures",
"Issue spacing instructions",
"Request altitude separation",
"Do nothing (monitor only)",
],
index=3,
)

note = st.text_input("Optional note / rationale", value="")

confirm = st.button("Confirm Action")

if confirm:
entry = {
"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
"action": action,
"note": note,
"state": best_state,
"clarity": round(clarity, 1),
"active_conflicts": len(conflicts),
"pred_conflicts": pred_conf,
"workload_idx": round(work["idx"], 2),
}
st.session_state.interventions.append(entry)
st.success(f"Action logged: {action} at {entry['timestamp']}")

st.markdown("---")
st.subheader("Audit Trail")

if st.session_state.interventions:
log_df = pd.DataFrame(st.session_state.interventions)
st.dataframe(log_df)

csv = log_df.to_csv(index=False).encode("utf-8")
st.download_button(
label="Download CSV",
data=csv,
file_name="intervention_log.csv",
mime="text/csv",
)
else:
st.info("No interventions logged yet.")


# ---------------------------
# TAB 4 — MODEL INTERNALS
# ---------------------------
with tab_model:
st.subheader("Model Internals")

colA, colB = st.columns(2)
with colA:
st.markdown("**Priors**")
st.json(priors)

st.markdown("**Evidence Inputs**")
st.json(
dict(
clarity=clarity,
conflicts=len(conflicts),
predicted_conflicts=pred_conf,
workload_index=work["idx"],
comms_fraction=comms["fraction"],
)
)

with colB:
st.markdown("**Evidence Weights**")
st.json(evidence)

st.markdown("**Posterior**")
st.json(posterior)


# ---------------------------
# TAB 5 — WHAT-IF LAB
# ---------------------------
with tab_whatif:
st.subheader("What-If / Stress-Test Lab")
st.caption("Override inputs and watch Bayesian state flip in real time.")

col1, col2 = st.columns(2)

with col1:
w_conflicts = st.slider("Conflicts Now", 0, 10, len(conflicts))
w_pred = st.slider("Predicted Conflicts (5 min)", 0, 10, pred_conf)
w_workload = st.slider("Workload Index", 0.0, 1.0, float(work["idx"]), step=0.01)

with col2:
w_comms = st.slider("Comms Fraction", 0.0, 0.5, float(comms["fraction"]), step=0.01)
w_clarity = st.slider("Clarity (%)", 0.0, 100.0, float(clarity), step=0.5)

what_evidence = eng.compute_evidence(
w_clarity,
w_conflicts,
w_pred,
w_workload,
w_comms,
)
what_posterior = eng.bayesian_confidence(priors, what_evidence)
what_state = max(what_posterior, key=what_posterior.get)

st.markdown("---")
st.subheader("What-If Posterior")
st.json(what_posterior)

st.success(f"**What-If Most Likely Condition:** `{what_state}` "
f"({what_posterior[what_state] * 100:.1f} %)")
