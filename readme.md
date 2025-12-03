ATC Clarity Console


A deterministic, human-gated airspace decision-support prototype.

Physics-first telemetry → conflict detection → Bayesian state estimation → operator-approved interventions.



Designed as a testbed for transparent, auditable autonomy in safety-critical domains.

Overview


This system simulates an airspace sector, generates telemetry, detects conflicts, evaluates systemic clarity, and computes a Bayesian confidence state (STABLE → CRITICAL).

The model is fully deterministic and inspectable.

The UI (Streamlit) is a thin viewer/controller around a pure Python engine.

Features


Deterministic Engine (engine.py)
Reproducible telemetry (seeded aircraft generation)

Deterministic conflict detection with severity scoring

Trend-based conflict prediction (least-squares slope)

Clarity / workload / comms modeling

Bayesian confidence model with complete evidence trace



Human-in-the-Loop Controls
System never executes actions autonomously

Operator-approved interventions only

Full audit log with timestamp, system state, and operator notes

One-click CSV export



Streamlit Interface (app.py)
Status & confidence dashboard

Traffic telemetry + spatial map

Conflict tables with severity

Intervention log

Model internals (priors, evidence, posterior)

What-If Lab — override inputs and watch posterior state flip in real time

Architecture
atc-clarity-console/
│
├── app.py          # Streamlit UI (thin shell)
├── engine.py       # Fully deterministic model engine
└── README.md
The engine contains all logic.

The UI does almost nothing except call pure functions and present results.

Running the App


Install dependencies
pip install -r requirements.txt
Run Streamlit app
streamlit run app.py


What-If Lab


Use the dedicated tab to override:

clarity

workload index

current conflicts

predicted conflicts

comms load



This allows you to stress-test the Bayesian model and validate state transitions under edge conditions.

Why This Exists


Autonomy in safety-critical systems demands:

determinism

inspectability

human gating

reversibility

explicit confidence modeling



This project demonstrates how to structure such a system while remaining simple, auditable, and engineer-driven.

License


MIT
