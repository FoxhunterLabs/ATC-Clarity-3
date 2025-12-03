import numpy as np
import random
from math import cos, radians, sqrt

# ---------------------------------------------
# CONSTANTS & PRIORS
# ---------------------------------------------
ALT_SEPARATION_FT = 800
LATERAL_SEPARATION_NM = 3
MAX_PLANES = 40
HISTORY_WINDOW = 8

LAT_CENTER = 39.0
LON_CENTER = -86.0

PRIORS = {
"STABLE": 0.45,
"ELEVATED": 0.30,
"HIGH_LOAD": 0.15,
"CRITICAL": 0.10,
}

# ---------------------------------------------
# TELEMETRY GENERATION
# ---------------------------------------------
def generate_aircraft(n: int = 20, seed: int | None = None):
rnd = random.Random(seed) if seed is not None else random

planes = []
for i in range(n):
planes.append(
{
"id": f"AC{i+1:02d}",
"alt": rnd.randint(2000, 38000),
"speed": rnd.randint(250, 520),
"lat": LAT_CENTER + rnd.uniform(-0.3, 0.3),
"lon": LON_CENTER + rnd.uniform(-0.3, 0.3),
"heading": rnd.randint(0, 359),
"destination": rnd.choice(["IND", "ORD", "SDF", "CVG"]),
}
)
return planes

# ---------------------------------------------
# DISTANCE
# ---------------------------------------------
def lateral_distance_nm(lat1, lon1, lat2, lon2):
dlat = (lat2 - lat1) * 60.0
dlon = (lon2 - lon1) * 60.0 * cos(radians(LAT_CENTER))
return sqrt(dlat**2 + dlon**2)

# ---------------------------------------------
# CONFLICT DETECTION
# ---------------------------------------------
def detect_conflicts(planes):
conflicts = []
planes_sorted = sorted(planes, key=lambda p: p["id"])

for i in range(len(planes_sorted)):
for j in range(i + 1, len(planes_sorted)):
p1, p2 = planes_sorted[i], planes_sorted[j]

dalt = abs(p1["alt"] - p2["alt"])
if dalt >= ALT_SEPARATION_FT:
continue

dist_nm = lateral_distance_nm(p1["lat"], p1["lon"], p2["lat"], p2["lon"])
if dist_nm >= LATERAL_SEPARATION_NM:
continue

severity = "PROXIMITY"
if dalt < ALT_SEPARATION_FT / 2 and dist_nm < LATERAL_SEPARATION_NM / 2:
severity = "LOSS_OF_SEPARATION"

conflicts.append(
{
"plane_a": p1["id"],
"plane_b": p2["id"],
"alt_sep_ft": int(dalt),
"lat_dist_nm": round(dist_nm, 2),
"severity": severity,
}
)
return conflicts

# ---------------------------------------------
# WORKLOAD / COMMS / CLARITY
# ---------------------------------------------
def compute_workload(planes, conflicts):
count = len(planes)
conflict_factor = len(conflicts) * 0.2
traffic_factor = count / max(1, MAX_PLANES)
idx = min(1.0, traffic_factor + conflict_factor)
return {"count": count, "idx": idx}

def compute_communications():
return {"fraction": random.uniform(0.05, 0.25)}

def compute_clarity(conflicts, pred_conf, workload_idx, comms_frac):
base = 100.0
base -= len(conflicts) * 15.0
base -= pred_conf * 8.0
base -= workload_idx * 20.0
base -= comms_frac * 25.0
return max(0.0, min(100.0, base))

# ---------------------------------------------
# CONFLICT PREDICTION
# ---------------------------------------------
def predict_conflicts(history_counts):
if len(history_counts) < 4:
return 0

recent = history_counts[-HISTORY_WINDOW:]
x = np.arange(len(recent))
y = np.array(recent)

A = np.vstack([x, np.ones(len(x))]).T
m, b = np.linalg.lstsq(A, y, rcond=None)[0]

future_x = len(recent) + 1
pred = m * future_x + b
return max(0, int(round(pred)))

# ---------------------------------------------
# BAYESIAN LAYER
# ---------------------------------------------
def bayesian_confidence(prior, evidence):
posterior = {}
for state in prior:
posterior[state] = prior[state] * evidence.get(state, 1e-9)

total = sum(posterior.values()) + 1e-12
return {k: v / total for k, v in posterior.items()}

def compute_evidence(
clarity,
conflicts_now,
pred_conflicts,
workload_idx,
comms_frac,
):
c = clarity / 100.0
conflict_term = conflicts_now + pred_conflicts
load_term = workload_idx
comms_term = comms_frac

evidence = {
"STABLE": float(np.exp(+2.0 * c - 0.5 * conflict_term - 0.5 * load_term)),
"ELEVATED": float(np.exp(+1.0 * load_term + 0.3 * conflict_term + 0.5 * comms_term)),
"HIGH_LOAD": float(np.exp(+1.5 * load_term + 0.7 * comms_term + 0.5 * pred_conflicts)),
"CRITICAL": float(np.exp(+2.0 * (1.0 - c) + 0.8 * conflict_term + 0.5 * load_term)),
}
return evidence
