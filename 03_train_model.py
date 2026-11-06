"""
Speed Limit - step 3: features, neural network training, evaluation, limit selection.

Usage:   python 03_train_model.py            (hourly_dataset_N.csv and hourly_dataset_Nplus1.csv in the same folder)
Output:  speedlimit_model.joblib, training_report.txt

The neural network predicts the NUMBER OF NEAR-ACCIDENTS PER HOUR from
(limit, water, light, temperature, wind, humidity, noise, traffic).
The limit is then chosen as the highest of 80, 70, 60, 50, 40 for which the prediction does not exceed 1 (REQ1).
"""
import os, io, sys, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from speedlimit_core import make_features, router_uses_nn, choose_limit, final_speed_limit

warnings.filterwarnings("ignore")   # we do not want sklearn convergence warnings in the log
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else HERE
LOG = io.StringIO()


def say(*a):
    t = " ".join(str(x) for x in a)
    print(t)
    LOG.write(t + "\n")


train_all = pd.read_csv(os.path.join(DATA_DIR, "hourly_dataset_N.csv"))
data_n1 = pd.read_csv(os.path.join(DATA_DIR, "hourly_dataset_Nplus1.csv"))

X = make_features(train_all)
y = train_all["near_accidents"]
say(f"Dataset of year N: {len(train_all)} hours, inputs: {list(X.columns)}")

# ---- train / test split (25 % for testing, as in the lecture) ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ---- model: normalization (StandardScaler) + neural network ----
model = make_pipeline(
    StandardScaler(),
    MLPRegressor(hidden_layer_sizes=(16, 8), alpha=0.01, early_stopping=True,
                 validation_fraction=0.2, max_iter=2000, random_state=42),
)
model.fit(X_train, y_train)


def evaluate(name, m, Xe, ye, baseline_value):
    p = np.clip(m.predict(Xe), 0, None)
    mse = mean_squared_error(ye, p)
    base_mse = mean_squared_error(ye, np.full(len(ye), baseline_value))
    say(f"{name:34s} MSE network: {mse:.3f} | MSE 'always the mean': {base_mse:.3f} | MAE: {mean_absolute_error(ye, p):.3f}")
    return mse, base_mse


say("\n--- EVALUATION ---")
baseline = y_train.mean()   # simple competitor: always predict the mean of the training data
mse_test, base_test = evaluate("Test (25 % of year N)", model, X_test, y_test, baseline)
evaluate("Training (75 % of year N)", model, X_train, y_train, baseline)
mse_n1, base_n1 = evaluate("Year N+1 (new, unseen data)", model, make_features(data_n1), data_n1["near_accidents"], baseline)

# subset where the network is actually used in production (dark or ice risk)
routed = train_all.apply(lambda r: router_uses_nn(r["light"], r["water"], r["temp"]), axis=1)
say(f"\nHours where the router passes the decision to the network (dark or ice risk): {int(routed.sum())} of {len(train_all)}")
evaluate("Only 'bad' hours (year N)", model, X[routed], y[routed], baseline)

say("\n--- WHAT THIS MEANS ---")
if mse_test >= 0.97 * base_test:
    say("The network is NOT clearly better than the simple mean. The number of near-accidents per hour cannot be\n"
        "reliably predicted from the sensors in this data (the variance of the counts is almost equal to their mean = looks like random noise).\n"
        "This belongs in the presentation as a finding (lessons learned) and as input for the analysis: more / better data is needed, retraining.")
else:
    say("The network is better than the simple mean.")

joblib.dump(model, os.path.join(DATA_DIR, "speedlimit_model.joblib"))
say("\nModel saved: speedlimit_model.joblib")

# ---- demo of the complete decision (router + network) ----
say("\n--- DEMO: speed limit decision (light, water, temperature -> limit) ---")
scenarios = [
    ("day, dry, +10 C",              50000,    0, 10),
    ("night, dry, +5 C",                 0,    0,  5),
    ("day, wet, +5 C",               50000, 3000,  5),
    ("day, wet, frost (ice risk)",   50000, 3000, -4),
    ("night, wet, frost",                0, 3000, -4),
    ("darkness boundary: light 499",   499,    0, 10),
    ("darkness boundary: light 500",   500,    0, 10),
]
for text, light, water, temp in scenarios:
    lim, preds, safe = final_speed_limit(model, light=light, water=water, temp=temp)
    extra = "" if not preds else f"   predictions per limit: {preds}" + ("" if safe else "   (NO limit satisfies REQ1 -> fallback 40)")
    say(f"{text:34s} -> {lim} km/h{extra}")

with open(os.path.join(DATA_DIR, "training_report.txt"), "w", encoding="utf-8") as f:
    f.write(LOG.getvalue())
print("\nReport saved to training_report.txt")
