"""
Speed Limit - neural network tests (script 03_train_model.py and speedlimit_core.py).

Usage:  python test_neural_network.py

The neural network (Software 2.0) predicts the number of near-accidents per hour. The code around it
chooses the speed limit. We cannot check "is the prediction correct" (the data is mostly noise, see the report),
but we can check everything that must be true no matter what the network learned.

Six tests:
  N1  inputs: the 8 inputs are in the right order, water and light are log-transformed, negative values are cut at 0
  N2  model structure: StandardScaler + MLPRegressor (16, 8) with early stopping, finite output for every limit
  N3  limit selection with a fake model (known predictions): highest safe limit, the boundary 1.0 is allowed,
      fallback 40 km/h if no limit is safe, negative predictions are cut at 0
  N4  guardrails on 300 random situations: the limit is always 40, 50, 60 or 70 (never 30, never 80 from the network),
      the "safe" flag matches the predictions, and a good-weather situation gives 80 km/h without the network
  N5  training is reproducible (two runs give the same model) and the network is not clearly worse than "always the mean"
  N6  input range check: water, light or temperature outside the training data raise an alert (only when the network decides)

The training script is started in a temporary folder, so nothing in the project folder is changed.
The second part checks the tests themselves: it makes deliberately broken copies of the code (mutants)
and shows that at least one test fails for each of them.
"""
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
FILES_TO_COPY = ["03_train_model.py", "speedlimit_core.py", "hourly_dataset_N.csv", "hourly_dataset_Nplus1.csv"]


# ---------------------------------------------------------------- helpers
def train_in(folder):
    """Runs 03_train_model.py in the folder; returns the trained model."""
    result = subprocess.run([sys.executable, "03_train_model.py"], cwd=folder, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-500:])
    return joblib.load(Path(folder) / "speedlimit_model.joblib")


def load_core(folder):
    spec = importlib.util.spec_from_file_location("core_under_test", Path(folder) / "speedlimit_core.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_bundle(replacements=()):
    """Copies the project files to a temporary folder, applies the 'replacements' (for mutants),
    trains the model twice and returns everything the tests need."""
    tmp = tempfile.mkdtemp()
    for name in FILES_TO_COPY:
        shutil.copy(HERE / name, Path(tmp) / name)
    for filename, old, new in replacements:
        path = Path(tmp) / filename
        text = path.read_text(encoding="utf-8")
        if old not in text:
            raise LookupError(f"the line was changed in {filename}")
        path.write_text(text.replace(old, new), encoding="utf-8")
    model = train_in(tmp)
    first = model.predict(pd.read_csv(Path(tmp) / "hourly_dataset_N.csv").pipe(load_core(tmp).make_features))
    model_again = train_in(tmp)
    return {"folder": tmp, "core": load_core(tmp), "model": model, "model_again": model_again, "first_predictions": first}


class FakeModel:
    """A model with known predictions (one value per limit), for testing the limit selection alone."""
    def __init__(self, by_limit):
        self.by_limit = by_limit

    def predict(self, features):
        return np.array([self.by_limit[int(lim)] for lim in features["limit"]])


# ---------------------------------------------------------------- the tests
def n1(b):
    core = b["core"]
    df = pd.DataFrame([{"limit": 60, "water": 1000, "light": 499, "temp": -3, "wind": 7, "humidity": 80, "noise": 90, "traffic": 12},
                       {"limit": 60, "water": -5, "light": -1, "temp": 0, "wind": 0, "humidity": 0, "noise": 0, "traffic": 0}])
    x = core.make_features(df)
    order_ok = list(x.columns) == ["limit", "log_water", "log_light", "temp", "wind", "humidity", "noise", "traffic"]
    log_ok = abs(x.loc[0, "log_water"] - np.log1p(1000)) < 1e-9 and abs(x.loc[0, "log_light"] - np.log1p(499)) < 1e-9
    clip_ok = x.loc[1, "log_water"] == 0 and x.loc[1, "log_light"] == 0
    return order_ok and log_ok and clip_ok, f"order ok: {order_ok}, log ok: {log_ok}, negatives cut at 0: {clip_ok}"


def n2(b):
    model = b["model"]
    names = [type(step).__name__ for _, step in model.steps]
    mlp = model.steps[-1][1]
    structure = names == ["StandardScaler", "MLPRegressor"] and mlp.hidden_layer_sizes == (16, 8) and mlp.early_stopping
    cond = dict(water=3000, light=0, temp=-4, wind=5, humidity=85, noise=95, traffic=15)
    rows = pd.DataFrame([{**cond, "limit": lim} for lim in (70, 60, 50, 40)])
    out = model.predict(b["core"].make_features(rows))
    finite = bool(np.all(np.isfinite(out))) and len(out) == 4
    return structure and finite, f"steps {names}, layers {mlp.hidden_layer_sizes}, early stopping {mlp.early_stopping}, output finite: {finite}"


def n3(b):
    core = b["core"]
    cond = dict(water=3000, light=0, temp=-4, wind=5, humidity=85, noise=95, traffic=15)
    cases = [
        ({70: 0.4, 60: 0.3, 50: 0.2, 40: 0.1}, 70, True),     # all safe -> the highest limit
        ({70: 1.5, 60: 1.2, 50: 0.9, 40: 0.5}, 50, True),     # 60 is too dangerous, 50 is the highest safe one
        ({70: 1.5, 60: 1.2, 50: 1.0, 40: 0.5}, 50, True),     # exactly 1.0 is still allowed (at most one)
        ({70: 2.0, 60: 1.8, 50: 1.5, 40: 1.1}, 40, False),    # nothing is safe -> fallback 40, flag False
        ({70: -0.3, 60: 0.0, 50: 0.0, 40: 0.0}, 70, True),    # a negative prediction is cut at 0
    ]
    problems = []
    for by_limit, want_limit, want_safe in cases:
        lim, preds, safe = core.choose_limit(FakeModel(by_limit), cond)
        if lim != want_limit or safe != want_safe:
            problems.append(f"{by_limit} -> {lim}, {safe}")
        if min(preds.values()) < 0:
            problems.append("negative prediction was not cut at 0")
    return not problems, "5 cases, all as expected" if not problems else "; ".join(problems)


def n4(b):
    core, model = b["core"], b["model"]
    rng = np.random.default_rng(0)
    problems = 0
    limits_seen = set()
    for _ in range(300):
        cond = dict(water=float(rng.uniform(0, 5000)), light=float(rng.uniform(0, 499)), temp=float(rng.uniform(-10, 25)),
                    wind=float(rng.uniform(0, 20)), humidity=float(rng.uniform(30, 100)), noise=float(rng.uniform(80, 110)),
                    traffic=float(rng.uniform(0, 40)))
        lim, preds, safe = core.choose_limit(model, cond)
        limits_seen.add(lim)
        if lim not in (40, 50, 60, 70) or safe != (preds[lim] <= 1.0) or min(preds.values()) < 0:
            problems += 1
    day = core.final_speed_limit(model, light=50000, water=0, temp=10)
    night_limit = core.final_speed_limit(model, light=499, water=0, temp=10)[0]
    day_ok = day[0] == 80 and day[1] == {} and night_limit in (40, 50, 60, 70)
    return problems == 0 and day_ok, f"300 random situations, problems: {problems}, limits seen: {sorted(limits_seen)}, good weather -> {day[0]} km/h, dark -> {night_limit} km/h"


def n5(b):
    core = b["core"]
    data = pd.read_csv(Path(b["folder"]) / "hourly_dataset_N.csv")
    x = core.make_features(data)
    same = np.allclose(b["first_predictions"], b["model_again"].predict(x), atol=1e-9)
    x_train, x_test, y_train, y_test = train_test_split(x, data["near_accidents"], test_size=0.25, random_state=42)
    mse = mean_squared_error(y_test, np.clip(b["model"].predict(x_test), 0, None))
    base = mean_squared_error(y_test, np.full(len(y_test), y_train.mean()))
    not_worse = mse <= 1.2 * base
    return same and not_worse, f"two training runs identical: {same}; test MSE network {mse:.3f} vs 'always the mean' {base:.3f} (limit: 1.2 x mean)"


def n6(b):
    core = b["core"]
    ranges = core.training_ranges(pd.read_csv(Path(b["folder"]) / "hourly_dataset_N.csv"))
    max_water, min_temp = ranges["water"][1], ranges["temp"][0]
    problems = []
    if core.range_alerts(ranges, 0, 1500, -3):
        problems.append("an ordinary situation raises an alert")
    too_wet = core.range_alerts(ranges, 50000, 10000, -1)
    if len(too_wet) != 1 or "water" not in too_wet[0]:
        problems.append("10000 um of water is not reported")
    too_cold = core.range_alerts(ranges, 0, 0, -50)
    if len(too_cold) != 1 or "temperature" not in too_cold[0]:
        problems.append("-50 C is not reported")
    if core.range_alerts(ranges, 10**9, 0, 10):
        problems.append("an alert although the network is not used")
    if core.range_alerts(ranges, 0, max_water, min_temp):
        problems.append("the boundary values of the training data raise an alert")
    return not problems, "5 cases, all as expected" if not problems else "; ".join(problems)


TESTS = [
    ("N1", "inputs: order, logarithm, negatives cut at 0", n1),
    ("N2", "model structure: scaler + MLP (16, 8)", n2),
    ("N3", "limit selection with a fake model", n3),
    ("N4", "guardrails on 300 random situations", n4),
    ("N5", "reproducible training, not worse than the mean", n5),
    ("N6", "input range check (alert outside training data)", n6),
]

# ---------------------------------------------------------------- part 1: run the tests
print("PART 1: neural network tests (training the model twice, please wait a few seconds)\n")
bundle = make_bundle()
print(f"{'ID':4s} {'what is checked':50s} result  details")
failed = 0
for tid, what, fn in TESTS:
    ok, details = fn(bundle)
    failed += not ok
    print(f"{tid:4s} {what:50s} {'PASS' if ok else 'FAIL':6s}  {details}")
shutil.rmtree(bundle["folder"], ignore_errors=True)

# ---------------------------------------------------------------- part 2: do the tests catch bugs?
CORE, TRAIN = "speedlimit_core.py", "03_train_model.py"
MUTANTS = {
    "limits tried from the lowest (picks 40 first)":
        [(CORE, "LIMITS_TO_TRY = [70, 60, 50, 40]", "LIMITS_TO_TRY = [40, 50, 60, 70]")],
    "boundary: 1.0 near-accidents is rejected":
        [(CORE, "if by_limit[lim] <= MAX_NEAR_ACCIDENTS_PER_HOUR:", "if by_limit[lim] < MAX_NEAR_ACCIDENTS_PER_HOUR:")],
    "fallback is the highest limit, not the lowest":
        [(CORE, "return min(LIMITS_TO_TRY), by_limit, False", "return max(LIMITS_TO_TRY), by_limit, False")],
    "negative predictions are not cut at 0":
        [(CORE, "np.clip(model.predict(make_features(rows)), 0, None)", "model.predict(make_features(rows))")],
    "no logarithm for water and light":
        [(CORE, 'np.log1p(df["water"].clip(lower=0))', 'df["water"]'), (CORE, 'np.log1p(df["light"].clip(lower=0))', 'df["light"]')],
    "network without normalization (no StandardScaler)":
        [(TRAIN, "make_pipeline(\n    StandardScaler(),\n    MLPRegressor(", "make_pipeline(\n    MLPRegressor(")],
    "wrong network size (4 neurons instead of 16, 8)":
        [(TRAIN, "hidden_layer_sizes=(16, 8)", "hidden_layer_sizes=(4,)")],
    "no fixed random seed (training not reproducible)":
        [(TRAIN, "max_iter=2000, random_state=42", "max_iter=2000")],
    "range check never raises an alert":
        [(CORE, "if not low <= values[name] <= high:", "if False:")],
    "range check: the boundary values raise an alert":
        [(CORE, "if not low <= values[name] <= high:", "if not low < values[name] < high:")],
    "range check also when the network is not used":
        [(CORE, "    if not router_uses_nn(light, water, temp):\n        return []\n    values = {", "    values = {")],
    "network never learns (1 iteration)":
        [(TRAIN, "max_iter=2000", "max_iter=1")],
}

print("\n\nPART 2: do the tests catch bugs? (each broken copy of the code should fail at least one test)\n")
print(f"{'broken version':50s} failing tests")
survivors = []
for name, replacements in MUTANTS.items():
    try:
        mutant = make_bundle(replacements)
    except LookupError as error:
        print(f"{name:50s} -- cannot build this mutant: {error} --")
        survivors.append(name)
        continue
    except Exception:
        print(f"{name:50s} crash (detected)")
        continue
    caught = []
    for tid, _what, fn in TESTS:
        try:
            if not fn(mutant)[0]:
                caught.append(tid)
        except Exception:
            caught.append(tid)
    shutil.rmtree(mutant["folder"], ignore_errors=True)
    if not caught:
        survivors.append(name)
    print(f"{name:50s} {', '.join(caught) if caught else '-- NOT DETECTED --'}")

print(f"\nBugs detected: {len(MUTANTS) - len(survivors)} of {len(MUTANTS)}")
print(f"Failed tests: {failed}")
sys.exit(1 if failed or survivors else 0)
