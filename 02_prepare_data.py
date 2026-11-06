"""
Speed Limit - step 2: data cleaning and the hourly table.

Usage:  python 02_prepare_data.py            (CSV files in the same folder)
        python 02_prepare_data.py "C:\\path\\to\\csv"

Output: hourly_dataset_N.csv, hourly_dataset_Nplus1.csv, cleaning_log.txt
One row = one hour. Columns: day, hour, limit, water, light, temp, wind, humidity, noise, traffic, near_accidents
"""
import sys, os, glob, io
import numpy as np
import pandas as pd

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
LOG = io.StringIO()

# ---- DECISIONS (each one goes into the presentation as an "assumption" / "error fix") ----
# Redundant sensors: they measure the same thing, so we average them
# (the mean ignores missing values -> it is also a form of imputation).
SENSOR_GROUPS = {
    "water":    [1, 8, 11],
    "light":    [2, 9, 12],
    "temp":     [15, 18, 19],   # sensor 19 has no description in Sensors.csv, but its values look like temperature (mean -3.5; correlation 0.998 with 15 and 18)
    "wind":     [6, 14],        # wind strength (wind direction 5, 13 is not used: it is non-zero even when the wind speed is zero)
    "humidity": [4],
    "noise":    [3],            # sensor 10 (also noise) has no readings at all
    "traffic":  [16],           # ASSUMPTION: code 'T' appears twice in SensorTypes; sensor 16 has a range of 2 to 30 and does not correlate with temperature (0.21) -> we treat it as traffic
}
LIMIT_SENSOR = -1
ACCIDENT_HOUR_OFFSET = 1  # ASSUMPTION: accident hours are 1..24, reading hours are 0..23 -> accident hour - 1


def say(*a):
    t = " ".join(str(x) for x in a)
    print(t)
    LOG.write(t + "\n")


def find(keyword):
    hits = [f for f in glob.glob(os.path.join(DATA_DIR, "*.csv")) if keyword in os.path.basename(f)]
    if not hits:
        raise SystemExit(f"No CSV file with '{keyword}' in its name found in folder {DATA_DIR}")
    return hits[0]


def read(keyword, **kw):
    return pd.read_csv(find(keyword), sep=";", decimal=",", encoding="utf-8-sig", **kw)


COLS = ["Month", "Day", "Hour", "Minute", "Second", "Sensor", "Value"]


def build_year(label, readings, accidents):
    say("\n" + "=" * 70)
    say(f"YEAR {label}")
    say("=" * 70)

    # --- CLEANING THE READINGS ---
    n0 = len(readings)
    readings = readings.dropna(subset=["Day", "Hour", "Sensor", "Value"]).drop_duplicates()
    say(f"Readings: {n0} rows, {len(readings)} after removing empty and duplicate rows")
    readings = readings.copy()
    readings["Sensor"] = readings["Sensor"].astype(int)

    # one value per (day, hour, sensor): if there were several readings in an hour, we take their mean
    pivot = readings.pivot_table(index=["Day", "Hour"], columns="Sensor", values="Value", aggfunc="mean")

    # average the redundant sensors of each group into one value per hour
    hourly = pd.DataFrame(index=pivot.index)
    for name, ids in SENSOR_GROUPS.items():
        present = [i for i in ids if i in pivot.columns]
        hourly[name] = pivot[present].mean(axis=1, skipna=True)
    hourly["limit"] = pivot[LIMIT_SENSOR]

    # IMPUTATION: if a quantity is missing in some hour, fill it by linear interpolation in time (neighbouring hours)
    missing_before = hourly.isna().sum()
    hourly = hourly.sort_index().interpolate(limit_direction="both")
    say("Missing values in the hourly table before imputation:", missing_before.to_dict())
    say("-> after imputation (linear interpolation in time):", int(hourly.isna().sum().sum()))

    # the limit must be one of 40, 50, ..., 80 -> round to the nearest
    hourly["limit"] = (hourly["limit"] / 10).round() * 10

    # --- ACCIDENTS -> number of near-accidents per hour ---
    acc = accidents.copy()
    empty = acc[["Damage", "Injured", "CloseCarCm", "CloseGuardrailCm", "SkidAngle"]].isna().all(axis=1)
    say(f"Accidents: {len(acc)} rows, of which completely empty (no Damage/Injured/Close*/Skid): {int(empty.sum())} -> removed")
    acc = acc[~empty]
    # SPEC4: near-accident = skid or too small a distance to a car/guardrail
    acc["near"] = acc[["CloseCarCm", "CloseGuardrailCm", "SkidAngle"]].notna().any(axis=1)
    say(f"Near-accidents according to SPEC4: {int(acc['near'].sum())}, real accidents without a near-accident (only Damage/Injured): {int((~acc['near']).sum())}")
    acc["hour0"] = acc["Hour"] - ACCIDENT_HOUR_OFFSET
    counts = acc[acc["near"]].groupby(["Day", "hour0"]).size()
    counts.index.names = ["Day", "Hour"]
    hourly["near_accidents"] = counts.reindex(hourly.index, fill_value=0)
    lost = int(acc["near"].sum()) - int(hourly["near_accidents"].sum())
    say(f"Near-accidents that could not be assigned to a reading hour: {lost}")

    # light check: lux vs. millilux
    say(f"Dark hours (light < 500): {int((hourly['light'] < 500).sum())}, (light < 0.5): {int((hourly['light'] < 0.5).sum())}  "
        "-> if the numbers are equal, it does not matter whether the unit is lux or millilux")
    ice = (hourly["water"] > 1000) & (hourly["temp"] < 0)
    say(f"Hours with ice risk: {int(ice.sum())}")
    say(f"Mean near-accidents per hour: {hourly['near_accidents'].mean():.3f}, variance: {hourly['near_accidents'].var():.3f}")

    out = hourly.reset_index().rename(columns={"Day": "day", "Hour": "hour"})
    return out


# ---- LOADING ----
readings_n = read("SensorReadingsYearN_")
readings_n1 = read("SensorReadingsYearNPlus1_", header=None, names=COLS)   # this file has no header
acc_n = read("AccidentsYearN_")
acc_n1 = read("AccidentsYearNPlus1")

say("Data source:", DATA_DIR)
say("Sensor groups used:", SENSOR_GROUPS)

ds_n = build_year("N", readings_n, acc_n)
ds_n1 = build_year("N+1", readings_n1, acc_n1)

ds_n.to_csv(os.path.join(DATA_DIR, "hourly_dataset_N.csv"), index=False)
ds_n1.to_csv(os.path.join(DATA_DIR, "hourly_dataset_Nplus1.csv"), index=False)
say("\nSaved: hourly_dataset_N.csv (%d rows), hourly_dataset_Nplus1.csv (%d rows)" % (len(ds_n), len(ds_n1)))
say("\nFirst 5 rows of year N:\n" + ds_n.head().round(2).to_string(index=False))

with open(os.path.join(DATA_DIR, "cleaning_log.txt"), "w", encoding="utf-8") as f:
    f.write(LOG.getvalue())
print("Log saved to cleaning_log.txt")
