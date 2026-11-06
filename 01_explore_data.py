"""
Speed Limit - step 1: first look at the data.

Usage (in the VS Code terminal, in the folder with the CSV files):
    python 01_explore_data.py
or, if the CSV files are elsewhere:
    python 01_explore_data.py "C:\\path\\to\\csv"

The script changes nothing. It only reads the CSV files, prints an overview
and saves it to explore_report.txt.
"""
import sys
import glob
import os
import io
import pandas as pd

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
REPORT = io.StringIO()


def say(*args):
    text = " ".join(str(a) for a in args)
    print(text)
    REPORT.write(text + "\n")


def find(keyword):
    """Finds a CSV file by a part of its name (e.g. 'SensorReadingsYearN_')."""
    hits = [f for f in glob.glob(os.path.join(DATA_DIR, "*.csv")) if keyword in os.path.basename(f)]
    if not hits:
        raise SystemExit(f"No CSV file with '{keyword}' in its name found in folder {DATA_DIR}")
    return hits[0]


# The CSV files use ';' as separator, a decimal comma and a BOM at the start -> utf-8-sig
def read(keyword, **kw):
    return pd.read_csv(find(keyword), sep=";", decimal=",", encoding="utf-8-sig", **kw)


READ_COLS = ["Month", "Day", "Hour", "Minute", "Second", "Sensor", "Value"]

types = read("SensorTypes")
sensors = read("Sensors_")
readings_n = read("SensorReadingsYearN_")                      # has a header
readings_n1 = read("SensorReadingsYearNPlus1_", header=None, names=READ_COLS)  # has NO header!
acc_n = read("AccidentsYearN_")
acc_n1 = read("AccidentsYearNPlus1")

# spaces in the type codes ("W ", "WD   ") would cause problems when joining
types["SensorTypeCode"] = types["SensorTypeCode"].str.strip()
types["Unit"] = types["Unit"].str.strip()
sensors["SensorTypeCode"] = sensors["SensorTypeCode"].str.strip()

say("=" * 70)
say("1. BASIC DIMENSIONS")
say("=" * 70)
for name, df in [("SensorTypes", types), ("Sensors", sensors),
                 ("SensorReadings year N", readings_n), ("SensorReadings year N+1", readings_n1),
                 ("Accidents year N", acc_n), ("Accidents year N+1", acc_n1)]:
    say(f"{name:26s} rows: {len(df):6d}   columns: {list(df.columns)}")

say()
say("=" * 70)
say("2. SENSORS AND THEIR TYPES")
say("=" * 70)
sens = sensors.merge(types.drop_duplicates("SensorTypeCode"), on="SensorTypeCode", how="left")  # for code T we take "temperature" (first occurrence)
say(sens[["SensorID", "SensorTypeCode", "Type", "Unit", "Min", "Max"]].to_string(index=False))
dup_codes = types["SensorTypeCode"][types["SensorTypeCode"].duplicated(keep=False)].unique()
if len(dup_codes):
    say(f"\nWARNING: type code {list(dup_codes)} appears several times in SensorTypes (ambiguous: which type is a sensor with this code?)")
no_type = sensors[~sensors["SensorTypeCode"].isin(types["SensorTypeCode"])]
if len(no_type):
    say("WARNING: sensors without a type:", no_type["SensorID"].tolist())

say()
say("=" * 70)
say("3. READINGS (year N)")
say("=" * 70)
r = readings_n
say("Time: months", sorted(int(x) for x in r["Month"].unique()), "| days", r["Day"].min(), "-", r["Day"].max(),
    "| hours", r["Hour"].min(), "-", r["Hour"].max(),
    "| minutes", r["Minute"].min(), "-", r["Minute"].max(),
    "| seconds", r["Second"].min(), "-", r["Second"].max())
say("Missing values:\n" + r.isna().sum().to_string())
say("Duplicate rows:", int(r.duplicated().sum()))
say("Duplicates (time + sensor):", int(r.duplicated(subset=["Month", "Day", "Hour", "Minute", "Second", "Sensor"]).sum()))

per_sensor = r.groupby("Sensor")["Value"].agg(["count", "min", "max", "mean"]).round(2)
per_sensor = per_sensor.join(sensors.set_index("SensorID")["SensorTypeCode"], how="outer")
per_sensor = per_sensor.join(types.drop_duplicates("SensorTypeCode").set_index("SensorTypeCode")[["Min", "Max"]].rename(
    columns={"Min": "TypeMin", "Max": "TypeMax"}), on="SensorTypeCode")
say("\nOverview per sensor (count/min/max/mean from the data vs. the allowed range of the type; sensor -1 = speed limit):")
say(per_sensor.to_string())

say("\nSensors in SENSORS.csv that have no readings:",
    sorted(int(x) for x in set(sensors["SensorID"]) - set(r["Sensor"].unique())))
say("Sensors in the readings that are not in SENSORS.csv:",
    sorted(int(x) for x in set(r["Sensor"].unique()) - set(sensors["SensorID"]) - {-1}))

# values outside the allowed range of the type
bad = r.merge(sensors, left_on="Sensor", right_on="SensorID").merge(
    types.drop_duplicates("SensorTypeCode"), on="SensorTypeCode")
out_of_range = bad[(bad["Value"] < bad["Min"]) | (bad["Value"] > bad["Max"])]
say(f"\nReadings outside the allowed range of their type: {len(out_of_range)}")
if len(out_of_range):
    say(out_of_range.groupby("Sensor")["Value"].agg(["count", "min", "max"]).to_string())

say()
say("=" * 70)
say("4. SPEED LIMIT (sensor -1)")
say("=" * 70)
lim = r[r["Sensor"] == -1]
say("Number of readings:", len(lim))
say("Distribution of limit values:\n" + lim["Value"].value_counts().sort_index().to_string())

say()
say("=" * 70)
say("5. ACCIDENTS")
say("=" * 70)
for name, a in [("year N", acc_n), ("year N+1", acc_n1)]:
    say(f"--- {name}: {len(a)} accidents")
    say("Months:", sorted(int(x) for x in a["Month"].unique()), "| days", a["Day"].min(), "-", a["Day"].max(),
        "| hours", a["Hour"].min(), "-", a["Hour"].max(),
        "| 'Second' (note: it may actually be the minute!)", a["Second"].min(), "-", a["Second"].max())
    say("Missing values per column:\n" + a.isna().sum().to_string())
    say("Duplicate rows:", int(a.duplicated().sum()))
    near = a[["CloseCarCm", "CloseGuardrailCm", "SkidAngle"]].notna().any(axis=1)
    real = a[["Damage", "Injured"]].notna().any(axis=1)
    say(f"Near-accidents (CloseCar/CloseGuardrail/SkidAngle filled in): {int(near.sum())}, "
        f"real accidents (Damage/Injured): {int(real.sum())}, neither: {int((~near & ~real).sum())}")
    say(a.describe().round(1).to_string())

say()
say("=" * 70)
say("6. YEAR N vs. YEAR N+1")
say("=" * 70)
same = readings_n.reset_index(drop=True).equals(readings_n1.reset_index(drop=True))
say("Readings of year N and N+1 are exactly the same:", same)
say("(if True: the new year only has different accidents, so the retraining analysis must rely on accidents, not on sensors)")

path = os.path.join(DATA_DIR, "explore_report.txt")
with open(path, "w", encoding="utf-8") as f:
    f.write(REPORT.getvalue())
print(f"\nReport saved to: {path}")
