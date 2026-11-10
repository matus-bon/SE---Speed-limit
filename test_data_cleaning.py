"""
Speed Limit - data cleaning tests (script 02_prepare_data.py).

Usage:  python test_data_cleaning.py

Script 02 turns the raw CSV files into the hourly table (one row = one hour).
Instead of the real data, the tests use a tiny hand-made data set (4 hours, 1 day) in which we know
the correct answer for every cleaning step. Script 02 is started on this data in a temporary folder,
so nothing in the project folder is changed.

Six tests (each one checks one cleaning decision):
  C1  duplicate rows and rows with an empty value are removed
  C2  redundant sensors are averaged (the mean, not the sum; a missing sensor is ignored)
  C3  a missing value is filled by linear interpolation between the neighbouring hours
  C4  the speed limit is rounded to the nearest 10 km/h
  C5  near-accidents are counted per hour (SPEC4: skid or too small a distance; damage only does not count;
      accident hours 1..24 are shifted to reading hours 0..23; the empty accident record is removed)
  C6  the real data gives exactly the committed hourly tables (744 hours, no missing values, 964 near-accidents);
      needs the four raw CSV files in the folder, otherwise it is skipped (the raw data is not published)

The second part checks the tests themselves: it makes deliberately broken copies of script 02 (mutants)
and shows that at least one test fails for each of them.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "02_prepare_data.py"
BOM = "﻿"

# ---------------------------------------------------------------- the hand-made data
DEFAULTS = {1: 0, 8: 0, 11: 0,            # water (micrometers)
            2: 50000, 9: 50000, 12: 50000,  # light
            15: 10, 18: 10, 19: 10,       # temperature
            6: 0, 14: 0,                  # wind
            4: 70, 3: 80, 16: 5,          # humidity, noise, traffic
            -1: 70}                       # speed limit

# hour -> changes against DEFAULTS (None = this sensor has no reading in that hour)
HOURS = {
    0:  {1: 100, 8: 200, 11: 300, 15: 10, 18: 10, 19: 10, -1: 71},
    1:  {1: 100, 8: 600, 11: None, 15: None, 18: None, 19: None, -1: 49},
    2:  {1: 100, 8: 200, 11: 300, 15: 20, 18: 20, 19: 20, -1: 76},
    23: {15: 30, 18: 30, 19: 30},
}
EXPECTED_WATER = [200, 350, 200, 0]
EXPECTED_TEMP = [10, 15, 20, 30]       # hour 1 has no temperature at all -> the middle of 10 and 20
EXPECTED_LIMIT = [70, 50, 80, 70]      # 71 -> 70, 49 -> 50, 76 -> 80
EXPECTED_NEAR = [2, 0, 1, 1]

# accidents: day, hour (1..24), damage, injured, close car, close guardrail, skid angle
ACCIDENTS = [
    (1, 1, None, None, 15, None, None),     # near-accident, reading hour 0
    (1, 1, None, None, None, 20, None),     # near-accident, reading hour 0
    (1, 2, 5000, None, None, None, None),   # only damage -> NOT a near-accident
    (1, 3, None, None, None, None, 30),     # skid only -> near-accident, reading hour 2
    (1, 24, None, None, None, None, 10),    # hour 24 -> reading hour 23
    (1, 5, None, None, None, None, None),   # completely empty record -> removed
]


def reading_rows():
    rows = []
    for hour, changes in HOURS.items():
        values = {**DEFAULTS, **changes}
        for sensor, value in values.items():
            if value is not None:
                rows.append((1, 1, hour, 0, 0, sensor, value))
    rows.append(rows[0])                       # an exact duplicate of the first row
    rows.append((1, 1, 0, 5, 0, 8, None))      # a reading without a value
    return rows


def csv_number(x):
    return "" if x is None else str(x).replace(".", ",")


def write_inputs(folder, with_header_n1=False):
    """Writes the four CSV files in the same format as the case study (';', decimal comma, BOM)."""
    rows = reading_rows()
    body = "\n".join(";".join(csv_number(v) for v in r) for r in rows) + "\n"
    header = "Month;Day;Hour;Minute;Second;Sensor;Value\n"
    (folder / "SensorReadingsYearN_V100.csv").write_text(BOM + header + body, encoding="utf-8")
    (folder / "SensorReadingsYearNPlus1_V100.csv").write_text(BOM + body, encoding="utf-8")   # no header, like the original
    acc_header = "Month;Day;Hour;Second;LicencePlate;Damage;Injured;CloseCarCm;CloseGuardrailCm;SkidAngle\n"
    acc_body = "".join(f"1;{d};{h};10;XX 111AAA;" + ";".join(csv_number(v) for v in rest) + "\n"
                       for d, h, *rest in ACCIDENTS)
    (folder / "AccidentsYearN_V100.csv").write_text(BOM + acc_header + acc_body, encoding="utf-8")
    (folder / "AccidentsYearNPlus1_V100.csv").write_text(BOM + acc_header + acc_body, encoding="utf-8")
    return len(rows)


def run_script(script_path, folder):
    result = subprocess.run([sys.executable, str(script_path), str(folder)], capture_output=True,
                            text=True, encoding="utf-8", errors="replace", cwd=folder)
    if result.returncode != 0:
        raise RuntimeError("script 02 crashed:\n" + result.stderr[-800:])
    table = pd.read_csv(folder / "hourly_dataset_N.csv")
    log = (folder / "cleaning_log.txt").read_text(encoding="utf-8")
    return table, log


def run_on_small_data(script_path):
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        n_rows = write_inputs(folder)
        table, log = run_script(script_path, folder)
    return table, log, n_rows


# ---------------------------------------------------------------- the tests
def close(values, expected):
    return len(values) == len(expected) and all(abs(a - b) < 1e-9 for a, b in zip(values, expected))


def c1(table, log, n_rows):
    expected_line = f"Readings: {n_rows} rows, {n_rows - 2} after removing empty and duplicate rows"
    ok = expected_line in log and len(table) == 4
    return ok, f"{len(table)} hours; log says '{expected_line}': {expected_line in log}"


def c2(table, log, n_rows):
    water = [float(x) for x in table["water"]]
    light = [float(x) for x in table["light"]]
    return close(water, EXPECTED_WATER) and close(light, [50000] * 4), f"water {water}"


def c3(table, log, n_rows):
    temp = [float(x) for x in table["temp"]]
    return close(temp, EXPECTED_TEMP) and not table.isna().any().any(), f"temp {temp}"


def c4(table, log, n_rows):
    limit = [float(x) for x in table["limit"]]
    return close(limit, EXPECTED_LIMIT), f"limit {limit}"


def c5(table, log, n_rows):
    near = [int(x) for x in table["near_accidents"]]
    log_ok = "completely empty (no Damage/Injured/Close*/Skid): 1 -> removed" in log and \
             "Near-accidents according to SPEC4: 4," in log
    return near == EXPECTED_NEAR and log_ok, f"near-accidents {near}; log ok: {log_ok}"


def c6(*_):
    """Real data: nothing may change unnoticed (compare with the committed hourly tables)."""
    names = ["Case_Study_Speed_Limit_SensorReadingsYearN_V100.csv", "Case_Study_Speed_Limit_SensorReadingsYearNPlus1_V100.csv",
             "Case_Study_Speed_Limit_AccidentsYearN_V100.csv", "AccidentsYearNPlus1_V100.csv"]
    missing = [n for n in names if not (HERE / n).exists()]
    if missing:   # the raw case-study CSV files are not published (private teaching material) -> the test is skipped, not failed
        return None, f"SKIPPED: the raw CSV files are not in this folder ({len(missing)} of 4 missing)"
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for n in names:
            shutil.copy(HERE / n, folder / n)
        table, _log = run_script(SCRIPT, folder)
        table_n1 = pd.read_csv(folder / "hourly_dataset_Nplus1.csv")
    committed = pd.read_csv(HERE / "hourly_dataset_N.csv")
    committed_n1 = pd.read_csv(HERE / "hourly_dataset_Nplus1.csv")
    try:
        pd.testing.assert_frame_equal(table, committed, check_exact=False, rtol=1e-9)
        pd.testing.assert_frame_equal(table_n1, committed_n1, check_exact=False, rtol=1e-9)
        same = True
    except AssertionError:
        same = False
    limits_ok = set(table["limit"].unique()) <= {40, 50, 60, 70, 80}
    facts = len(table) == 744 and not table.isna().any().any() and int(table["near_accidents"].sum()) == 964
    return same and limits_ok and facts, f"{len(table)} hours, {int(table['near_accidents'].sum())} near-accidents, same as committed files: {same}"


SMALL_TESTS = [
    ("C1", "duplicate and empty rows are removed", c1),
    ("C2", "redundant sensors: mean, missing sensor ignored", c2),
    ("C3", "missing value is interpolated linearly", c3),
    ("C4", "speed limit is rounded to the nearest 10", c4),
    ("C5", "near-accidents per hour (SPEC4, hour shift)", c5),
]

# ---------------------------------------------------------------- part 1: run the tests
print("PART 1: data cleaning tests\n")
table, log, n_rows = run_on_small_data(SCRIPT)
print(f"{'ID':4s} {'what is checked':50s} result  details")
failed = 0
for tid, what, fn in SMALL_TESTS + [("C6", "real data = committed hourly tables", c6)]:
    ok, details = fn(table, log, n_rows)
    failed += ok is False
    print(f"{tid:4s} {what:50s} {'SKIP' if ok is None else 'PASS' if ok else 'FAIL':6s}  {details}")

# ---------------------------------------------------------------- part 2: do the tests catch bugs?
MUTANTS = {
    "duplicate rows are not removed":
        ('.dropna(subset=["Day", "Hour", "Sensor", "Value"]).drop_duplicates()', '.dropna(subset=["Day", "Hour", "Sensor", "Value"])'),
    "empty values are not removed":
        ('.dropna(subset=["Day", "Hour", "Sensor", "Value"]).drop_duplicates()', '.drop_duplicates()'),
    "sensors are summed, not averaged":
        ("pivot[present].mean(axis=1, skipna=True)", "pivot[present].sum(axis=1)"),
    "one missing sensor spoils the whole group":
        ("pivot[present].mean(axis=1, skipna=True)", "pivot[present].mean(axis=1, skipna=False)"),
    "missing values filled with 0 (no interpolation)":
        ('hourly.sort_index().interpolate(limit_direction="both")', "hourly.sort_index().fillna(0)"),
    "limit is not rounded":
        ('hourly["limit"] = (hourly["limit"] / 10).round() * 10', 'hourly["limit"] = hourly["limit"]'),
    "accident hours are not shifted (offset 0)":
        ("ACCIDENT_HOUR_OFFSET = 1", "ACCIDENT_HOUR_OFFSET = 0"),
    "skid angle is ignored in near-accidents":
        ('acc["near"] = acc[["CloseCarCm", "CloseGuardrailCm", "SkidAngle"]].notna().any(axis=1)',
         'acc["near"] = acc[["CloseCarCm", "CloseGuardrailCm"]].notna().any(axis=1)'),
    "every accident counts as a near-accident":
        ('counts = acc[acc["near"]].groupby(["Day", "hour0"]).size()', 'counts = acc.groupby(["Day", "hour0"]).size()'),
}

print("\n\nPART 2: do the tests catch bugs? (each broken copy of script 02 should fail at least one test)\n")
print(f"{'broken version of script 02':50s} failing tests")
source = SCRIPT.read_text(encoding="utf-8")
survivors = []
for name, (old, new) in MUTANTS.items():
    if old not in source:
        print(f"{name:50s} -- cannot build this mutant: the line was changed in script 02 --")
        survivors.append(name)
        continue
    with tempfile.TemporaryDirectory() as tmp:
        broken = Path(tmp) / "02_broken.py"
        broken.write_text(source.replace(old, new), encoding="utf-8")
        data_folder = Path(tmp) / "data"
        data_folder.mkdir()
        n = write_inputs(data_folder)
        try:
            t, lg = run_script(broken, data_folder)
            caught = [tid for tid, _w, fn in SMALL_TESTS if not fn(t, lg, n)[0]]
        except Exception:                      # a crash also counts as detected
            caught = ["crash"]
    if not caught:
        survivors.append(name)
    print(f"{name:50s} {', '.join(caught) if caught else '-- NOT DETECTED --'}")

print(f"\nBugs detected: {len(MUTANTS) - len(survivors)} of {len(MUTANTS)}")
print(f"Failed tests: {failed}")
sys.exit(1 if failed or survivors else 0)
