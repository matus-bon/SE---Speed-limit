"""
Speed Limit - air quality component tests (exercise SpeedLimit10c).

Usage:  python test_air_quality.py

Part 1 needs nothing else running (it includes the Open-Meteo outage tests with a fake service and a fake clock). Part 2 needs LM Studio's local server with Qwen 3.5 9B loaded.
The LLM makes a judgment, so only clear cases have a golden answer; borderline cases only have to be valid.
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

import air_quality
from air_quality import ALLOWED_REDUCTIONS, ask_llm, aqi_category, decide_reduction, forecast_with_fallback, full_decision
from speedlimit_core import combine_limits, training_ranges

failed = 0


def check(test_id, description, actual, expected):
    global failed
    ok = actual in expected if isinstance(expected, (set, tuple)) else actual == expected
    failed += not ok
    shown = "/".join(map(str, sorted(expected))) if isinstance(expected, (set, tuple)) else expected
    print(f"{test_id:4s} {description:48s} expected {str(shown):14s} actual {str(actual):14s} {'PASS' if ok else 'FAIL'}")


def offline(_air):
    raise ConnectionError("server off")


def air(aqi, pm2_5=10, pm10=10, no2=10, o3=10):
    return dict(aqi=aqi, pm2_5=pm2_5, pm10=pm10, no2=no2, o3=o3)


print("PART 1: guardrails and combination (no LLM needed)\n")
check("G1", "invalid answer 'banana', AQI 90 -> rule", decide_reduction(air(90), lambda a: "banana")["reduction"], 20)
check("G2", "value not allowed '15', AQI 10 -> rule", decide_reduction(air(10), lambda a: "15")["reduction"], 0)
check("G3", "LLM server off, AQI 120 -> rule", decide_reduction(air(120), offline)["reduction"], 30)
check("G4", "server off is reported as fallback", decide_reduction(air(120), offline)["source"], "fallback rule")
check("G5", "valid answer '10' is used", decide_reduction(air(70), lambda a: "10\n")["source"], "llm")
check("C1", "weather 80, air -0 -> 80 (REQ3)", combine_limits(80, 0), 80)
check("C2", "weather 80, air -20 -> 60 (REQ2)", combine_limits(80, 20), 60)
check("C3", "weather 50, air -10 -> stricter weather 50", combine_limits(50, 10), 50)
check("C4", "weather 60, air -30 -> stricter air 50", combine_limits(60, 30), 50)


# ---- outage of Open-Meteo (fallback option A: last known forecast up to 3 hours, otherwise no reduction + alert)
def down(*_):
    raise ConnectionError("Open-Meteo is down")


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


model = joblib.load(Path(__file__).with_name("speedlimit_model.joblib"))
KNOWN = air(70)
air_quality._last_forecast.clear()
clock = Clock()
live = forecast_with_fallback(fetch=lambda lat, lon: KNOWN, now=clock)
check("F1", "service works -> live forecast, no alert", [live["source"], live["alert"] is None], ["live", True])
clock.t += 3600
stored = forecast_with_fallback(fetch=down, now=clock)
check("F2", "down after 1 h -> last known forecast + alert", [stored["source"], stored["air"] == KNOWN, stored["alert"] is not None], ["stored", True, True])
clock.t += 4 * 3600
old = forecast_with_fallback(fetch=down, now=clock)
check("F3", "down, stored forecast 5 h old -> no forecast", [old["source"], old["air"]], ["none", None])
air_quality._last_forecast.clear()
nothing = forecast_with_fallback(fetch=down, now=clock)
check("F4", "down, nothing stored -> no forecast + alert", [nothing["source"], nothing["alert"] is not None], ["none", True])
calls = []
air_quality._last_forecast.clear()
forecast_with_fallback(fetch=lambda lat, lon: calls.append(1) or KNOWN, now=clock)
clock.t += 300
forecast_with_fallback(fetch=lambda lat, lon: calls.append(1) or KNOWN, now=clock)
check("F5", "second call within 10 minutes -> one request only", len(calls), 1)
r = full_decision(model, 50000, 0, 10, None)
check("F6", "no air data, good weather -> 80 km/h, reduction 0", [r["limit"], r["reduction"], r["source"], bool(r["alerts"])], [80, 0, "no air data", True])
r = full_decision(model, 0, 0, 5, None)
check("F7", "no air data, dark -> the weather limit only", r["limit"], r["weather_limit"])
ranges = training_ranges(pd.read_csv(Path(__file__).with_name("hourly_dataset_N.csv")))
r = full_decision(model, 50000, 10000, -1, air(20), ranges=ranges)
check("F8", "water 10000 um (outside training data) -> alert", [r["limit"] <= 70, any("water" in a for a in r["alerts"])], [True, True])

print("\n\nPART 2: live LLM\n")
try:
    ask_llm(air(10))
except Exception:
    print("NOT RUN: LM Studio server is not reachable.")
else:
    NO_REDUCTION = {0}
    CLEAR_REDUCTION = {20, 30}
    CASES = [
        ("L1", "very clean", air(12, 4, 8, 8, 40), NO_REDUCTION),
        ("L2", "clean", air(25, 8, 14, 25, 55), NO_REDUCTION),
        ("L3", "fair", air(36, 14, 25, 35, 70), NO_REDUCTION),
        ("L4", "very poor", air(92, 60, 120, 200, 150), CLEAR_REDUCTION),
        ("L5", "extremely poor", air(130, 110, 210, 300, 100), CLEAR_REDUCTION),
        ("L6", "severe smog", air(250, 250, 400, 420, 60), CLEAR_REDUCTION),
        ("N1", "moderate", air(55, 22, 45, 70, 110), set(ALLOWED_REDUCTIONS)),
        ("N2", "poor, all pollutants raised", air(72, 38, 70, 100, 120), set(ALLOWED_REDUCTIONS)),
        ("N3", "poor, NO2 driven", air(75, 15, 30, 190, 40), set(ALLOWED_REDUCTIONS)),
        ("N4", "poor, ozone driven", air(70, 10, 20, 30, 190), set(ALLOWED_REDUCTIONS)),
    ]
    for test_id, name, values, expected in CASES:
        result = decide_reduction(values)
        ok_source = result["source"] == "llm"
        check(test_id, f"{name} (AQI {values['aqi']}, {aqi_category(values['aqi'])})", result["reduction"] if ok_source else "fallback", expected)

print(f"\nFailed tests: {failed}")
sys.exit(1 if failed else 0)
