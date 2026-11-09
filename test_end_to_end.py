"""
Speed Limit - end-to-end test: inputs -> router -> neural network -> local LLM -> final limit.

Usage:  python test_end_to_end.py        (LM Studio's local server with Qwen 3.5 9B must be running)
Only clear air quality cases are used; where the LLM may choose 20 or 30, either final limit is accepted.
"""
import sys
from pathlib import Path

import joblib

from air_quality import fetch_forecast, full_decision

model = joblib.load(Path(__file__).with_name("speedlimit_model.joblib"))

CLEAN = dict(aqi=25, pm2_5=8, pm10=14, no2=25, o3=55)
VERY_POOR = dict(aqi=92, pm2_5=60, pm10=120, no2=200, o3=150)
EXTREME = dict(aqi=130, pm2_5=110, pm10=210, no2=300, o3=100)

# name, light, water, temperature, air quality, accepted limits
SCENARIOS = [
    ("E1 day, dry, clean air",                  50000,    0, 10, CLEAN,     {80}),
    ("E2 day, dry, very poor air",              50000,    0, 10, VERY_POOR, {50, 60}),
    ("E3 night, dry, clean air",                    0,    0,  5, CLEAN,     {40}),
    ("E4 day, wet, frost, extremely poor air",  50000, 3000, -4, EXTREME,   {50, 60}),
    ("E5 night, wet, frost, clean air",             0, 3000, -4, CLEAN,     {70}),
]

failed = 0
print(f"{'scenario':42s} {'weather':>7s} {'AQI':>4s} {'air':>4s} {'by':14s} {'limit':>5s}  accepted")
for name, light, water, temp, air, accepted in SCENARIOS:
    r = full_decision(model, light, water, temp, air)
    ok = r["limit"] in accepted and r["source"] == "llm"
    failed += not ok
    print(f"{name:42s} {r['weather_limit']:7d} {air['aqi']:4d} {-r['reduction']:4d} {r['source']:14s} {r['limit']:5d}  {sorted(accepted)} {'PASS' if ok else 'FAIL'}")

live = fetch_forecast()
r = full_decision(model, 50000, 0, 10, live)
print(f"\nLive forecast for Krems: {live}\n  -> {r['category']}, reduction {r['reduction']} by {r['source']}, limit {r['limit']} km/h")
print(f"Failed scenarios: {failed}")
sys.exit(1 if failed else 0)
