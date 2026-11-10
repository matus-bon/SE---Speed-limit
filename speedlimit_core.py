"""
Speed Limit - system core (shared by the scripts, and later by the GUI and the tests).

Contains:
  * constants from the specification (SPEC1, SPEC2, speed limits),
  * make_features()      - turns the hourly table into the neural network inputs,
  * is_dark(), is_ice_risk(), router_uses_nn()  - the ROUTER (Software 1.0, no AI),
  * choose_limit()       - picks the reduced speed limit with the neural network (Software 2.0),
  * final_speed_limit()  - the complete weather decision according to REQ1 and REQ3.
"""
import numpy as np
import pandas as pd

# ---- Specification from the case study -----------------------------------------
DARK_BELOW = 500          # SPEC1: dark = light below 500 (millilux; the data gives the same result as for lux, see script 02)
ICE_WATER_ABOVE = 1000    # SPEC2: more than 1000 micrometers of water ...
ICE_TEMP_BELOW = 0        # ... and temperature below zero
DEFAULT_LIMIT = 80        # REQ3: used when there is no reason to reduce the limit
# REQ1 requires a reduction. 80 km/h is also excluded because the operator always reduced the limit
# in dark or icy hours, so the data contains no such hours at 80 km/h and the network cannot judge it.
LIMITS_TO_TRY = [70, 60, 50, 40]       # from the highest (most permissive) to the lowest
MAX_NEAR_ACCIDENTS_PER_HOUR = 1.0      # REQ1: on average at most one near-accident per hour

FEATURES = ["limit", "log_water", "log_light", "temp", "wind", "humidity", "noise", "traffic"]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Neural network inputs. df must contain: limit, water, light, temp, wind, humidity, noise, traffic.
    water and light have a huge range (0 to 100000), so we apply a logarithm (part of the normalization)."""
    out = pd.DataFrame(index=df.index)
    out["limit"] = df["limit"]
    out["log_water"] = np.log1p(df["water"].clip(lower=0))
    out["log_light"] = np.log1p(df["light"].clip(lower=0))
    out["temp"] = df["temp"]
    out["wind"] = df["wind"]
    out["humidity"] = df["humidity"]
    out["noise"] = df["noise"]
    out["traffic"] = df["traffic"]
    return out[FEATURES]


# ---- ROUTER (Software 1.0) -----------------------------------------------------
def is_dark(light) -> bool:
    return light < DARK_BELOW


def is_ice_risk(water, temp) -> bool:
    return water > ICE_WATER_ABOVE and temp < ICE_TEMP_BELOW


def router_uses_nn(light, water, temp) -> bool:
    """True = conditions are bad (dark or ice risk), so the neural network has to decide."""
    return is_dark(light) or is_ice_risk(water, temp)


# ---- Input range check (human in the loop) ------------------------------------
RANGE_INPUTS = {"water": "water on the road (µm)", "light": "light", "temp": "temperature (°C)"}


def training_ranges(df: pd.DataFrame) -> dict:
    """Smallest and largest value of each checked input in the training data: {name: (min, max)}."""
    return {name: (float(df[name].min()), float(df[name].max())) for name in RANGE_INPUTS}


def range_alerts(ranges: dict, light, water, temp) -> list:
    """Alert texts for inputs outside the range the network was trained on (it would extrapolate, so its answer is unreliable).
    Only checked when the network decides (dark or ice risk); otherwise the network is not used."""
    if not router_uses_nn(light, water, temp):
        return []
    values = {"water": water, "light": light, "temp": temp}
    alerts = []
    for name, label in RANGE_INPUTS.items():
        low, high = ranges[name]
        if not low <= values[name] <= high:
            alerts.append(f"{label} = {values[name]:g} is outside the training data ({low:g} to {high:g}): "
                          f"the network has to extrapolate and its prediction is unreliable.")
    return alerts


# ---- Limit selection (Software 2.0 plus the code wrapped around it) ------------
def choose_limit(model, conditions: dict):
    """Picks the HIGHEST limit of 70, 60, 50, 40 for which the network predicts at most 1 near-accident per hour.
    conditions: dict with the keys water, light, temp, wind, humidity, noise, traffic.
    Returns (limit, predictions_for_every_limit, safe_limit_found: bool).
    If no limit satisfies REQ1, returns the lowest one (40) and False - a safety fallback (guardrail)."""
    rows = pd.DataFrame([{**conditions, "limit": lim} for lim in LIMITS_TO_TRY])
    preds = np.clip(model.predict(make_features(rows)), 0, None)   # output guardrail: no negative counts
    by_limit = {lim: round(float(p), 3) for lim, p in zip(LIMITS_TO_TRY, preds)}
    for lim in LIMITS_TO_TRY:
        if by_limit[lim] <= MAX_NEAR_ACCIDENTS_PER_HOUR:
            return lim, by_limit, True
    return min(LIMITS_TO_TRY), by_limit, False


def final_speed_limit(model, light, water, temp, wind=0.0, humidity=80.0, noise=95.0, traffic=15.0):
    """REQ1 + REQ3: if it is dark or there is ice risk, the network decides; otherwise 80 km/h."""
    if not router_uses_nn(light, water, temp):
        return DEFAULT_LIMIT, {}, True
    cond = dict(water=water, light=light, temp=temp, wind=wind, humidity=humidity, noise=noise, traffic=traffic)
    return choose_limit(model, cond)


def combine_limits(weather_limit, air_reduction):
    """REQ2: the stricter of the weather limit and the air quality limit applies."""
    return min(weather_limit, DEFAULT_LIMIT - air_reduction)
