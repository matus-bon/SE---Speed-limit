"""
Speed Limit - air quality component (REQ2, Software 3.0).

Flow:  Open-Meteo forecast -> local LLM (Qwen 3.5 9B in LM Studio) -> validated reduction in km/h.

The operator has no air quality training data, so the LLM judges the raw values with its own knowledge.
The prompt deliberately contains no thresholds or reduction table.
The LLM may only answer 0, 10, 20 or 30. If it is not reachable or answers anything else,
rule_reduction() decides instead, so the system always returns a valid value.
"""
import re

import requests

from speedlimit_core import combine_limits, final_speed_limit

KREMS = (48.41, 15.60)
FORECAST_HOURS = 3      # REQ2 speaks of *predicted* air quality: we take the worst of the next hours
OPEN_METEO_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPEN_METEO_FIELDS = {"aqi": "european_aqi", "pm2_5": "pm2_5", "pm10": "pm10", "no2": "nitrogen_dioxide", "o3": "ozone"}

LLM_URL = "http://localhost:1234/v1/completions"
LLM_MODEL = "qwen/qwen3.5-9b"
ALLOWED_REDUCTIONS = (0, 10, 20, 30)

PROMPT = """You advise a highway operator. Speed limits can be lowered to reduce traffic emissions when air pollution is expected.
The normal limit is 80 km/h. Use your own knowledge of air pollution and its health effects.

Forecast air quality near the highway for the next hours:
European AQI: {aqi}
PM2.5: {pm2_5} µg/m³
PM10: {pm10} µg/m³
NO2: {no2} µg/m³
Ozone: {o3} µg/m³

Should the limit be reduced because of air quality, and by how much?
Answer with exactly one number and nothing else: 0, 10, 20 or 30 (km/h reduction)."""

# Qwen's long reasoning takes minutes on a 16 GB laptop and the usual settings do not turn it off.
# An empty <think> block in Qwen's chat format makes it answer directly (about 2.3 s).
RAW_FORMAT = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def fetch_forecast(lat=KREMS[0], lon=KREMS[1], hours=FORECAST_HOURS, timeout=10):
    """Worst forecast value of the next hours for the AQI and each pollutant. Raises an exception if the service fails."""
    params = {"latitude": lat, "longitude": lon, "hourly": ",".join(OPEN_METEO_FIELDS.values()),
              "forecast_hours": hours, "timezone": "auto"}
    response = requests.get(OPEN_METEO_URL, params=params, timeout=timeout)
    response.raise_for_status()
    hourly = response.json()["hourly"]
    air = {}
    for key, field in OPEN_METEO_FIELDS.items():
        values = [v for v in hourly[field] if v is not None]
        if not values:
            raise ValueError(f"Open-Meteo returned no values for {field}")
        air[key] = max(values)
    return air


def aqi_category(aqi):
    """Official band name of a European AQI value."""
    for upper, name in [(20, "good"), (40, "fair"), (60, "moderate"), (80, "poor"), (100, "very poor")]:
        if aqi <= upper:
            return name
    return "extremely poor"


def rule_reduction(aqi):
    """Fallback when the LLM cannot decide, based on the European AQI bands."""
    if aqi <= 60:
        return 0
    if aqi <= 80:
        return 10
    if aqi <= 100:
        return 20
    return 30


def parse_reduction(text):
    """Output guardrail: returns the reduction, or None unless the answer is exactly an allowed value."""
    match = re.fullmatch(r"\s*(\d+)\s*(km/h)?\s*\.?\s*", text or "")
    if not match or int(match.group(1)) not in ALLOWED_REDUCTIONS:
        return None
    return int(match.group(1))


def ask_llm(air, timeout=30):
    """Sends the forecast to the local LLM and returns its raw text answer."""
    body = {"model": LLM_MODEL, "temperature": 0, "seed": 42, "max_tokens": 8, "stop": ["<|im_end|>"],
            "prompt": RAW_FORMAT.format(prompt=PROMPT.format(**air))}
    response = requests.post(LLM_URL, json=body, timeout=timeout)
    response.raise_for_status()
    return response.json()["choices"][0]["text"]


def decide_reduction(air, llm=ask_llm):
    """Returns {"reduction": km/h, "source": "llm" or "fallback rule", "raw": the LLM's answer or the error}."""
    try:
        raw = llm(air)
    except Exception as error:      # server off, timeout, unexpected response
        return {"reduction": rule_reduction(air["aqi"]), "source": "fallback rule", "raw": f"LLM unavailable: {type(error).__name__}"}
    reduction = parse_reduction(raw)
    if reduction is None:
        return {"reduction": rule_reduction(air["aqi"]), "source": "fallback rule", "raw": raw}
    return {"reduction": reduction, "source": "llm", "raw": raw}


def full_decision(model, light, water, temp, air, llm=ask_llm):
    """The complete system: router and network (REQ1, REQ3), then air quality (REQ2)."""
    weather_limit, predictions, safe_limit_found = final_speed_limit(model, light, water, temp)
    decision = decide_reduction(air, llm)
    return {"limit": combine_limits(weather_limit, decision["reduction"]), "weather_limit": weather_limit,
            "predictions": predictions, "safe_limit_found": safe_limit_found,
            "aqi": air["aqi"], "category": aqi_category(air["aqi"]), **decision}
