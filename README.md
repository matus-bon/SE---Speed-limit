# 🚗 Speed Limit

Dynamic speed limit for a highway: the system chooses a limit (40–80 km/h) every hour from weather sensors and an air quality forecast.
Project for *Software Engineering for AI* (IMC Krems, SEA1VO) by Matúš Bondor and Aleksander Juraszek.

## How it works

```
sensor data ──► router (rules) ──► neural network ──► weather limit ─┐
                                                                      ├─► min(weather limit, 80 − reduction) ──► GUI
Open-Meteo forecast ──► local LLM ──► guardrail / fallback rule ──────┘
```

| Part | What it does | Style |
|---|---|---|
| Router (`speedlimit_core.py`) | Dark (light < 500 millilux) or ice risk (water > 1000 µm and temperature < 0 °C)? If not, the limit is 80 km/h. | Software 1.0 |
| Neural network (`03_train_model.py`) | Predicts near-accidents per hour for 70, 60, 50 and 40 km/h; the highest limit with at most 1 predicted near-accident is chosen, otherwise 40 km/h. | Software 2.0 |
| Air quality component (`air_quality.py`) | A local LLM (Qwen 3.5 9B in LM Studio) judges the forecast and reduces the limit by 0, 10, 20 or 30 km/h. A guardrail rejects any other answer and a rule by AQI takes over. | Software 3.0 |
| GUI (`app.py`) | Streamlit app showing the final limit and the reason. | |

The lowest possible result is 40 km/h. A human stays in the loop: the dispatcher can override the limit and gets an alert in the GUI when the system is unsure (fallback rule used, no air quality forecast, an input outside the range of the training data, no safe limit found).

If Open-Meteo is down, the last known forecast (up to 3 hours old) is used; without one, no air quality reduction is applied and an alert is shown.

## Finding

The network is **not better than the simple mean** (test MSE 1.40 vs. 1.28; see `training_report.txt`).
That is why safety rests on the rules before it, the guardrail after it and the human above it.

## Setup

```
python -m venv .venv
.venv\Scripts\activate          (Windows)   |   source .venv/bin/activate   (macOS/Linux)
pip install -r requirements.txt
```

## Run

| Step | Command | Result |
|---|---|---|
| 1. Explore the raw data | `python 01_explore_data.py ["path to the CSV files"]` | `explore_report.txt` |
| 2. Clean and build the hourly table | `python 02_prepare_data.py ["path to the CSV files"]` | `hourly_dataset_N.csv`, `hourly_dataset_Nplus1.csv`, `cleaning_log.txt` |
| 3. Train and evaluate the network | `python 03_train_model.py` | `speedlimit_model.joblib`, `training_report.txt` |
| 4. Start the GUI | `streamlit run app.py` | browser opens |

The GUI works without the LLM (the fallback rule is used). For the LLM start the LM Studio local server on port 1234 with Qwen 3.5 9B loaded.

## Tests

| Command | What it checks |
|---|---|
| `python test_router.py` | 4 boundary tests of the router and 14 deliberately broken routers |
| `python test_data_cleaning.py` | 6 tests of the cleaning script (the last one compares the result with the real data) and 9 deliberately broken versions |
| `python test_neural_network.py` | 6 tests of the network, the limit selection and the input range check, and 12 deliberately broken versions |
| `python test_air_quality.py` | guardrail, fallback rule, combiner, Open-Meteo outage (17 tests); part 2 (10 tests with golden answers) needs LM Studio |
| `python test_end_to_end.py` | the whole chain; needs LM Studio |

## Data

The raw case-study CSV files (sensors, sensor readings, accidents of year N and N+1) are included, so all scripts and all tests run as they are.
They are course material: please keep this repository private.
