"""
Phase 5: Data Collection & Data Understanding
------------------------------------------------
DATA SOURCE
-----------
This project uses the nycflights13 dataset — real, complete on-time
performance data for all 336,776 flights departing NYC's three airports
(EWR, JFK, LGA) in 2013, maintained by Hadley Wickham/Posit as a
teaching dataset built directly from the US Bureau of Transportation
Statistics:
    https://github.com/hadley/nycflights13

Five related tables are used:
    - flights.csv  — one row per flight: schedule, actual times, delay,
                      carrier, tail number, origin/destination, distance
    - weather.csv  — hourly weather observations at each NYC origin
                      airport (temperature, wind, precipitation, etc.)
    - airlines.csv — carrier code -> full airline name
    - planes.csv   — aircraft metadata by tail number, including
                      manufacture year (used to engineer plane age)
    - airports.csv — airport metadata (not used directly in this
                      project's feature set, included for completeness)

If you're following along fresh: all five CSVs are already included at
`data/`.

WHY THIS DATASET
-----------------
- It's real, complete-year operational data (not a sample), with a
  companion weather table already keyed for a direct join — most public
  flight-delay datasets don't ship with matched weather data at all.
- It supports a genuinely operational framing: predicting departure
  delay (the FAA/DOT standard >=15-minute definition) ahead of time to
  support proactive crew/gate planning — directly matching this
  project's "improve operational & crew planning" objective.
- Delay clearly has real temporal structure (see eda.py: hour-of-day and
  month both show strong, sensible patterns), which is what justifies
  the time-based train/val/test split this project uses (see split.py)
  rather than a random one.
"""

import pandas as pd


def load_raw_data(data_dir: str = "data"):
    flights = pd.read_csv(f"{data_dir}/flights.csv")
    weather = pd.read_csv(f"{data_dir}/weather.csv")
    airlines = pd.read_csv(f"{data_dir}/airlines.csv")
    planes = pd.read_csv(f"{data_dir}/planes.csv")
    return flights, weather, airlines, planes


if __name__ == "__main__":
    flights, weather, airlines, planes = load_raw_data()
    print(f"Flights: {len(flights)} rows  |  Weather: {len(weather)} rows  |  Airlines: {len(airlines)}  |  Planes: {len(planes)}")
    n_cancelled = flights["dep_time"].isnull().sum()
    print(f"\nCancelled flights (no dep_time recorded): {n_cancelled} ({n_cancelled / len(flights):.2%})")
    print(f"\nWeather missing values:")
    print(weather.isnull().sum()[weather.isnull().sum() > 0])
