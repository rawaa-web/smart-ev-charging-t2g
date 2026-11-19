"""Chargement des observations Kaggle et simulation de l'état des bornes."""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "ev_charging_patterns.csv"
VEHICLE = "Vehicle Model"
BATTERY = "Battery Capacity (kWh)"
STATION_ID = "Charging Station ID"
LOCATION = "Charging Station Location"
CHARGER = "Charger Type"
RATE = "Charging Rate (kW)"
START = "Charging Start Time"
END = "Charging End Time"
SOC_START = "State of Charge (Start %)"
SOC_END = "State of Charge (End %)"
ENERGY = "Energy Consumed (kWh)"
COST = "Charging Cost (USD)"
DURATION = "Charging Duration (hours)"

AVAILABLE = "Disponible"
CHARGING = "Charge V1G"
EXPORTING = "Décharge V2G"
MIN_V2G_SOC = 30.0
SLOT_MINUTES = 15
EXPORT_EFFICIENCY = 0.90

V2G_MODELS = frozenset(
    name.casefold()
    for name in (
        "Nissan Leaf",
        "Nissan e-NV200",
        "Mitsubishi Outlander PHEV",
        "Volkswagen ID.3",
        "Volkswagen ID.4",
        "Renault 5 E-Tech",
        "BMW iX3",
        "Ford F-150 Lightning",
    )
)


def load_charging_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Conserve les sessions exploitables et enrichit leurs modèles avec le référentiel V2G."""
    text_columns = (VEHICLE, STATION_ID, LOCATION, CHARGER)
    positive_columns = (BATTERY, RATE, ENERGY, DURATION)
    numeric_columns = (*positive_columns, COST, SOC_START, SOC_END)
    required = (*text_columns, *numeric_columns, START, END)
    data = pd.read_csv(path, dtype={STATION_ID: "string", "User ID": "string"})
    missing = [column for column in required if column not in data]
    if missing:
        raise ValueError(f"Colonnes Kaggle manquantes ({', '.join(missing)})")

    for column in text_columns:
        data[column] = data[column].astype("string").str.strip().replace("", pd.NA)
    for column in numeric_columns:
        values = pd.to_numeric(data[column], errors="coerce")
        data[column] = values.where(np.isfinite(values))
    for column in (START, END):
        data[column] = pd.to_datetime(data[column], errors="coerce", format="mixed")

    data = data.dropna(subset=list(required))
    valid = (
        data[list(positive_columns)].gt(0).all(axis=1)
        & data[COST].ge(0)
        & data[SOC_START].between(0, 100)
        & data[SOC_END].between(0, 100)
        & data[SOC_END].ge(data[SOC_START])
        & data[END].gt(data[START])
    )
    data = data.loc[valid].drop_duplicates().reset_index(drop=True)
    if data.empty:
        raise ValueError("Le CSV ne contient aucune session exploitable après validation.")
    data["is_v2g_compatible"] = data[VEHICLE].str.casefold().isin(V2G_MODELS)
    return data


def get_stations_status(df: pd.DataFrame, *, network_peak: bool = False, seed: int = 42) -> pd.DataFrame:
    """Simule une occupation stable et actualise l'injection lors d'une pointe."""
    site_keys = [STATION_ID, LOCATION]
    group_keys = [*site_keys, CHARGER]
    groups = df.groupby(group_keys, sort=True)
    stations = groups.agg(
        **{
            RATE: (RATE, "median"),
            "session_count": (RATE, "size"),
        }
    ).reset_index()
    # Plusieurs types peuvent être associés au même site. Le plus observé représente le site.
    stations = stations.sort_values("session_count", ascending=False, kind="stable")
    stations = stations.drop_duplicates(site_keys).sort_values([LOCATION, STATION_ID])
    local_totals = df.groupby([LOCATION, CHARGER])[[COST, ENERGY]].sum()
    local_price = (local_totals[COST] / local_totals[ENERGY]).rename("price_usd_per_kwh")
    stations = stations.join(local_price, on=[LOCATION, CHARGER], validate="many_to_one")

    connected_vehicle = df.sort_values(START, kind="stable").drop_duplicates(group_keys, keep="last")
    connected_vehicle = connected_vehicle[[*group_keys, VEHICLE, BATTERY, SOC_START, "is_v2g_compatible"]]
    stations = stations.merge(connected_vehicle, on=group_keys, validate="one_to_one")

    # Le CSV ne documente ni le matériel bidirectionnel ni les sessions en cours.
    stations["export_limit_kw"] = np.where(
        stations[CHARGER].eq("DC Fast Charger"), stations[RATE].clip(upper=20) / 2, 0
    )
    occupancy, consent, loading = np.random.default_rng(seed).random((3, len(stations)))
    connected = occupancy < 0.55
    eligible = stations["is_v2g_compatible"] & (consent < 0.65)
    headroom_kwh = (stations[SOC_START] - MIN_V2G_SOC).clip(lower=0) / 100 * stations[BATTERY]
    safe_export_kw = headroom_kwh * EXPORT_EFFICIENCY / (SLOT_MINUTES / 60)
    export_kw = np.minimum(stations["export_limit_kw"], safe_export_kw)
    exporting = network_peak & connected & eligible & (export_kw > 0)

    stations["status"] = np.select([exporting, connected], [EXPORTING, CHARGING], default=AVAILABLE)
    stations["connected_vehicle"] = stations[VEHICLE].where(connected, pd.NA)
    stations["power_kw"] = np.select(
        [exporting, connected], [-export_kw, stations[RATE] * (0.45 + 0.55 * loading)], default=0.0
    )
    return stations.drop(columns=[VEHICLE, BATTERY, SOC_START, "is_v2g_compatible"])
