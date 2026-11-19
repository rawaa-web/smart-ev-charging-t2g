"""Génère les CSV de démonstration T2G à Dakar, sans dépendance externe.

Les puissances, capacités, implantations et tarifs sont des hypothèses de simulation.
La colonne is_v2g reprend les règles d'éligibilité du projet.
"""

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from random import Random
from typing import Literal


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_START_DATE = date(2026, 11, 19)
SLOT_MINUTES = 15
CHARGE_EFFICIENCY = 0.90


@dataclass(frozen=True)
class Vehicle:
    vehicle_model: str
    battery_capacity_kwh: float
    is_v2g: bool
    max_ac_power_kw: float
    max_dc_power_kw: float


@dataclass(frozen=True)
class Station:
    station_id: str
    location: str
    charger_type: Literal["AC", "DC"]
    max_power_kw: float
    is_available: bool
    max_export_power_kw: float
    transformer_id: str


DEMO_VEHICLE = Vehicle("Renault 5 E-Tech", 52, True, 11, 100)
VEHICLES = (
    Vehicle("Nissan Leaf", 40, True, 6.6, 50),
    Vehicle("Nissan e-NV200", 40, True, 6.6, 50),
    Vehicle("Mitsubishi Outlander PHEV", 20, True, 3.7, 22),
    Vehicle("Volkswagen ID.3", 58, True, 11, 120),
    Vehicle("Volkswagen ID.4", 77, True, 11, 135),
    DEMO_VEHICLE,
    Vehicle("BMW iX3", 74, True, 11, 150),
    Vehicle("Ford F-150 Lightning", 131, True, 11, 150),
    Vehicle("Tesla Model 3", 60, False, 11, 170),
    Vehicle("Tesla Model Y", 75, False, 11, 170),
    Vehicle("Hyundai Ioniq 5", 77.4, False, 11, 220),  # V2L uniquement
    Vehicle("Kia EV6", 77.4, False, 11, 220),
    Vehicle("Dacia Spring", 26.8, False, 6.6, 30),
    Vehicle("Renault Zoe", 52, False, 22, 0),
)

DEMO_STATION = Station("ST-01", "Plateau", "AC", 11, True, 7.4, "TR-01")
STATIONS = (
    DEMO_STATION,
    Station("ST-02", "Corniche Ouest", "DC", 100, True, 0, "TR-01"),
    Station("ST-03", "Point E", "AC", 22, True, 0, "TR-01"),
    Station("ST-04", "Mermoz", "AC", 7.4, True, 0, "TR-02"),
    Station("ST-05", "Ouakam", "DC", 60, True, 20, "TR-02"),
    Station("ST-06", "Almadies", "DC", 50, True, 25, "TR-02"),
    Station("ST-07", "Liberté 6", "AC", 11, True, 0, "TR-03"),
    Station("ST-08", "Sacré-Cœur", "AC", 22, True, 11, "TR-03"),
    Station("ST-09", "Grand Yoff", "DC", 120, True, 0, "TR-03"),
    Station("ST-10", "Hann Maristes", "AC", 7.4, False, 0, "TR-03"),
)

# Limite de puissance active et charge de fond hors bornes, en kW.
TRANSFORMERS = {
    "TR-01": (130, 35),
    "TR-02": (95, 25),
    "TR-03": (150, 45),
}

# Minutes depuis minuit. Une fin supérieure à 1440 indique un départ le lendemain.
AC_WINDOWS = ((8 * 60, 12 * 60), (13 * 60, 17 * 60), (18 * 60, 31 * 60))
DC_WINDOWS = ((8 * 60, 10 * 60), (12 * 60, 14 * 60), (18 * 60, 20 * 60))
TARIFF_PERIODS = (
    (0, 6 * 60, "HEURES_CREUSES", 95, 0),
    (6 * 60, 18 * 60, "HEURES_PLEINES", 135, 0),
    (18 * 60, 22 * 60, "POINTE", 185, 230),
    (22 * 60, 24 * 60, "HEURES_CREUSES", 95, 0),
)


def make_charging_sessions(start_date: date, days: int, rng: Random) -> list[dict]:
    """Simule des fenêtres de branchement distinctes par borne."""
    sessions = []
    dc_vehicles = tuple(vehicle for vehicle in VEHICLES if vehicle.max_dc_power_kw > 0)
    # Le scénario utilise l'heure locale de Dakar (UTC+00:00).
    first_midnight = datetime.combine(start_date, time.min, tzinfo=timezone.utc)
    for day_offset in range(days):
        midnight = first_midnight + timedelta(days=day_offset)
        for station in STATIONS:
            if not station.is_available:
                continue
            is_ac = station.charger_type == "AC"
            windows = AC_WINDOWS if is_ac else DC_WINDOWS
            vehicles = VEHICLES if is_ac else dc_vehicles

            for start_minute, end_minute in windows:
                is_demo = (
                    day_offset == 0
                    and station == DEMO_STATION
                    and (start_minute, end_minute) == AC_WINDOWS[-1]
                )
                if is_demo:
                    # Véhicule présent à la pointe et encore branché en heures creuses.
                    vehicle = DEMO_VEHICLE
                    start_soc, target_soc = 65, 80
                    consent = True
                else:
                    if rng.random() < 0.15:
                        continue
                    vehicle = rng.choice(vehicles)
                    start_minute += rng.randint(0, 2) * SLOT_MINUTES
                    end_minute -= rng.randint(0, 2) * SLOT_MINUTES
                    target_soc = rng.choices((70, 80, 90, 100), weights=(15, 55, 25, 5))[0]

                    vehicle_power_kw = vehicle.max_ac_power_kw if is_ac else vehicle.max_dc_power_kw
                    power_kw = min(vehicle_power_kw, station.max_power_kw)
                    hours_connected = (end_minute - start_minute) / 60
                    energy_kwh = power_kw * hours_connected * CHARGE_EFFICIENCY
                    max_soc_gain = int(100 * energy_kwh / vehicle.battery_capacity_kwh)
                    min_start_soc = max(10, target_soc - max_soc_gain)
                    # Les fenêtres et puissances ci-dessus laissent au moins 5 points de charge.
                    most_likely_soc = max(min_start_soc, target_soc - 30)
                    start_soc = round(rng.triangular(min_start_soc, target_soc - 5, most_likely_soc))
                    consent = vehicle.is_v2g and rng.random() < 0.70

                arrival = midnight + timedelta(minutes=start_minute)
                departure = midnight + timedelta(minutes=end_minute)
                sessions.append(
                    {
                        **asdict(station),
                        "vehicle_model": vehicle.vehicle_model,
                        "battery_capacity_kwh": vehicle.battery_capacity_kwh,
                        "soc_start_percent": start_soc,
                        "soc_target_percent": target_soc,
                        "arrival_at": arrival.isoformat(timespec="minutes"),
                        "departure_at": departure.isoformat(timespec="minutes"),
                        "v2g_consent": consent,
                    }
                )

    sessions.sort(key=lambda session: (session["arrival_at"], session["station_id"]))
    return [{"session_id": f"SES-{index:05d}", **session} for index, session in enumerate(sessions, 1)]


def make_tariffs() -> list[dict]:
    """Développe la grille journalière récurrente en 96 créneaux de 15 minutes."""
    return [
        {
            "time_local": f"{minute // 60:02d}:{minute % 60:02d}",
            "period": period,
            "import_price_xof_per_kwh": import_price,
            "export_price_xof_per_kwh": export_price,
        }
        for start, end, period, import_price, export_price in TARIFF_PERIODS
        for minute in range(start, end, SLOT_MINUTES)
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Dossier des CSV générés")
    parser.add_argument(
        "--start-date", type=date.fromisoformat, default=DEFAULT_START_DATE, help="Premier jour (AAAA-MM-JJ)"
    )
    parser.add_argument("--days", type=int, default=14, help="Nombre de jours d'arrivées")
    parser.add_argument("--seed", type=int, default=42, help="Graine de la simulation")
    args = parser.parse_args()
    if args.days < 1:
        parser.error("--days doit être supérieur ou égal à 1")
    if args.days > (date.max - args.start_date).days:
        parser.error("Les départs de nuit dépassent la date maximale prise en charge")

    tables = {
        "vehicles.csv": [asdict(vehicle) for vehicle in VEHICLES],
        "stations.csv": [asdict(station) for station in STATIONS],
        "transformers.csv": [
            {"transformer_id": identifier, "max_load_kw": limit, "base_load_kw": base_load}
            for identifier, (limit, base_load) in TRANSFORMERS.items()
        ],
        "tariffs.csv": make_tariffs(),
        "charging_sessions.csv": make_charging_sessions(args.start_date, args.days, Random(args.seed)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        path = args.output_dir / name
        write_csv(path, rows)
        print(f"{path} ({len(rows)} lignes)")


if __name__ == "__main__":
    main()
