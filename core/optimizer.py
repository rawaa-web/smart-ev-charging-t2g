"""Planification V1G/V2G à partir des observations Kaggle nettoyées."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite

import pandas as pd

from core.data_loader import (
    AVAILABLE,
    BATTERY,
    CHARGER,
    EXPORT_EFFICIENCY,
    LOCATION,
    MIN_V2G_SOC,
    RATE,
    SLOT_MINUTES,
    SOC_END,
    SOC_START,
    STATION_ID,
    VEHICLE,
)


CHARGE_EFFICIENCY = 0.90
OFF_PEAK = "Heures creuses"
PEAK = "Pointe"
OFF_PEAK_FACTOR = 0.70
EPSILON_KWH = 1e-9


@dataclass(frozen=True)
class VehicleProfile:
    model: str
    battery_capacity_kwh: float
    mean_start_soc: float
    mean_end_soc: float
    is_v2g_compatible: bool


@dataclass(frozen=True)
class Request:
    profile: VehicleProfile
    battery_capacity_kwh: float
    soc_start: float
    soc_target: float
    arrival_at: datetime
    departure_at: datetime
    v2g_consent: bool


@dataclass(frozen=True)
class Slot:
    start_at: datetime
    end_at: datetime
    period: str
    import_price: float
    export_price: float

    @property
    def hours(self) -> float:
        return (self.end_at - self.start_at).total_seconds() / 3600


@dataclass(frozen=True)
class Analysis:
    request: Request
    station_id: str
    location: str
    charger_type: str
    charge_power_kw: float
    export_power_kw: float
    historical_price_usd_per_kwh: float
    slots: tuple[Slot, ...]


@dataclass(frozen=True)
class Plan:
    analysis: Analysis
    grid_energy_kwh: tuple[float, ...]
    required_battery_energy_kwh: float
    imported_kwh: float
    exported_kwh: float
    charging_cost_usd: float
    export_revenue_usd: float
    baseline_cost_usd: float
    departure_soc: float

    @property
    def target_reached(self) -> bool:
        return self.departure_soc + 1e-9 >= self.analysis.request.soc_target

    @property
    def charge_duration_hours(self) -> float:
        return self.imported_kwh / self.analysis.charge_power_kw

    @property
    def net_cost_usd(self) -> float:
        return self.charging_cost_usd - self.export_revenue_usd

    @property
    def savings_usd(self) -> float:
        return self.baseline_cost_usd - self.net_cost_usd


def get_vehicle_profile(history: pd.DataFrame, vehicle_model: str) -> VehicleProfile:
    model = vehicle_model.strip().casefold()
    observations = history.loc[history[VEHICLE].str.casefold().eq(model)]
    if observations.empty:
        raise ValueError(f"Modèle absent du CSV ({vehicle_model})")
    return VehicleProfile(
        model=str(observations[VEHICLE].iloc[0]),
        battery_capacity_kwh=float(observations[BATTERY].mean()),
        mean_start_soc=float(observations[SOC_START].mean()),
        mean_end_soc=float(observations[SOC_END].mean()),
        is_v2g_compatible=bool(observations["is_v2g_compatible"].iloc[0]),
    )


def collecter(
    history: pd.DataFrame,
    vehicle_model: str,
    arrival_at: datetime,
    departure_at: datetime,
    *,
    battery_capacity_kwh: float | None = None,
    soc_start: float | None = None,
    soc_target: float | None = None,
    v2g_consent: bool = False,
) -> Request:
    profile = get_vehicle_profile(history, vehicle_model)
    capacity = profile.battery_capacity_kwh if battery_capacity_kwh is None else battery_capacity_kwh
    start = profile.mean_start_soc if soc_start is None else soc_start
    target = profile.mean_end_soc if soc_target is None else soc_target
    if not isfinite(capacity) or capacity <= 0:
        raise ValueError("La capacité de la batterie doit être strictement positive.")
    if not (0 <= start <= 100 and 0 <= target <= 100):
        raise ValueError("Les niveaux de batterie doivent être compris entre 0 et 100 %.")
    if departure_at <= arrival_at:
        raise ValueError("Le départ doit être postérieur à l'arrivée.")
    return Request(profile, capacity, start, target, arrival_at, departure_at, v2g_consent)


def _make_slots(request: Request, historical_price: float) -> tuple[Slot, ...]:
    slots = []
    start = request.arrival_at
    while start < request.departure_at:
        boundary = start.replace(minute=start.minute // SLOT_MINUTES * SLOT_MINUTES, second=0, microsecond=0)
        end = min(boundary + timedelta(minutes=SLOT_MINUTES), request.departure_at)
        if start.hour < 6 or start.hour >= 22:
            period, import_factor, export_factor = OFF_PEAK, OFF_PEAK_FACTOR, 0.0
        elif start.hour >= 18:
            period, import_factor, export_factor = PEAK, 1.45, 1.25
        else:
            period, import_factor, export_factor = "Heures pleines", 1.0, 0.0
        slots.append(Slot(start, end, period, historical_price * import_factor, historical_price * export_factor))
        start = end
    return tuple(slots)


def analyser(request: Request, station: pd.Series) -> Analysis:
    if station["status"] != AVAILABLE:
        raise ValueError("La borne sélectionnée est occupée dans la simulation.")
    price = float(station["price_usd_per_kwh"])
    can_export = request.profile.is_v2g_compatible and request.v2g_consent
    return Analysis(
        request=request,
        station_id=str(station[STATION_ID]),
        location=str(station[LOCATION]),
        charger_type=str(station[CHARGER]),
        charge_power_kw=float(station[RATE]),
        export_power_kw=float(station["export_limit_kw"]) if can_export else 0.0,
        historical_price_usd_per_kwh=price,
        slots=_make_slots(request, price),
    )


def _allocate_charge(analysis: Analysis, grid_energy_needed: float, order: list[int]) -> list[float]:
    allocation = [0.0] * len(analysis.slots)
    for index in order:
        if grid_energy_needed <= EPSILON_KWH:
            break
        available_kwh = analysis.charge_power_kw * analysis.slots[index].hours
        allocation[index] = min(grid_energy_needed, available_kwh)
        grid_energy_needed -= allocation[index]
    return allocation


def optimiser(analysis: Analysis) -> Plan:
    request = analysis.request
    slots = analysis.slots
    capacity = request.battery_capacity_kwh
    initial = request.soc_start / 100 * capacity
    target = request.soc_target / 100 * capacity
    required = max(target - initial, 0.0)
    floor = MIN_V2G_SOC / 100 * capacity

    cheapest_first = sorted(
        range(len(slots)), key=lambda index: (slots[index].import_price, slots[index].period != OFF_PEAK)
    )
    allocation = _allocate_charge(analysis, required / CHARGE_EFFICIENCY, cheapest_first)
    baseline_cost = sum(energy * slot.import_price for energy, slot in zip(allocation, slots))

    selected_peak = None
    selected_export = 0.0
    best_cost = baseline_cost
    later_off_peak_kwh = 0.0
    if analysis.export_power_kw > 0 and initial > floor:
        for index in reversed(range(len(slots))):
            slot = slots[index]
            if slot.period == OFF_PEAK:
                later_off_peak_kwh += analysis.charge_power_kw * slot.hours * CHARGE_EFFICIENCY
            if slot.period != PEAK or later_off_peak_kwh <= EPSILON_KWH:
                continue
            battery_export = min(
                initial - floor,
                analysis.export_power_kw * slot.hours / EXPORT_EFFICIENCY,
                initial + later_off_peak_kwh - target,
            )
            if battery_export <= EPSILON_KWH:
                continue
            grid_export = battery_export * EXPORT_EFFICIENCY
            recharge = max(target - initial + battery_export, 0.0) / CHARGE_EFFICIENCY
            off_peak_price = analysis.historical_price_usd_per_kwh * OFF_PEAK_FACTOR
            net_cost = recharge * off_peak_price - grid_export * slot.export_price
            if net_cost < best_cost:
                selected_peak, selected_export, best_cost = index, grid_export, net_cost

    if selected_peak is not None:
        after_peak = [index for index in range(selected_peak + 1, len(slots)) if slots[index].period == OFF_PEAK]
        recharge = max(target - initial + selected_export / EXPORT_EFFICIENCY, 0.0) / CHARGE_EFFICIENCY
        allocation = _allocate_charge(analysis, recharge, after_peak)
        allocation[selected_peak] = -selected_export

    imported = sum(energy for energy in allocation if energy > 0)
    exported = sum(-energy for energy in allocation if energy < 0)
    charging_cost = sum(energy * slot.import_price for energy, slot in zip(allocation, slots) if energy > 0)
    revenue = exported * slots[selected_peak].export_price if selected_peak is not None else 0.0
    departure_soc = 100 * (initial + imported * CHARGE_EFFICIENCY - exported / EXPORT_EFFICIENCY) / capacity
    return Plan(
        analysis, tuple(allocation), required, imported, exported, charging_cost, revenue, baseline_cost, departure_soc
    )


def piloter(plan: Plan) -> pd.DataFrame:
    """Retourne la puissance moyenne et le SoC de fin pour chaque créneau."""
    request = plan.analysis.request
    battery_kwh = request.soc_start / 100 * request.battery_capacity_kwh
    rows = []
    for slot, energy in zip(plan.analysis.slots, plan.grid_energy_kwh):
        if energy > 0:
            state = "CHARGE_V1G"
            battery_kwh += energy * CHARGE_EFFICIENCY
        elif energy < 0:
            state = "DECHARGE_V2G"
            battery_kwh += energy / EXPORT_EFFICIENCY
        else:
            state = "VEILLE"
        rows.append(
            {
                "start_at": slot.start_at,
                "end_at": slot.end_at,
                "state": state,
                "period": slot.period,
                "power_kw": energy / slot.hours,
                "soc_percent": 100 * battery_kwh / request.battery_capacity_kwh,
                "slot_cost_usd": energy * (slot.import_price if energy >= 0 else slot.export_price),
            }
        )
    return pd.DataFrame(rows)
