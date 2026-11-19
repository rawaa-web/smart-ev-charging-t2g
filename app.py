"""Interface Streamlit de planification et de supervision T2G."""

from datetime import date, datetime, time, timedelta
from math import ceil

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core.data_loader import (
    AVAILABLE,
    CHARGER,
    CHARGING,
    DATA_PATH,
    ENERGY,
    EXPORTING,
    LOCATION,
    MIN_V2G_SOC,
    RATE,
    SOC_START,
    STATION_ID,
    VEHICLE,
    get_stations_status,
    load_charging_data,
)
from core.optimizer import CHARGE_EFFICIENCY, Plan, analyser, collecter, get_vehicle_profile, optimiser, piloter


GRID_EXPORT = "INJECTION V2G (-kW)"
GRID_REDUCED = "CHARGE RÉDUITE V1G"
GRID_IDLE = "Veille · branché"
PEAK_CHARGING_FACTOR = 0.50


def _classic_charge_cost(plan: Plan) -> float:
    """Chiffre une recharge immédiate jusqu'au SoC cible ou au départ."""
    remaining_kwh = plan.required_battery_energy_kwh / CHARGE_EFFICIENCY
    cost = 0.0
    for slot in plan.analysis.slots:
        energy = min(remaining_kwh, plan.analysis.charge_power_kw * slot.hours)
        cost += energy * slot.import_price
        remaining_kwh -= energy
        if remaining_kwh <= 0:
            break
    return cost


@st.cache_data(show_spinner=False)
def _load_history(modified_ns: int) -> pd.DataFrame:
    return load_charging_data()


def _battery_chart(plan: Plan, schedule: pd.DataFrame) -> go.Figure:
    request = plan.analysis.request
    slots = plan.analysis.slots
    day_start = datetime.combine(request.arrival_at.date(), time.min)
    day_end = datetime.combine((slots[-1].end_at - timedelta(microseconds=1)).date(), time.min)
    view_end = day_end + timedelta(days=1)
    figure = make_subplots(specs=[[{"secondary_y": True}]])

    day = day_start
    while day < view_end:
        for start_hour, end_hour, name, color in (
            (0, 6, "Heures creuses", "#16a34a"),
            (18, 22, "Pic de consommation", "#dc2626"),
            (22, 24, "Heures creuses", "#16a34a"),
        ):
            figure.add_vrect(
                x0=day + timedelta(hours=start_hour),
                x1=day + timedelta(hours=end_hour),
                fillcolor=color,
                opacity=0.15,
                line_width=0,
                layer="below",
                name=name,
                showlegend=day == day_start and start_hour != 22,
                exclude_empty_subplots=False,
            )
        day += timedelta(days=1)

    figure.add_trace(
        go.Scatter(
            x=[slot.start_at for slot in slots] + [slots[-1].end_at],
            y=[slot.import_price for slot in slots] + [slots[-1].import_price],
            name="Tarif d'achat",
            line={"color": "#c27715", "shape": "hv", "dash": "dot"},
            hovertemplate="Tarif %{y:.2f} $/kWh<extra></extra>",
        ),
        secondary_y=True,
    )
    figure.add_trace(
        go.Scatter(
            x=[request.arrival_at, *schedule["end_at"]],
            y=[request.soc_start, *schedule["soc_percent"]],
            name="SoC de la batterie",
            line={"color": "#2563eb", "width": 3},
            hovertemplate="SoC %{y:.1f} %<extra></extra>",
        ),
        secondary_y=False,
    )
    injection = schedule.loc[schedule["state"].eq("DECHARGE_V2G")]
    if not injection.empty:
        figure.add_trace(
            go.Scatter(
                x=injection["end_at"],
                y=injection["soc_percent"],
                customdata=injection["power_kw"],
                mode="markers",
                name="Injection V2G",
                marker={"color": "#dc2626", "size": 10, "symbol": "diamond"},
                hovertemplate="Injection %{customdata:.1f} kW · SoC %{y:.1f} %<extra></extra>",
            ),
            secondary_y=False,
        )
    if request.profile.is_v2g_compatible:
        figure.add_hline(y=MIN_V2G_SOC, line_dash="dash", line_color="#64748b", annotation_text="Seuil V2G")

    figure.update_xaxes(
        range=[day_start, view_end],
        tickformat="%Hh<br>%d/%m",
        hoverformat="%d/%m à %H:%M",
        dtick=3 * 60 * 60 * 1000 if (view_end - day_start).days <= 2 else None,
    )
    figure.update_yaxes(title_text="État de charge (%)", range=[0, 105], secondary_y=False)
    figure.update_yaxes(title_text="Tarif d'achat ($/kWh)", rangemode="tozero", showgrid=False, secondary_y=True)
    figure.update_layout(
        title="SoC et tarification horaire", hovermode="x unified", legend={"orientation": "h", "y": -0.22}, height=440
    )
    return figure


def _show_plan(plan: Plan) -> None:
    analysis = plan.analysis
    schedule = piloter(plan)
    st.subheader(f"Borne {analysis.station_id} · {analysis.location}")
    st.caption(
        f"Session active dans la supervision · {analysis.charger_type} · {analysis.charge_power_kw:.1f} kW observés"
    )
    arrival = analysis.request.arrival_at
    departure = analysis.request.departure_at
    st.write(f"Branchement du {arrival:%d/%m à %H:%M} au {departure:%d/%m à %H:%M}.")
    charge_slots = schedule.loc[schedule["state"].eq("CHARGE_V1G"), "start_at"]
    export_slots = schedule.loc[schedule["state"].eq("DECHARGE_V2G"), "start_at"]
    if not export_slots.empty:
        st.write(f"Injection V2G prévue le {export_slots.iloc[0]:%d/%m à %H:%M}.")
    if not charge_slots.empty:
        st.write(f"Début de la recharge le {charge_slots.iloc[0]:%d/%m à %H:%M}.")
    if not plan.target_reached:
        st.warning(
            f"Le créneau permet d'atteindre {plan.departure_soc:.1f} % "
            f"sur les {analysis.request.soc_target:g} % demandés."
        )

    classic_cost = _classic_charge_cost(plan)
    savings = round(classic_cost - plan.net_cost_usd, 2)
    classic, optimized, export = st.columns(3)
    classic.metric(
        "Recharge classique", f"{classic_cost:.2f} $", help="Recharge immédiate dès l'arrivée, sans injection V2G."
    )
    optimized.metric("Coût net optimisé", f"{plan.net_cost_usd:.2f} $", delta=f"{savings:+.2f} $ d'économie")
    export.metric("Énergie réinjectée", f"{plan.exported_kwh:.2f} kWh")
    st.caption(
        f"SoC au départ {plan.departure_soc:.1f} % · Durée de charge {plan.charge_duration_hours * 60:.0f} min · "
        f"Rémunération V2G {plan.export_revenue_usd:.2f} $"
    )
    st.plotly_chart(_battery_chart(plan, schedule))

    with st.expander("Plan par créneau de 15 minutes"):
        st.dataframe(
            schedule,
            hide_index=True,
            column_config={
                "start_at": "Début",
                "end_at": "Fin",
                "state": "État",
                "period": "Période tarifaire",
                "power_kw": st.column_config.NumberColumn("Puissance réseau (kW)", format="%.2f"),
                "soc_percent": st.column_config.NumberColumn("SoC en fin de créneau (%)", format="%.2f"),
                "slot_cost_usd": st.column_config.NumberColumn("Coût net (USD)", format="%.2f"),
            },
        )


def _driver_tab(history: pd.DataFrame, city_stations: pd.DataFrame, city: str) -> None:
    models = sorted(history[VEHICLE].unique())
    eligible_models = history.loc[history["is_v2g_compatible"], VEHICLE]
    index = models.index(eligible_models.iloc[0]) if not eligible_models.empty else 0
    model = st.selectbox("Véhicule", models, index=index)
    profile = get_vehicle_profile(history, model)
    if profile.is_v2g_compatible:
        st.success(
            f"**{model} · Compatible V2G.** L'injection est possible avec votre accord sur une borne bidirectionnelle."
        )
    else:
        st.info(f"**{model} · Smart Charging V1G.** Le modèle ne figure pas dans le référentiel V2G du prototype.")

    available = city_stations.loc[city_stations["status"].eq(AVAILABLE)].set_index(STATION_ID, drop=False)
    if available.empty:
        st.info("Toutes les bornes de cette ville sont occupées dans la simulation.")
        return
    selected_id = st.selectbox(
        "Borne",
        ["", *available.index],
        format_func=lambda identifier: (
            f"{identifier} · {available.at[identifier, CHARGER]} · {available.at[identifier, RATE]:.1f} kW"
            if identifier else "Recommander une borne"
        ),
    )
    sessions = st.session_state["driver_sessions"]
    recommendations = st.session_state["driver_recommendations"]
    station_id = selected_id or recommendations.get((city, model))
    saved = sessions.get((city, station_id))
    if saved is not None and saved.analysis.request.profile.model != model:
        st.warning(f"La borne {station_id} est occupée par {saved.analysis.request.profile.model}.")
        return

    today = date.today()
    with st.form(f"session_{model}"):
        battery, start, target = st.columns(3)
        capacity = battery.number_input(
            "Capacité batterie (kWh)", min_value=0.01, value=round(profile.battery_capacity_kwh, 2)
        )
        soc_start = start.number_input("SoC actuel (%)", 0.0, 100.0, round(profile.mean_start_soc, 2))
        soc_target = target.number_input("SoC cible (%)", 0.0, 100.0, round(profile.mean_end_soc, 2))
        st.caption("Horaires locaux. Heures creuses simulées de 22 h à 6 h, pointe de 18 h à 22 h.")
        arrive, leave = st.columns(2)
        arrival_date = arrive.date_input("Date d'arrivée", today)
        arrival_time = arrive.time_input("Heure d'arrivée", time(17))
        departure_date = leave.date_input("Date de départ", today + timedelta(days=1))
        departure_time = leave.time_input("Heure de départ", time(7))
        consent = st.checkbox("Autoriser l'injection V2G", disabled=not profile.is_v2g_compatible)
        submitted = st.form_submit_button("Valider la session", type="primary")

    if submitted:
        try:
            request = collecter(
                history,
                model,
                datetime.combine(arrival_date, arrival_time),
                datetime.combine(departure_date, departure_time),
                battery_capacity_kwh=capacity,
                soc_start=soc_start,
                soc_target=soc_target,
                v2g_consent=consent,
            )
            if station_id:
                plan = optimiser(analyser(request, available.loc[station_id]))
            else:
                candidates = (
                    optimiser(analyser(request, row))
                    for identifier, row in available.iterrows()
                    if (city, identifier) not in sessions
                )
                plan = min(
                    candidates,
                    key=lambda candidate: (
                        not candidate.target_reached,
                        0 if candidate.target_reached else -candidate.departure_soc,
                        candidate.net_cost_usd,
                        candidate.charge_duration_hours,
                    ),
                    default=None,
                )
                if plan is None:
                    raise ValueError("Toutes les bornes de cette ville ont déjà une session active.")
        except ValueError as exc:
            st.error(str(exc))
        else:
            sessions[(city, plan.analysis.station_id)] = plan
            if not selected_id:
                recommendations[(city, model)] = plan.analysis.station_id
            saved = plan

    if saved is not None:
        _show_plan(saved)


def _toggle_peak(city: str) -> None:
    peak_cities = st.session_state["peak_cities"]
    if city in peak_cities:
        peak_cities.remove(city)
    else:
        peak_cities.add(city)


def _station_view(city_stations: pd.DataFrame, city: str, network_peak: bool) -> pd.DataFrame:
    stations = city_stations.set_index(STATION_ID, drop=False).copy()
    stations["session_source"] = stations["connected_vehicle"].notna().map({True: "Simulation Kaggle", False: ""})
    if network_peak:
        peak_catalog = st.session_state["stations_during_peak"]
        peak_stations = peak_catalog.loc[peak_catalog[LOCATION].eq(city)].set_index(STATION_ID)
        exporters = peak_stations.index[peak_stations["status"].eq(EXPORTING)]
        charging = stations["status"].eq(CHARGING) & ~stations.index.isin(exporters)
        stations.loc[charging, "status"] = GRID_REDUCED
        stations.loc[charging, "power_kw"] *= PEAK_CHARGING_FACTOR
        stations.loc[exporters, "status"] = GRID_EXPORT
        stations.loc[exporters, "power_kw"] = peak_stations.loc[exporters, "power_kw"]

    for (session_city, identifier), plan in st.session_state["driver_sessions"].items():
        if session_city != city:
            continue
        request = plan.analysis.request
        powers = [energy / slot.hours for energy, slot in zip(plan.grid_energy_kwh, plan.analysis.slots)]
        charge_power = max(0.0, max(powers))
        export_power = min(0.0, min(powers))
        if network_peak and export_power < 0:
            status, power = GRID_EXPORT, export_power
        elif network_peak:
            status, power = GRID_REDUCED, charge_power * PEAK_CHARGING_FACTOR
        else:
            status, power = (CHARGING if charge_power > 0 else GRID_IDLE), charge_power
        stations.at[identifier, "status"] = status
        stations.at[identifier, "power_kw"] = power
        stations.at[identifier, "connected_vehicle"] = request.profile.model
        stations.at[identifier, "session_source"] = "Session conducteur"
    return stations.reset_index(drop=True)


def _operator_tab(city_history: pd.DataFrame, city_stations: pd.DataFrame, city: str) -> None:
    network_peak = city in st.session_state["peak_cities"]
    baseline = _station_view(city_stations, city, network_peak=False)
    stations = _station_view(city_stations, city, network_peak=True) if network_peak else baseline
    st.caption(
        "La supervision simule la puissance des véhicules branchés. En cas de pic, l'injection suit le plan V2G "
        "ou le scénario Kaggle, avec consentement sur borne bidirectionnelle et un SoC d'au moins 30 %. "
        "La charge des autres véhicules est réduite de moitié."
    )
    st.button(
        "Terminer le pic réseau" if network_peak else "Simuler un Pic de Demande Réseau",
        on_click=_toggle_peak,
        args=(city,),
        type="primary",
    )
    imported = float(stations["power_kw"].clip(lower=0).sum())
    exported = float(-stations["power_kw"].clip(upper=0).sum())
    net_power = imported - exported
    difference = net_power - baseline["power_kw"].sum()
    stations_count, grid, demand, export = st.columns(4)
    stations_count.metric("Bornes de la ville", len(stations))
    grid.metric(
        "Charge nette des bornes",
        f"{net_power:.1f} kW",
        delta=f"{difference:+.1f} kW" if network_peak else None,
        delta_color="inverse",
    )
    demand.metric("Puissance de charge", f"{imported:.1f} kW")
    export.metric("Injection V2G", f"{exported:.1f} kW")

    # La capacité par défaut représente la moitié de la puissance observée, arrondie aux 100 kW supérieurs.
    default_capacity = float(max(100, ceil(city_stations[RATE].sum() / 200) * 100))
    capacity = st.number_input(
        "Capacité simulée du transformateur (kW)",
        min_value=1.0,
        value=default_capacity,
        step=100.0,
        key=f"transformer_capacity_{city}",
    )
    utilization = max(0.0, net_power / capacity)
    st.progress(
        min(utilization, 1.0),
        text=f"Charge du transformateur · {net_power:.1f} / {capacity:.0f} kW · {utilization:.0%}",
    )
    if utilization > 1:
        st.warning(f"Capacité du transformateur dépassée de {net_power - capacity:.1f} kW.")

    power_chart = px.bar(
        stations,
        x=STATION_ID,
        y="power_kw",
        color="status",
        hover_data=["connected_vehicle"],
        category_orders={STATION_ID: stations[STATION_ID].tolist()},
        title="Puissance par borne",
        labels={
            STATION_ID: "Borne", "power_kw": "Puissance réseau (kW)", "status": "État", "connected_vehicle": "Véhicule"
        },
        color_discrete_map={
            CHARGING: "#2563eb", GRID_REDUCED: "#d97706", GRID_EXPORT: "#16a34a", AVAILABLE: "#94a3b8"
        },
    )
    power_chart.update_xaxes(nticks=14, tickangle=-30)
    power_chart.update_yaxes(
        range=[-max(1.0, city_stations["export_limit_kw"].max()) * 1.1, city_stations[RATE].max() * 1.1],
        zeroline=True,
        zerolinewidth=2,
    )
    power_chart.update_layout(legend={"orientation": "h", "y": -0.30})
    st.plotly_chart(power_chart)

    st.dataframe(
        stations[[STATION_ID, LOCATION, CHARGER, RATE, "status", "connected_vehicle", "power_kw", "session_source"]],
        hide_index=True,
        column_config={
            LOCATION: "Ville",
            CHARGER: "Type de borne",
            RATE: st.column_config.NumberColumn("Puissance observée (kW)", format="%.2f"),
            "status": "Statut simulé",
            "connected_vehicle": "Véhicule branché",
            "power_kw": st.column_config.NumberColumn("Puissance réseau signée (kW)", format="%.2f"),
            "session_source": "Origine de la session",
        },
    )

    st.subheader("Historique Kaggle")
    distribution, consumption = st.columns(2)
    soc_chart = px.histogram(
        city_history,
        x=SOC_START,
        nbins=20,
        title="Distribution du SoC au branchement",
        labels={SOC_START: "SoC au branchement (%)"},
    )
    soc_chart.update_layout(yaxis_title="Sessions")
    energy_by_type = city_history.groupby(CHARGER, as_index=False)[ENERGY].sum()
    energy_chart = px.bar(
        energy_by_type,
        x=CHARGER,
        y=ENERGY,
        title="Consommation par type de borne",
        labels={CHARGER: "Type de borne", ENERGY: "Énergie consommée (kWh)"},
    )
    distribution.plotly_chart(soc_chart)
    consumption.plotly_chart(energy_chart)


def main() -> None:
    st.set_page_config(page_title="Smart EV Charging & T2G", layout="wide")
    st.title("Smart EV Charging & T2G")
    try:
        modified_ns = DATA_PATH.stat().st_mtime_ns
        history = _load_history(modified_ns)
    except FileNotFoundError:
        st.error("Placez le CSV Kaggle dans data/ev_charging_patterns.csv pour ouvrir l'application.")
        st.stop()
    except (OSError, ValueError) as exc:
        st.error(f"Impossible de charger le CSV Kaggle ({exc}).")
        st.stop()

    if st.session_state.get("stations_revision") != modified_ns:
        st.session_state["stations_catalog"] = get_stations_status(history)
        st.session_state["stations_during_peak"] = get_stations_status(history, network_peak=True)
        st.session_state["driver_sessions"] = {}
        st.session_state["driver_recommendations"] = {}
        st.session_state["peak_cities"] = set()
        st.session_state["stations_revision"] = modified_ns

    city = st.selectbox("Ville", sorted(history[LOCATION].unique()))
    stations = st.session_state["stations_catalog"]
    city_stations = stations.loc[stations[LOCATION].eq(city)]
    city_history = history.loc[history[LOCATION].eq(city)]
    driver, operator = st.tabs(["Conducteur", "Station de recharge"])
    with driver:
        _driver_tab(history, city_stations, city)
    with operator:
        _operator_tab(city_history, city_stations, city)


if __name__ == "__main__":
    main()
