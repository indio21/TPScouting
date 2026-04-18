"""Generador de datos sinteticos para la base de entrenamiento.

Este script crea jugadores juveniles, sintetiza su trayectoria tecnica y,
a partir de esa evolucion, deriva su historial de rendimiento.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import math
import os
import random
from typing import Dict, List

from sqlalchemy.orm import sessionmaker

from db_utils import create_app_engine, ensure_player_columns, normalize_db_url
from models import (
    Base,
    Match,
    Player,
    PlayerAttributeHistory,
    PlayerMatchParticipation,
    PlayerStat,
    ScoutReport,
)
from player_logic import (
    ATTRIBUTE_FIELDS,
    EVAL_MAX_AGE,
    EVAL_MIN_AGE,
    default_player_photo_url,
    normalized_position,
    position_weights,
    recommend_position_from_attrs,
    weighted_score_from_attrs,
)

used_identifiers = set()
DEFAULT_SEED = int(os.environ.get("SEED", "42"))
OPPONENT_NAMES = [
    "Racing Juvenil",
    "Talleres Formativo",
    "Belgrano Inferiores",
    "Instituto Juvenil",
    "Newell's Academy",
    "Lanus Proyeccion",
    "Banfield Formativo",
    "Velez Desarrollo",
]
TOURNAMENTS = [
    "Liga Juvenil Regional",
    "Torneo Formativo Metropolitano",
    "Copa Proyeccion",
    "Liga de Desarrollo",
]
COMPETITION_CATEGORIES = [
    "Sub-13",
    "Sub-15",
    "Sub-17",
    "Reserva juvenil",
]


def next_identifier() -> str:
    while True:
        value = random.randint(10000000, 59999999)
        if value not in used_identifiers:
            used_identifiers.add(value)
            return str(value)


def load_existing_identifiers(session) -> None:
    existing_ids = session.query(Player.national_id).filter(Player.national_id.isnot(None)).all()
    used_identifiers.update(
        int(national_id)
        for (national_id,) in existing_ids
        if national_id is not None and str(national_id).isdigit()
    )


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def player_attr_map(player: Player) -> Dict[str, int]:
    return {field: int(getattr(player, field) or 0) for field in ATTRIBUTE_FIELDS}


def age_potential_bonus(age: int) -> float:
    return round((EVAL_MAX_AGE - age) * 0.18, 2)


def mental_bonus(position: str, attrs: Dict[str, int]) -> float:
    determination = float(attrs["determination"])
    technique = float(attrs["technique"])
    vision = float(attrs["vision"])
    if position == "Mediocampista":
        return 0.35 * vision + 0.30 * technique + 0.20 * determination
    if position == "Delantero":
        return 0.20 * vision + 0.35 * technique + 0.25 * determination
    if position == "Portero":
        return 0.15 * vision + 0.20 * technique + 0.30 * determination
    return 0.18 * vision + 0.22 * technique + 0.28 * determination


def label_probability(position: str, age: int, attrs: Dict[str, int]) -> tuple[float, float]:
    weighted_score = weighted_score_from_attrs(attrs, position)
    recommended_position, recommended_score = recommend_position_from_attrs(attrs)
    alignment_bonus = 0.55 if recommended_position == position else -0.20
    mental_component = mental_bonus(position, attrs) / 20.0
    youth_bonus = age_potential_bonus(age)
    controlled_noise = random.gauss(0.0, 0.35)

    latent_score = (
        weighted_score
        + alignment_bonus
        + youth_bonus
        + mental_component
        + controlled_noise
        + (recommended_score - weighted_score) * 0.15
    )
    probability = 1.0 / (1.0 + math.exp(-(latent_score - 13.65) / 0.90))
    return max(0.01, min(probability, 0.99)), weighted_score


def generate_player(min_age: int = EVAL_MIN_AGE, max_age: int = EVAL_MAX_AGE) -> Player:
    """Genera un jugador con atributos aleatorios y etiqueta sintetica."""
    names = [
        "Juan", "Pedro", "Carlos", "Mateo", "Lucas", "Gabriel",
        "Nicolas", "Diego", "Federico", "Martin", "Rodrigo", "Franco",
        "Facundo", "Santiago", "Tomas", "Julian", "Pablo", "Ignacio",
        "Marcelo", "Hernan",
    ]
    surnames = [
        "Garcia", "Lopez", "Martinez", "Rodriguez", "Gonzalez", "Perez",
        "Sanchez", "Romero", "Ferreira", "Suarez", "Herrera", "Ramirez",
        "Flores", "Torres", "Luna", "Alvarez", "Rojas", "Bautista",
        "Cordoba", "Vega",
    ]
    positions = ["Portero", "Defensa", "Lateral", "Mediocampista", "Delantero"]
    clubs = [
        "Club A", "Club B", "Club C", "Club D", "Club E", "Club F",
        "Academia Juvenil", "Escuela de Futbol",
    ]
    countries = ["Argentina", "Brasil", "Uruguay", "Chile", "Paraguay"]

    name = f"{random.choice(names)} {random.choice(surnames)}"
    age = random.randint(min_age, max_age)
    position = normalized_position(random.choice(positions))
    club = random.choice(clubs)
    country = random.choice(countries)

    attrs = {field: random.randint(0, 20) for field in ATTRIBUTE_FIELDS}
    probability, _weighted_score = label_probability(position, age, attrs)
    potential = random.random() < probability

    return Player(
        name=name,
        national_id=next_identifier(),
        age=age,
        position=position,
        club=club,
        country=country,
        photo_url=default_player_photo_url(name=name),
        pace=attrs["pace"],
        shooting=attrs["shooting"],
        passing=attrs["passing"],
        dribbling=attrs["dribbling"],
        defending=attrs["defending"],
        physical=attrs["physical"],
        vision=attrs["vision"],
        tackling=attrs["tackling"],
        determination=attrs["determination"],
        technique=attrs["technique"],
        potential_label=bool(potential),
    )


def build_development_profile(player: Player) -> Dict[str, float]:
    attrs = player_attr_map(player)
    age = int(player.age or EVAL_MAX_AGE)
    resilience = clamp(
        0.45 + attrs["determination"] / 26.0 + attrs["physical"] / 90.0 + random.uniform(-0.12, 0.12),
        0.45,
        1.45,
    )
    tactical_discipline = clamp(
        0.40 + attrs["vision"] / 38.0 + attrs["passing"] / 80.0 + attrs["defending"] / 95.0 + random.uniform(-0.10, 0.10),
        0.35,
        1.40,
    )
    adaptability_bias = clamp(
        0.35 + attrs["technique"] / 44.0 + attrs["vision"] / 70.0 + attrs["pace"] / 110.0 + random.uniform(-0.10, 0.10),
        0.30,
        1.35,
    )
    professionalism = clamp(
        0.42 + attrs["determination"] / 34.0 + attrs["technique"] / 75.0 + random.uniform(-0.10, 0.10),
        0.35,
        1.35,
    )
    growth_factor = clamp(
        0.55
        + (EVAL_MAX_AGE - age) * 0.12
        + attrs["determination"] / 24.0
        + attrs["technique"] / 42.0
        + attrs["vision"] / 48.0,
        0.65,
        2.8,
    )
    volatility = clamp(
        0.10 + (20 - attrs["determination"]) / 110.0 + random.uniform(0.0, 0.12),
        0.08,
        0.42,
    )
    availability = clamp(
        0.66 + attrs["physical"] / 55.0 + attrs["determination"] / 95.0 + random.uniform(-0.06, 0.06),
        0.55,
        0.97,
    )
    snapshot_count = random.randint(6, 12)
    slump_month = random.randint(2, snapshot_count - 1) if snapshot_count >= 4 and random.random() < 0.42 else None
    return {
        "growth_factor": growth_factor * ((0.88 + professionalism * 0.10) * (0.92 + resilience * 0.08)),
        "volatility": volatility,
        "availability": availability,
        "snapshot_count": snapshot_count,
        "slump_month": slump_month or 0,
        "form_bias": random.gauss(0.0, 0.18),
        "resilience": resilience,
        "tactical_discipline": tactical_discipline,
        "adaptability_bias": adaptability_bias,
        "professionalism": professionalism,
    }


def total_growth_by_attribute(player: Player, profile: Dict[str, float]) -> Dict[str, float]:
    weights = position_weights(player.position)
    current_attrs = player_attr_map(player)
    growth_map: Dict[str, float] = {}
    for field in ATTRIBUTE_FIELDS:
        current_value = float(current_attrs[field])
        ceiling_room = max(0.10, (20.0 - current_value) / 20.0)
        weighted_factor = 0.55 + (weights[field] * 2.0)
        trait_multiplier = 1.0
        if field in ("determination", "technique"):
            trait_multiplier *= 0.92 + profile["professionalism"] * 0.10
        if field in ("vision", "passing"):
            trait_multiplier *= 0.90 + profile["tactical_discipline"] * 0.12
        if field in ("pace", "dribbling"):
            trait_multiplier *= 0.90 + profile["adaptability_bias"] * 0.10
        if field in ("physical", "tackling", "defending"):
            trait_multiplier *= 0.90 + profile["resilience"] * 0.10
        random_adjustment = random.uniform(0.85, 1.25)
        growth = profile["growth_factor"] * weighted_factor * ceiling_room * trait_multiplier * random_adjustment
        growth_map[field] = clamp(growth, 0.0, 5.5)
    return growth_map


def avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def adjacent_positions(position: str) -> List[str]:
    mapping = {
        "Portero": ["Portero"],
        "Defensa": ["Defensa", "Lateral"],
        "Lateral": ["Lateral", "Defensa", "Mediocampista"],
        "Mediocampista": ["Mediocampista", "Lateral", "Delantero"],
        "Delantero": ["Delantero", "Mediocampista"],
    }
    return mapping.get(position, [position])


def synthetic_attribute_history(
    player: Player,
    profile: Dict[str, float],
) -> List[PlayerAttributeHistory]:
    current_attrs = player_attr_map(player)
    growth_map = total_growth_by_attribute(player, profile)
    snapshot_count = int(profile["snapshot_count"])
    slump_month = int(profile["slump_month"])
    history: List[PlayerAttributeHistory] = []
    today = date.today()

    for months_back in range(snapshot_count, 0, -1):
        remaining_fraction = months_back / float(snapshot_count + 1)
        wave = math.sin(((snapshot_count - months_back + 1) / float(snapshot_count + 1)) * math.pi)
        values: Dict[str, int] = {}
        for field in ATTRIBUTE_FIELDS:
            current_value = float(current_attrs[field])
            weight = position_weights(player.position)[field]
            progression_component = growth_map[field] * remaining_fraction
            curve_component = growth_map[field] * 0.10 * wave
            noise = random.gauss(0.0, profile["volatility"] * (0.30 + weight))
            slump = 0.0
            if slump_month and months_back == slump_month:
                slump = random.uniform(-0.9, -0.3) * (0.30 + weight) * (1.18 - profile["resilience"] * 0.14)
            growth_spurt = 0.0
            if player.age <= 14 and months_back in (1, 2) and random.random() < 0.35:
                growth_spurt = random.uniform(0.10, 0.35) * (0.25 + weight) * profile["professionalism"]
            historical_value = current_value - progression_component + curve_component + noise + slump
            values[field] = int(round(clamp(historical_value + growth_spurt, 0.0, current_value)))

        history.append(
            PlayerAttributeHistory(
                record_date=today - timedelta(days=30 * months_back),
                notes="Dato sintetico de trayectoria tecnica",
                **values,
            )
        )
    return history


def synthetic_matches_and_stats(
    player: Player,
    attribute_history: List[PlayerAttributeHistory],
    profile: Dict[str, float],
) -> tuple[List[Match], List[PlayerMatchParticipation], List[PlayerStat]]:
    if not attribute_history:
        return [], [], []

    matches: List[Match] = []
    participations: List[PlayerMatchParticipation] = []
    stats: List[PlayerStat] = []
    previous_weighted_score = None

    for entry in attribute_history:
        attrs = {field: float(getattr(entry, field) or 0) for field in ATTRIBUTE_FIELDS}
        weighted_score = weighted_score_from_attrs(attrs, player.position)
        avg_attr_score = sum(attrs.values()) / len(attrs)
        momentum = 0.0 if previous_weighted_score is None else weighted_score - previous_weighted_score
        previous_weighted_score = weighted_score

        if random.random() > profile["availability"]:
            continue

        monthly_match_count = random.randint(1, 3)
        monthly_participations: List[PlayerMatchParticipation] = []
        alternative_positions = [value for value in adjacent_positions(player.position) if value != player.position]

        for match_idx in range(monthly_match_count):
            opponent_level = random.choices([1, 2, 3, 4, 5], weights=[0.10, 0.24, 0.32, 0.22, 0.12], k=1)[0]
            venue = "Local" if random.random() < 0.56 else "Visitante"
            pressure = (opponent_level - 3) * 0.22 + (-0.10 if venue == "Visitante" else 0.06)
            pressure -= (profile["resilience"] - 0.85) * 0.18
            pressure -= (profile["tactical_discipline"] - 0.80) * 0.10
            starter_probability = clamp(
                0.24
                + weighted_score / 28.0
                + momentum / 6.0
                + profile["availability"] * 0.10
                + (profile["professionalism"] - 0.85) * 0.08
                - max(0, opponent_level - 3) * 0.05,
                0.10,
                0.93,
            )
            is_starter = random.random() < starter_probability
            if alternative_positions and random.random() > (0.82 - profile["adaptability_bias"] * 0.12):
                position_played = random.choice(alternative_positions)
            else:
                position_played = player.position

            minutes_base = random.randint(62, 90) if is_starter else random.randint(12, 38)
            fatigue_penalty = max(
                0.0,
                (1.0 - profile["availability"]) * random.uniform(0.0, 1.5) * (1.12 - profile["resilience"] * 0.10),
            )
            minutes_played = int(round(clamp(minutes_base - fatigue_penalty * 8, 5, 90)))
            pass_accuracy = clamp(
                (attrs["passing"] * 0.40 + attrs["vision"] * 0.32 + attrs["technique"] * 0.28) * 5
                + momentum * 3.0
                - pressure * 4.5
                + (profile["tactical_discipline"] - 0.85) * 6.0
                + random.gauss(0.0, 5.0),
                35.0,
                96.0,
            )
            shot_accuracy = clamp(
                (attrs["shooting"] * 0.58 + attrs["technique"] * 0.24 + attrs["vision"] * 0.18) * 5
                + momentum * 2.4
                - pressure * 3.4
                + (profile["professionalism"] - 0.85) * 4.0
                + random.gauss(0.0, 6.5),
                18.0,
                92.0,
            )
            duels_won_pct = clamp(
                (attrs["defending"] * 0.40 + attrs["physical"] * 0.34 + attrs["tackling"] * 0.26) * 5
                + momentum * 2.8
                - pressure * 4.0
                + (profile["resilience"] - 0.85) * 5.5
                + random.gauss(0.0, 6.0),
                25.0,
                95.0,
            )
            final_score = clamp(
                (weighted_score / 20.0) * 5.5
                + (avg_attr_score / 20.0) * 1.8
                + momentum * 0.55
                - pressure
                + profile["form_bias"]
                + (0.18 if is_starter else -0.12)
                + (profile["professionalism"] - 0.85) * 0.35
                + (profile["adaptability_bias"] - 0.80) * (0.25 if position_played != player.position else 0.08)
                + random.gauss(0.0, 0.45),
                1.0,
                10.0,
            )

            scoring_chance = max(0.0, (final_score - 5.5) / 5.0)
            goals = 0
            assists = 0
            if player.position == "Delantero":
                goals = 1 if random.random() < scoring_chance * 0.65 else 0
                assists = 1 if random.random() < scoring_chance * 0.25 else 0
            elif player.position == "Mediocampista":
                goals = 1 if random.random() < scoring_chance * 0.25 else 0
                assists = 1 if random.random() < scoring_chance * 0.45 else 0
            elif player.position == "Lateral":
                assists = 1 if random.random() < scoring_chance * 0.30 else 0

            match = Match(
                match_date=entry.record_date + timedelta(days=min(27, match_idx * 7 + random.randint(0, 5))),
                opponent_name=random.choice(OPPONENT_NAMES),
                opponent_level=opponent_level,
                tournament=random.choice(TOURNAMENTS),
                competition_category=random.choice(COMPETITION_CATEGORIES),
                venue=venue,
                notes="Dato sintetico de contexto de partido",
            )
            participation = PlayerMatchParticipation(
                player_id=player.id,
                match=match,
                started=is_starter,
                position_played=position_played,
                minutes_played=minutes_played,
                final_score=round(final_score, 2),
                goals=goals,
                assists=assists,
                pass_accuracy=round(pass_accuracy, 2),
                shot_accuracy=round(shot_accuracy, 2),
                duels_won_pct=round(duels_won_pct, 2),
                yellow_cards=random.randint(0, 1 if minutes_played < 45 else 2),
                red_cards=1 if random.random() < 0.012 else 0,
                role_notes="Dato sintetico de participacion por partido",
            )
            matches.append(match)
            participations.append(participation)
            monthly_participations.append(participation)

        if not monthly_participations:
            continue

        stats.append(
            PlayerStat(
                player_id=player.id,
                record_date=entry.record_date,
                matches_played=len(monthly_participations),
                goals=sum(item.goals for item in monthly_participations),
                assists=sum(item.assists for item in monthly_participations),
                minutes_played=sum(item.minutes_played for item in monthly_participations),
                yellow_cards=sum(item.yellow_cards for item in monthly_participations),
                red_cards=sum(item.red_cards for item in monthly_participations),
                pass_accuracy=round(avg([float(item.pass_accuracy or 0.0) for item in monthly_participations]), 2),
                shot_accuracy=round(avg([float(item.shot_accuracy or 0.0) for item in monthly_participations]), 2),
                duels_won_pct=round(avg([float(item.duels_won_pct or 0.0) for item in monthly_participations]), 2),
                final_score=round(avg([float(item.final_score or 0.0) for item in monthly_participations]), 2),
                notes="Dato sintetico agregado desde participacion por partido",
            )
        )

    return matches, participations, stats


def synthetic_scout_reports(
    player: Player,
    attribute_history: List[PlayerAttributeHistory],
    stats: List[PlayerStat],
    profile: Dict[str, float],
) -> List[ScoutReport]:
    if not attribute_history:
        return []

    report_stride = random.randint(2, 3)
    stats_by_date = {stat.record_date: stat for stat in stats}
    reports: List[ScoutReport] = []
    previous_weighted_score = None

    for idx, entry in enumerate(attribute_history):
        if idx != len(attribute_history) - 1 and ((idx + 1) % report_stride != 0):
            continue

        attrs = {field: float(getattr(entry, field) or 0) for field in ATTRIBUTE_FIELDS}
        weighted_score = weighted_score_from_attrs(attrs, player.position)
        recent_stat = stats_by_date.get(entry.record_date)
        recent_final_score = float(recent_stat.final_score) if recent_stat and recent_stat.final_score is not None else 6.0
        trend = 0.0 if previous_weighted_score is None else weighted_score - previous_weighted_score
        previous_weighted_score = weighted_score

        if player.position in ("Defensa", "Lateral"):
            tactical_base = attrs["defending"] * 0.28 + attrs["tackling"] * 0.28 + attrs["vision"] * 0.20
        elif player.position == "Mediocampista":
            tactical_base = attrs["vision"] * 0.30 + attrs["passing"] * 0.24 + attrs["technique"] * 0.20
        elif player.position == "Delantero":
            tactical_base = attrs["vision"] * 0.24 + attrs["shooting"] * 0.24 + attrs["technique"] * 0.20
        else:
            tactical_base = attrs["vision"] * 0.20 + attrs["physical"] * 0.18 + attrs["technique"] * 0.18

        decision_making = int(
            round(
                clamp(
                    attrs["vision"] * 0.34
                    + attrs["technique"] * 0.26
                    + attrs["determination"] * 0.22
                    + recent_final_score * 0.45
                    + random.gauss(0.0, 1.0),
                    0.0,
                    20.0,
                )
            )
        )
        tactical_reading = int(round(clamp(tactical_base + recent_final_score * 0.40 + random.gauss(0.0, 1.1), 0.0, 20.0)))
        mental_profile = int(
            round(
                clamp(
                    attrs["determination"] * 0.62
                    + attrs["physical"] * 0.16
                    + profile["availability"] * 4.0
                    + profile["resilience"] * 2.2
                    + recent_final_score * 0.32
                    + random.gauss(0.0, 1.0),
                    0.0,
                    20.0,
                )
            )
        )
        adaptability = int(
            round(
                clamp(
                    attrs["technique"] * 0.28
                    + attrs["vision"] * 0.24
                    + attrs["pace"] * 0.14
                    + attrs["physical"] * 0.14
                    + profile["adaptability_bias"] * 2.4
                    + recent_final_score * 0.28
                    + random.gauss(0.0, 1.1),
                    0.0,
                    20.0,
                )
            )
        )
        observed_projection_score = clamp(
            (weighted_score / 20.0) * 6.0
            + age_potential_bonus(player.age) * 0.30
            + trend * 0.75
            + (recent_final_score - 6.0) * 0.35
            + (profile["professionalism"] - 0.85) * 0.45
            + random.gauss(0.0, 0.35),
            1.0,
            10.0,
        )

        trend_label = "progreso sostenido" if trend >= 0.15 else ("estancado" if trend <= -0.10 else "evolucion gradual")
        reports.append(
            ScoutReport(
                player_id=player.id,
                report_date=entry.record_date,
                decision_making=decision_making,
                tactical_reading=tactical_reading,
                mental_profile=mental_profile,
                adaptability=adaptability,
                observed_projection_score=round(observed_projection_score, 2),
                notes=f"Reporte sintetico: {trend_label} con contexto de rendimiento reciente.",
            )
        )

    return reports


def main(
    num_players: int,
    db_url: str,
    seed: int = DEFAULT_SEED,
    min_age: int = EVAL_MIN_AGE,
    max_age: int = EVAL_MAX_AGE,
    reset_existing: bool = False,
) -> None:
    """Genera `num_players` jugadores y sus historiales coherentes."""
    random.seed(seed)
    used_identifiers.clear()
    normalized_db_url = normalize_db_url(db_url, base_dir=os.path.dirname(os.path.abspath(__file__)))
    engine = create_app_engine(normalized_db_url)
    if reset_existing:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    ensure_player_columns(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    load_existing_identifiers(session)

    batch_size = 500
    created = 0
    while created < num_players:
        remaining = num_players - created
        current_batch = min(batch_size, remaining)
        players_batch = [generate_player(min_age=min_age, max_age=max_age) for _ in range(current_batch)]
        session.add_all(players_batch)
        session.flush()

        attribute_history_batch: List[PlayerAttributeHistory] = []
        matches_batch: List[Match] = []
        participations_batch: List[PlayerMatchParticipation] = []
        stats_batch: List[PlayerStat] = []
        scout_reports_batch: List[ScoutReport] = []
        for player in players_batch:
            profile = build_development_profile(player)
            attribute_history = synthetic_attribute_history(player, profile)
            for entry in attribute_history:
                entry.player_id = player.id
            attribute_history_batch.extend(attribute_history)

            generated_matches, generated_participations, generated_stats = synthetic_matches_and_stats(
                player,
                attribute_history,
                profile,
            )
            generated_reports = synthetic_scout_reports(player, attribute_history, generated_stats, profile)

            matches_batch.extend(generated_matches)
            participations_batch.extend(generated_participations)
            stats_batch.extend(generated_stats)
            scout_reports_batch.extend(generated_reports)

        if attribute_history_batch:
            session.add_all(attribute_history_batch)
        if matches_batch:
            session.add_all(matches_batch)
        if participations_batch:
            session.add_all(participations_batch)
        if stats_batch:
            session.add_all(stats_batch)
        if scout_reports_batch:
            session.add_all(scout_reports_batch)
        session.commit()
        created += current_batch

    print(
        f"Se generaron {num_players} jugadores en la base de datos: {normalized_db_url} "
        f"(seed={seed}, edades={min_age}-{max_age}, reset={reset_existing})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera jugadores aleatorios")
    parser.add_argument("--num-players", type=int, default=1000, help="Numero de jugadores a crear")
    parser.add_argument("--db-url", type=str, default="sqlite:///players.db", help="URL de la base de datos")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Semilla para reproduccion de datos")
    parser.add_argument("--min-age", type=int, default=EVAL_MIN_AGE, help="Edad minima de generacion")
    parser.add_argument("--max-age", type=int, default=EVAL_MAX_AGE, help="Edad maxima de generacion")
    parser.add_argument("--reset", action="store_true", help="Recrea la base de entrenamiento antes de generar")
    args = parser.parse_args()
    main(
        args.num_players,
        args.db_url,
        seed=args.seed,
        min_age=args.min_age,
        max_age=args.max_age,
        reset_existing=args.reset,
    )
