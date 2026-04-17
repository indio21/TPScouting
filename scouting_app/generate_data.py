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
from models import Base, Player, PlayerAttributeHistory, PlayerStat
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


def next_identifier() -> str:
    while True:
        value = random.randint(10000000, 59999999)
        if value not in used_identifiers:
            used_identifiers.add(value)
            return str(value)


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
        "growth_factor": growth_factor,
        "volatility": volatility,
        "availability": availability,
        "snapshot_count": snapshot_count,
        "slump_month": slump_month or 0,
        "form_bias": random.gauss(0.0, 0.18),
    }


def total_growth_by_attribute(player: Player, profile: Dict[str, float]) -> Dict[str, float]:
    weights = position_weights(player.position)
    current_attrs = player_attr_map(player)
    growth_map: Dict[str, float] = {}
    for field in ATTRIBUTE_FIELDS:
        current_value = float(current_attrs[field])
        ceiling_room = max(0.10, (20.0 - current_value) / 20.0)
        weighted_factor = 0.55 + (weights[field] * 2.0)
        random_adjustment = random.uniform(0.85, 1.25)
        growth = profile["growth_factor"] * weighted_factor * ceiling_room * random_adjustment
        growth_map[field] = clamp(growth, 0.0, 5.5)
    return growth_map


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
                slump = random.uniform(-0.9, -0.3) * (0.30 + weight)
            historical_value = current_value - progression_component + curve_component + noise + slump
            values[field] = int(round(clamp(historical_value, 0.0, current_value)))

        history.append(
            PlayerAttributeHistory(
                record_date=today - timedelta(days=30 * months_back),
                notes="Dato sintetico de trayectoria tecnica",
                **values,
            )
        )
    return history


def synthetic_player_stats(
    player: Player,
    attribute_history: List[PlayerAttributeHistory],
    profile: Dict[str, float],
) -> List[PlayerStat]:
    if not attribute_history:
        return []

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

        starter_probability = clamp(0.28 + weighted_score / 28.0 + momentum / 7.0, 0.12, 0.92)
        is_starter = random.random() < starter_probability
        matches_played = random.randint(1, 3)
        minutes_base = random.randint(60, 90) if is_starter else random.randint(10, 35)
        minutes_played = matches_played * minutes_base

        pass_accuracy = clamp(
            (attrs["passing"] * 0.42 + attrs["vision"] * 0.33 + attrs["technique"] * 0.25) * 5
            + momentum * 3.5
            + random.gauss(0.0, 5.5),
            35.0,
            96.0,
        )
        shot_accuracy = clamp(
            (attrs["shooting"] * 0.60 + attrs["technique"] * 0.25 + attrs["vision"] * 0.15) * 5
            + momentum * 2.5
            + random.gauss(0.0, 7.0),
            20.0,
            92.0,
        )
        duels_won_pct = clamp(
            (attrs["defending"] * 0.42 + attrs["physical"] * 0.33 + attrs["tackling"] * 0.25) * 5
            + momentum * 3.0
            + random.gauss(0.0, 6.0),
            28.0,
            95.0,
        )
        final_score = clamp(
            (weighted_score / 20.0) * 6.1
            + (avg_attr_score / 20.0) * 1.5
            + momentum * 0.55
            + profile["form_bias"]
            + random.gauss(0.0, 0.40),
            1.0,
            10.0,
        )

        goals = 0
        assists = 0
        if player.position == "Delantero":
            goals = random.randint(0, max(1, matches_played))
            assists = random.randint(0, max(0, matches_played))
        elif player.position == "Mediocampista":
            goals = random.randint(0, 2)
            assists = random.randint(0, 3)
        elif player.position == "Lateral":
            assists = random.randint(0, 2)

        stats.append(
            PlayerStat(
                record_date=entry.record_date,
                matches_played=matches_played,
                goals=goals,
                assists=assists,
                minutes_played=minutes_played,
                yellow_cards=random.randint(0, 2),
                red_cards=1 if random.random() < 0.015 else 0,
                pass_accuracy=round(pass_accuracy, 2),
                shot_accuracy=round(shot_accuracy, 2),
                duels_won_pct=round(duels_won_pct, 2),
                final_score=round(final_score, 2),
                notes="Dato sintetico derivado de trayectoria tecnica",
            )
        )
    return stats


def main(
    num_players: int,
    db_url: str,
    seed: int = DEFAULT_SEED,
    min_age: int = EVAL_MIN_AGE,
    max_age: int = EVAL_MAX_AGE,
) -> None:
    """Genera `num_players` jugadores y sus historiales coherentes."""
    random.seed(seed)
    used_identifiers.clear()
    normalized_db_url = normalize_db_url(db_url, base_dir=os.path.dirname(os.path.abspath(__file__)))
    engine = create_app_engine(normalized_db_url)
    Base.metadata.create_all(engine)
    ensure_player_columns(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    batch_size = 500
    created = 0
    while created < num_players:
        remaining = num_players - created
        current_batch = min(batch_size, remaining)
        players_batch = [generate_player(min_age=min_age, max_age=max_age) for _ in range(current_batch)]
        session.add_all(players_batch)
        session.flush()

        attribute_history_batch: List[PlayerAttributeHistory] = []
        stats_batch: List[PlayerStat] = []
        for player in players_batch:
            profile = build_development_profile(player)
            attribute_history = synthetic_attribute_history(player, profile)
            for entry in attribute_history:
                entry.player_id = player.id
            attribute_history_batch.extend(attribute_history)

            generated_stats = synthetic_player_stats(player, attribute_history, profile)
            for stat in generated_stats:
                stat.player_id = player.id
            stats_batch.extend(generated_stats)

        if attribute_history_batch:
            session.add_all(attribute_history_batch)
        if stats_batch:
            session.add_all(stats_batch)
        session.commit()
        created += current_batch

    print(
        f"Se generaron {num_players} jugadores en la base de datos: {normalized_db_url} "
        f"(seed={seed}, edades={min_age}-{max_age})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera jugadores aleatorios")
    parser.add_argument("--num-players", type=int, default=1000, help="Numero de jugadores a crear")
    parser.add_argument("--db-url", type=str, default="sqlite:///players.db", help="URL de la base de datos")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Semilla para reproduccion de datos")
    parser.add_argument("--min-age", type=int, default=EVAL_MIN_AGE, help="Edad minima de generacion")
    parser.add_argument("--max-age", type=int, default=EVAL_MAX_AGE, help="Edad maxima de generacion")
    args = parser.parse_args()
    main(args.num_players, args.db_url, seed=args.seed, min_age=args.min_age, max_age=args.max_age)
