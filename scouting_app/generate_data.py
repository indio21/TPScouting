"""Generador de datos sintéticos para la base de entrenamiento.

Este script crea jugadores juveniles y les asigna una etiqueta de potencial
con una logica ponderada por posicion, ajuste etario y ruido controlado.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import math
import os
import random

from sqlalchemy.orm import sessionmaker

from db_utils import create_app_engine, ensure_player_columns, normalize_db_url
from models import Base, Player, PlayerStat
from player_logic import (
    EVAL_MAX_AGE,
    EVAL_MIN_AGE,
    ATTRIBUTE_FIELDS,
    default_player_photo_url,
    normalized_position,
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


def age_potential_bonus(age: int) -> float:
    # Jugadores mas jovenes con buenos atributos reciben un extra moderado.
    return round((EVAL_MAX_AGE - age) * 0.18, 2)


def mental_bonus(position: str, attrs: dict) -> float:
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


def label_probability(position: str, age: int, attrs: dict) -> tuple[float, float]:
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
    """Genera un jugador con atributos aleatorios y etiqueta sintética."""
    names = [
        "Juan", "Pedro", "Carlos", "Mateo", "Lucas", "Gabriel",
        "Nicolas", "Diego", "Federico", "Martin", "Rodrigo", "Franco",
        "Facundo", "Santiago", "Tomas", "Julian", "Pablo", "Ignacio",
        "Marcelo", "Hernan"
    ]
    surnames = [
        "Garcia", "Lopez", "Martinez", "Rodriguez", "Gonzalez", "Perez",
        "Sanchez", "Romero", "Ferreira", "Suarez", "Herrera", "Ramirez",
        "Flores", "Torres", "Luna", "Alvarez", "Rojas", "Bautista",
        "Cordoba", "Vega"
    ]
    positions = ["Portero", "Defensa", "Lateral", "Mediocampista", "Delantero"]
    clubs = [
        "Club A", "Club B", "Club C", "Club D", "Club E", "Club F",
        "Academia Juvenil", "Escuela de Futbol"
    ]
    countries = ["Argentina", "Brasil", "Uruguay", "Chile", "Paraguay"]

    name = f"{random.choice(names)} {random.choice(surnames)}"
    age = random.randint(min_age, max_age)
    position = normalized_position(random.choice(positions))
    club = random.choice(clubs)
    country = random.choice(countries)

    attrs = {
        field: random.randint(0, 20)
        for field in ATTRIBUTE_FIELDS
    }

    probability, weighted_score = label_probability(position, age, attrs)
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


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def synthetic_player_stats(player: Player) -> list[PlayerStat]:
    entry_count = random.choices([0, 1, 2, 3, 4], weights=[0.12, 0.26, 0.28, 0.22, 0.12], k=1)[0]
    if entry_count == 0:
        return []

    attrs = {
        field: float(getattr(player, field) or 0)
        for field in ATTRIBUTE_FIELDS
    }
    fit_score = weighted_score_from_attrs(attrs, player.position)
    base_final = clamp((fit_score / 20.0) * 6.2 + (1.4 if player.potential_label else 0.2), 1.0, 9.8)
    base_pass = clamp((attrs["passing"] * 0.45) + (attrs["vision"] * 0.30) + (attrs["technique"] * 0.25), 2.0, 19.5)
    base_duels = clamp((attrs["defending"] * 0.45) + (attrs["physical"] * 0.30) + (attrs["tackling"] * 0.25), 2.0, 19.5)
    base_shot = clamp((attrs["shooting"] * 0.60) + (attrs["technique"] * 0.25) + (attrs["vision"] * 0.15), 2.0, 19.5)

    today = date.today()
    stats: list[PlayerStat] = []
    for idx in range(entry_count):
        progress = (idx + 1) / max(entry_count, 1)
        trend = (progress - 0.5) * (0.5 if player.potential_label else 0.2)
        final_score = clamp(base_final + trend + random.gauss(0.0, 0.45), 1.0, 10.0)
        pass_accuracy = clamp(base_pass * 5 + random.gauss(0.0, 5.5), 35.0, 96.0)
        duels_won_pct = clamp(base_duels * 5 + random.gauss(0.0, 6.0), 30.0, 94.0)
        shot_accuracy = clamp(base_shot * 5 + random.gauss(0.0, 7.0), 20.0, 92.0)
        matches_played = random.randint(1, 4)
        minutes_played = matches_played * random.randint(55, 90)
        goals = 0
        assists = 0
        if player.position == "Delantero":
            goals = random.randint(0, max(0, matches_played))
            assists = random.randint(0, max(0, matches_played - goals))
        elif player.position == "Mediocampista":
            goals = random.randint(0, 2)
            assists = random.randint(0, 3)
        elif player.position == "Lateral":
            assists = random.randint(0, 2)

        stats.append(
            PlayerStat(
                record_date=today - timedelta(days=(entry_count - idx) * 28),
                matches_played=matches_played,
                goals=goals,
                assists=assists,
                minutes_played=minutes_played,
                yellow_cards=random.randint(0, 2),
                red_cards=1 if random.random() < 0.02 else 0,
                pass_accuracy=round(pass_accuracy, 2),
                shot_accuracy=round(shot_accuracy, 2),
                duels_won_pct=round(duels_won_pct, 2),
                final_score=round(final_score, 2),
                notes="Dato sintetico de entrenamiento",
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
    """Genera `num_players` jugadores y los guarda en la base de datos."""
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
        stats_batch: list[PlayerStat] = []
        for player in players_batch:
            generated_stats = synthetic_player_stats(player)
            for stat in generated_stats:
                stat.player_id = player.id
            stats_batch.extend(generated_stats)
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
