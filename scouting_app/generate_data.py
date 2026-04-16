"""Generador de datos sintéticos para la base de entrenamiento.

Este script crea jugadores juveniles y les asigna una etiqueta de potencial
con una logica ponderada por posicion, ajuste etario y ruido controlado.
"""

from __future__ import annotations

import argparse
import math
import os
import random

from sqlalchemy.orm import sessionmaker

from db_utils import create_app_engine, ensure_player_columns, normalize_db_url
from models import Base, Player
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

    players = [generate_player(min_age=min_age, max_age=max_age) for _ in range(num_players)]
    session.bulk_save_objects(players)
    session.commit()
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
