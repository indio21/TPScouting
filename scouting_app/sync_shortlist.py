"""Sincroniza un subconjunto de jugadores desde la base de entrenamiento."""

import argparse
import os
from typing import Dict, List, Optional

from sqlalchemy.orm import Session as SQLAlchemySession, sessionmaker

from models import (
    Base,
    Match,
    PhysicalAssessment,
    Player,
    PlayerAttributeHistory,
    PlayerAvailability,
    PlayerMatchParticipation,
    PlayerStat,
    ScoutReport,
)
from db_utils import normalize_db_url, create_app_engine, ensure_player_columns
from player_logic import ATTRIBUTE_FIELDS, normalized_position, default_player_photo_url


def copy_player_data(src_player: Player, dst_player: Player) -> None:
    dst_player.name = src_player.name
    dst_player.national_id = src_player.national_id
    dst_player.age = src_player.age
    dst_player.position = normalized_position(src_player.position)
    dst_player.club = src_player.club
    dst_player.country = src_player.country
    dst_player.photo_url = src_player.photo_url or default_player_photo_url(
        name=src_player.name,
        national_id=src_player.national_id,
        fallback=str(src_player.id),
    )
    dst_player.potential_label = src_player.potential_label
    for field in ATTRIBUTE_FIELDS:
        setattr(dst_player, field, getattr(src_player, field))


def clear_player_related_data(dst_session: SQLAlchemySession, player: Player) -> None:
    old_match_ids = [participation.match_id for participation in player.match_participations]
    for model in (
        PlayerStat,
        PlayerAttributeHistory,
        PlayerMatchParticipation,
        ScoutReport,
        PhysicalAssessment,
        PlayerAvailability,
    ):
        dst_session.query(model).filter(model.player_id == player.id).delete(synchronize_session=False)
    dst_session.flush()
    for match_id in old_match_ids:
        if match_id and not dst_session.query(PlayerMatchParticipation).filter_by(match_id=match_id).first():
            dst_session.query(Match).filter_by(id=match_id).delete(synchronize_session=False)


def clear_operational_players(dst_session: SQLAlchemySession) -> None:
    for model in (
        PlayerMatchParticipation,
        Match,
        ScoutReport,
        PhysicalAssessment,
        PlayerAvailability,
        PlayerAttributeHistory,
        PlayerStat,
        Player,
    ):
        dst_session.query(model).delete(synchronize_session=False)
    dst_session.flush()


def copy_match(
    src_match: Optional[Match],
    dst_session: SQLAlchemySession,
    match_map: Dict[int, Match],
) -> Optional[Match]:
    if src_match is None:
        return None
    if src_match.id in match_map:
        return match_map[src_match.id]
    dst_match = Match(
        match_date=src_match.match_date,
        opponent_name=src_match.opponent_name,
        opponent_level=src_match.opponent_level,
        tournament=src_match.tournament,
        competition_category=src_match.competition_category,
        venue=src_match.venue,
        notes=src_match.notes,
    )
    dst_session.add(dst_match)
    match_map[src_match.id] = dst_match
    return dst_match


def copy_player_related_data(
    src_player: Player,
    dst_player: Player,
    dst_session: SQLAlchemySession,
    match_map: Dict[int, Match],
) -> None:
    clear_player_related_data(dst_session, dst_player)

    for stat in sorted(src_player.stats, key=lambda item: (item.record_date, item.id)):
        dst_session.add(
            PlayerStat(
                player=dst_player,
                record_date=stat.record_date,
                matches_played=stat.matches_played,
                goals=stat.goals,
                assists=stat.assists,
                minutes_played=stat.minutes_played,
                yellow_cards=stat.yellow_cards,
                red_cards=stat.red_cards,
                pass_accuracy=stat.pass_accuracy,
                shot_accuracy=stat.shot_accuracy,
                duels_won_pct=stat.duels_won_pct,
                final_score=stat.final_score,
                notes=stat.notes,
            )
        )

    for entry in sorted(src_player.attribute_history, key=lambda item: (item.record_date, item.id)):
        dst_session.add(
            PlayerAttributeHistory(
                player=dst_player,
                record_date=entry.record_date,
                pace=entry.pace,
                shooting=entry.shooting,
                passing=entry.passing,
                dribbling=entry.dribbling,
                defending=entry.defending,
                physical=entry.physical,
                vision=entry.vision,
                tackling=entry.tackling,
                determination=entry.determination,
                technique=entry.technique,
                notes=entry.notes,
            )
        )

    for participation in sorted(src_player.match_participations, key=lambda item: (item.match.match_date if item.match else item.id, item.id)):
        dst_match = copy_match(participation.match, dst_session, match_map)
        if dst_match is None:
            continue
        dst_session.add(
            PlayerMatchParticipation(
                player=dst_player,
                match=dst_match,
                started=participation.started,
                position_played=participation.position_played,
                minutes_played=participation.minutes_played,
                final_score=participation.final_score,
                goals=participation.goals,
                assists=participation.assists,
                pass_accuracy=participation.pass_accuracy,
                shot_accuracy=participation.shot_accuracy,
                duels_won_pct=participation.duels_won_pct,
                yellow_cards=participation.yellow_cards,
                red_cards=participation.red_cards,
                role_notes=participation.role_notes,
            )
        )

    for report in sorted(src_player.scout_reports, key=lambda item: (item.report_date, item.id)):
        dst_session.add(
            ScoutReport(
                player=dst_player,
                report_date=report.report_date,
                decision_making=report.decision_making,
                tactical_reading=report.tactical_reading,
                mental_profile=report.mental_profile,
                adaptability=report.adaptability,
                observed_projection_score=report.observed_projection_score,
                notes=report.notes,
            )
        )

    for assessment in sorted(src_player.physical_assessments, key=lambda item: (item.assessment_date, item.id)):
        dst_session.add(
            PhysicalAssessment(
                player=dst_player,
                assessment_date=assessment.assessment_date,
                height_cm=assessment.height_cm,
                weight_kg=assessment.weight_kg,
                dominant_foot=assessment.dominant_foot,
                estimated_speed=assessment.estimated_speed,
                endurance=assessment.endurance,
                in_growth_spurt=assessment.in_growth_spurt,
                notes=assessment.notes,
            )
        )

    for record in sorted(src_player.availability_records, key=lambda item: (item.record_date, item.id)):
        dst_session.add(
            PlayerAvailability(
                player=dst_player,
                record_date=record.record_date,
                availability_pct=record.availability_pct,
                fatigue_pct=record.fatigue_pct,
                training_load_pct=record.training_load_pct,
                missed_days=record.missed_days,
                injury_flag=record.injury_flag,
                notes=record.notes,
            )
        )


def sync_shortlist(src_db: str, dst_db: str, limit: int, min_age: int, max_age: int, replace: bool = False) -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_engine = create_app_engine(normalize_db_url(src_db, base_dir=base_dir))
    dst_engine = create_app_engine(normalize_db_url(dst_db, base_dir=base_dir))
    Base.metadata.create_all(src_engine)
    Base.metadata.create_all(dst_engine)
    ensure_player_columns(src_engine)
    ensure_player_columns(dst_engine)
    SrcSession = sessionmaker(bind=src_engine)
    DstSession = sessionmaker(bind=dst_engine)

    src_session = SrcSession()
    dst_session = DstSession()
    try:
        query = (
            src_session.query(Player)
            .filter(Player.age >= min_age, Player.age <= max_age)
            .order_by(Player.potential_label.desc(), Player.age.asc(), Player.determination.desc())
        )
        players: List[Player] = query.limit(limit).all()
        synced = 0
        inserted = 0
        updated = 0
        skipped = 0
        if replace:
            clear_operational_players(dst_session)
        existing_total = dst_session.query(Player).count()
        match_map: Dict[int, Match] = {}
        for src_player in players:
            if not src_player.national_id:
                skipped += 1
                continue
            existing = (
                dst_session.query(Player)
                .filter(Player.national_id == src_player.national_id)
                .one_or_none()
            )
            if existing:
                copy_player_data(src_player, existing)
                copy_player_related_data(src_player, existing, dst_session, match_map)
                updated += 1
            else:
                if existing_total >= limit:
                    skipped += 1
                    continue
                new_player = Player(
                    name=src_player.name,
                    national_id=src_player.national_id,
                    age=src_player.age,
                    position=normalized_position(src_player.position),
                    club=src_player.club,
                    country=src_player.country,
                    photo_url=src_player.photo_url or default_player_photo_url(
                        name=src_player.name,
                        national_id=src_player.national_id,
                        fallback=str(src_player.id),
                    ),
                    potential_label=src_player.potential_label,
                )
                for field in ATTRIBUTE_FIELDS:
                    setattr(new_player, field, getattr(src_player, field))
                dst_session.add(new_player)
                dst_session.flush()
                copy_player_related_data(src_player, new_player, dst_session, match_map)
                inserted += 1
                existing_total += 1
            synced += 1
        dst_session.commit()
        print(
            f"Sincronizacion completada. actualizados={updated}, insertados={inserted}, "
            f"procesados={synced}, omitidos={skipped}, total_operativo={existing_total}, "
            f"limite={limit}, rango={min_age}-{max_age}, replace={replace}."
        )
    finally:
        src_session.close()
        dst_session.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Copia jugadores juveniles a la base operativa.")
    parser.add_argument("--src-db", default="sqlite:///players_training.db", help="Base de datos origen (entrenamiento)")
    parser.add_argument("--dst-db", default="sqlite:///players_updated_v2.db", help="Base de datos destino (shortlist)")
    parser.add_argument("--limit", type=int, default=100, help="Cantidad maxima de jugadores a sincronizar")
    parser.add_argument("--min-age", type=int, default=12, help="Edad minima")
    parser.add_argument("--max-age", type=int, default=18, help="Edad maxima")
    parser.add_argument("--replace", action="store_true", help="Reemplaza jugadores y datos deportivos de la base operativa")
    args = parser.parse_args()
    if args.min_age < 10 or args.max_age < args.min_age:
        raise SystemExit("Rango de edades invalido.")
    sync_shortlist(args.src_db, args.dst_db, args.limit, args.min_age, args.max_age, replace=args.replace)


if __name__ == "__main__":
    main()
