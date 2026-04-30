"""Rutas del dashboard principal."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from types import SimpleNamespace
from typing import Dict

from flask import Blueprint, render_template, request
from sqlalchemy import func
from sqlalchemy.orm import load_only


def create_dashboard_blueprint(*, deps: SimpleNamespace) -> Blueprint:
    bp = Blueprint("dashboard", __name__)

    Session = deps.Session
    Player = deps.Player
    PlayerStat = deps.PlayerStat

    @bp.route("/dashboard")
    @deps.login_required
    def dashboard():
        period = request.args.get("period", "month")
        start_str = request.args.get("start")
        end_str = request.args.get("end")
        dashboard_cache_key = f"dashboard:{period}:{start_str}:{end_str}"
        cached_html = deps.cache_get(dashboard_cache_key)
        if cached_html is not None:
            return cached_html

        today = date.today()
        if period == "custom":
            try:
                start_date = (
                    datetime.strptime(start_str, "%Y-%m-%d").date()
                    if start_str
                    else today - timedelta(days=30)
                )
            except ValueError:
                start_date = today - timedelta(days=30)
            try:
                end_date = (
                    datetime.strptime(end_str, "%Y-%m-%d").date()
                    if end_str
                    else today
                )
            except ValueError:
                end_date = today
        else:
            end_date = today
            delta = {
                "week": 7,
                "month": 30,
                "year": 365,
            }.get(period, 30)
            start_date = end_date - timedelta(days=delta)
        if start_date > end_date:
            start_date, end_date = end_date, start_date
        start_str = start_date.isoformat()
        end_str = end_date.isoformat()

        db = Session()
        pos_rows = db.query(Player.position, func.count(Player.id)).group_by(Player.position).all()
        positions = [deps.display_position_label(row[0]) if row[0] else "Sin posicion" for row in pos_rows]
        pos_values = [row[1] for row in pos_rows]
        total_players = db.query(func.count(Player.id)).scalar() or 0
        avg_attrs = db.query(
            func.avg(Player.pace),
            func.avg(Player.shooting),
            func.avg(Player.passing),
            func.avg(Player.dribbling),
            func.avg(Player.defending),
            func.avg(Player.physical),
            func.avg(Player.vision),
            func.avg(Player.tackling),
            func.avg(Player.determination),
            func.avg(Player.technique),
        ).one()
        avg_labels = [deps.ATTRIBUTE_LABELS[field] for field in deps.ATTRIBUTE_FIELDS]
        avg_values = [round(float(value), 2) if value is not None else 0 for value in avg_attrs]

        players = (
            db.query(Player)
            .options(
                load_only(
                    Player.id,
                    Player.name,
                    Player.age,
                    Player.position,
                    Player.pace,
                    Player.shooting,
                    Player.passing,
                    Player.dribbling,
                    Player.defending,
                    Player.physical,
                    Player.vision,
                    Player.tackling,
                    Player.determination,
                    Player.technique,
                )
            )
            .all()
        )
        player_avg_rows = db.query(
            PlayerStat.player_id,
            func.avg(PlayerStat.final_score),
        ).group_by(PlayerStat.player_id).all()
        avg_score_map = {
            player_id: (float(avg_score) if avg_score is not None else None)
            for player_id, avg_score in player_avg_rows
        }
        top_potential = []
        category_counts = {
            "Alto potencial": 0,
            "Potencial medio": 0,
            "Bajo potencial": 0,
            "Sin datos": 0,
        }
        if players and deps.get_model() is not None:
            projections = deps.batch_project_players(players, avg_score_map)
            for player in players:
                projection = projections.get(player.id)
                if not projection:
                    continue
                top_potential.append(
                    {
                        "id": player.id,
                        "name": player.name,
                        "probability": projection["combined_prob"],
                        "category": projection["category"],
                    }
                )
                category_counts[projection["category"]] += 1
            top_potential.sort(key=lambda item: item["probability"], reverse=True)
            top_potential = top_potential[:10]
        else:
            category_counts["Sin datos"] = total_players

        stats_in_range = (
            db.query(PlayerStat)
            .filter(
                PlayerStat.record_date >= start_date,
                PlayerStat.record_date <= end_date,
                PlayerStat.final_score != None,
            )
            .order_by(
                PlayerStat.player_id.asc(),
                PlayerStat.record_date.asc(),
                PlayerStat.id.asc(),
            )
            .all()
        )
        player_map = {player.id: player for player in players}
        evolution_map: Dict[int, Dict[str, object]] = {}
        for stat in stats_in_range:
            entry = evolution_map.setdefault(
                stat.player_id,
                {
                    "first_score": stat.final_score,
                    "first_date": stat.record_date,
                    "last_score": stat.final_score,
                    "last_date": stat.record_date,
                },
            )
            if stat.record_date < entry["first_date"]:
                entry["first_date"] = stat.record_date
                entry["first_score"] = stat.final_score
            if stat.record_date >= entry["last_date"]:
                entry["last_date"] = stat.record_date
                entry["last_score"] = stat.final_score

        top_evolution = []
        for player_id, values in evolution_map.items():
            first = values["first_score"]
            last = values["last_score"]
            if first is None or last is None:
                continue
            delta = round(float(last - first), 2)
            top_evolution.append(
                {
                    "id": player_id,
                    "name": player_map[player_id].name if player_id in player_map else f"Jugador {player_id}",
                    "delta": delta,
                    "start": values["first_date"],
                    "end": values["last_date"],
                }
            )
        top_evolution.sort(key=lambda item: item["delta"], reverse=True)
        top_evolution = top_evolution[:10]
        final_score_avg = db.query(func.avg(PlayerStat.final_score)).filter(PlayerStat.final_score != None).scalar()
        final_score_avg = round(float(final_score_avg), 2) if final_score_avg is not None else None
        db.close()

        pot_labels = list(category_counts.keys())
        pot_values = [category_counts[label] for label in pot_labels]
        html = render_template(
            "dashboard.html",
            positions=positions,
            pos_values=pos_values,
            pot_labels=pot_labels,
            pot_values=pot_values,
            avg_labels=avg_labels,
            avg_values=avg_values,
            total_players=total_players,
            final_score_avg=final_score_avg,
            top_potential=top_potential,
            top_evolution=top_evolution,
            selected_period=period,
            start_date_str=start_str,
            end_date_str=end_str,
        )
        deps.cache_set(dashboard_cache_key, html)
        return html

    return bp
