"""Rutas de jugadores, ficha, historial y proyeccion."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for
from sqlalchemy import desc, func
from sqlalchemy.orm import load_only


def create_players_blueprint(*, deps: SimpleNamespace) -> Blueprint:
    bp = Blueprint("players", __name__)

    Session = deps.Session
    Player = deps.Player
    PlayerStat = deps.PlayerStat
    PlayerAttributeHistory = deps.PlayerAttributeHistory

    @bp.route("/players")
    @deps.login_required
    def index():
        search_term = request.args.get("q")
        pos_filter = request.args.get("position")
        club_filter = request.args.get("club")
        country_filter = request.args.get("country")
        top_potential = request.args.get("top_potential")
        order_attr = request.args.get("order_attr")
        page = request.args.get("page", 1, type=int)
        per_page = deps.PLAYER_LIST_PER_PAGE
        db = Session()
        player_list_columns = [
            Player.id,
            Player.name,
            Player.national_id,
            Player.age,
            Player.position,
            Player.club,
            Player.country,
            Player.photo_url,
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
            Player.potential_label,
        ]
        query = db.query(Player).options(load_only(*player_list_columns))
        pos_list = [row[0] for row in db.query(Player.position).distinct().all()]
        club_list = [row[0] for row in db.query(Player.club).distinct().all() if row[0]]
        country_list = [row[0] for row in db.query(Player.country).distinct().all() if row[0]]
        if search_term:
            query = query.filter(Player.name.ilike(f"%{search_term}%"))
        if pos_filter:
            query = query.filter(Player.position == pos_filter)
        if club_filter:
            query = query.filter(Player.club == club_filter)
        if country_filter:
            query = query.filter(Player.country == country_filter)
        if order_attr and hasattr(Player, order_attr):
            query = query.order_by(desc(getattr(Player, order_attr)))
        total = query.count()
        total_pages = (total + per_page - 1) // per_page
        players = query.offset((page - 1) * per_page).limit(per_page).all()
        if top_potential:
            players = query.all()
        player_ids = [player.id for player in players]
        avg_score_map: Dict[int, Optional[float]] = {}
        if player_ids:
            player_avg_rows = (
                db.query(PlayerStat.player_id, func.avg(PlayerStat.final_score))
                .filter(PlayerStat.player_id.in_(player_ids))
                .group_by(PlayerStat.player_id)
                .all()
            )
            avg_score_map = {
                player_id: (float(avg_score) if avg_score is not None else None)
                for player_id, avg_score in player_avg_rows
            }
        projections = deps.batch_project_players(players, avg_score_map)
        player_rows = []
        for player in players:
            projection = projections.get(player.id)
            if projection:
                combined_pct = projection["combined_prob"] * 100
                row = {
                    "player": player,
                    "photo_url": player.photo_url or deps.default_player_photo_url(
                        name=player.name,
                        national_id=player.national_id,
                        fallback=str(player.id),
                    ),
                    "category": projection["category"],
                    "probability": f"{combined_pct:.1f}%",
                    "prob_value": combined_pct,
                    "fit_score": projection["fit_score"],
                    "best_position": projection["recommended_position"],
                    "best_score": projection["recommended_score"],
                }
            else:
                attr_map = deps.player_attribute_map(player)
                best_position, best_score = deps.recommend_position_from_attrs(attr_map)
                fit_score = deps.weighted_score_from_attrs(attr_map, player.position)
                row = {
                    "player": player,
                    "photo_url": player.photo_url or deps.default_player_photo_url(
                        name=player.name,
                        national_id=player.national_id,
                        fallback=str(player.id),
                    ),
                    "category": "Sin datos suficientes",
                    "probability": "--",
                    "prob_value": None,
                    "fit_score": fit_score,
                    "best_position": best_position,
                    "best_score": best_score,
                }
            player_rows.append(row)

        if top_potential:
            player_rows = [
                item for item in player_rows
                if item["prob_value"] is not None and deps.is_high_potential_probability(item["prob_value"] / 100.0)
            ]
            player_rows.sort(
                key=lambda item: (item["prob_value"] is not None, item["prob_value"] or -1.0),
                reverse=True,
            )
            total = len(player_rows)
            total_pages = max(1, (total + per_page - 1) // per_page)
            page = min(max(page, 1), total_pages)
            start = (page - 1) * per_page
            end = start + per_page
            player_rows = player_rows[start:end]
        db.close()
        return render_template(
            "players.html",
            players=player_rows,
            search_term=search_term,
            pos_list=pos_list,
            club_list=club_list,
            country_list=country_list,
            pos_filter=pos_filter,
            club_filter=club_filter,
            country_filter=country_filter,
            top_potential=top_potential,
            order_attr=order_attr,
            page=page,
            total_pages=total_pages,
            total_results=total,
        )

    @bp.route("/player/<int:player_id>")
    @deps.login_required
    def player_detail(player_id: int):
        db = Session()
        player = db.query(Player).filter(Player.id == player_id).first()
        if not player:
            db.close()
            abort(404)
        history_synced = False
        if deps.can_edit_player_data():
            history_synced = deps.sync_player_attribute_history(
                player,
                db,
                note="Sincronizacion automatica al abrir ficha",
            )
        if history_synced:
            db.commit()
        stats = deps.fetch_player_stats(player_id, db_session=db)
        recent_stats = list(reversed(stats[-3:])) if stats else []
        attr_history = deps.fetch_attribute_history(player_id, db_session=db)
        recent_attributes = list(reversed(attr_history[-3:])) if attr_history else []
        attribute_summary = deps.summarize_attribute_history(attr_history)
        stats_summary = deps.summarize_stats(stats)
        db.close()
        player_photo_url = player.photo_url or deps.default_player_photo_url(
            name=player.name,
            national_id=player.national_id,
            fallback=str(player.id),
        )
        projection = deps.compute_projection(player, stats)
        attr_map = deps.player_attribute_map(player)
        best_position, best_position_score = deps.recommend_position_from_attrs(attr_map)
        current_fit = deps.weighted_score_from_attrs(attr_map, player.position)
        position_ranking = [
            {
                "position": pos,
                "score": deps.weighted_score_from_attrs(attr_map, pos),
                "is_current": deps.normalized_position(player.position) == pos,
            }
            for pos in deps.POSITION_OPTIONS
        ]
        position_ranking.sort(key=lambda item: item["score"], reverse=True)
        technical_attributes = [
            {"label": "Ritmo", "value": player.pace},
            {"label": "Disparo", "value": player.shooting},
            {"label": "Pase", "value": player.passing},
            {"label": "Regate", "value": player.dribbling},
            {"label": "Defensa", "value": player.defending},
            {"label": "Físico", "value": player.physical},
            {"label": "Visión", "value": player.vision},
            {"label": "Marcaje", "value": player.tackling},
            {"label": "Determinación", "value": player.determination},
            {"label": "Técnica", "value": player.technique},
        ]

        def build_trait(name: str, score: float, strengths: str, follow_up: str, improvement: str) -> dict:
            band = deps.score_band(score)
            messaging = {"Alto": strengths, "Medio": follow_up, "Bajo": improvement}
            return {
                "name": name,
                "score": round(score, 1),
                "band": band,
                "message": messaging[band],
            }

        psychological_profile = [
            build_trait(
                "Resiliencia competitiva",
                player.determination,
                "Sostiene el esfuerzo bajo presión; indicado para partidos decisivos.",
                "Trabajar rutinas de respiración y feedback constante para fortalecer su respuesta en escenarios adversos.",
                "Recomendar intervención del área psicológica y refuerzo en hábitos de disciplina diaria.",
            ),
            build_trait(
                "Visión táctica",
                (player.vision + player.passing) / 2,
                "Lee espacios y acelera cambios de juego, facilita la progresión del equipo.",
                "Incrementar análisis de vídeo para mejorar la toma de decisiones en el último tercio.",
                "Diseñar ejercicios de toma de decisiones en superioridad/inferioridad numérica.",
            ),
            build_trait(
                "Creatividad ofensiva",
                (player.technique + player.dribbling) / 2,
                "Desborde y control diferenciales; puede romper líneas defensivas.",
                "Trabajar gestos técnicos específicos a alta velocidad para trasladar virtudes al contexto profesional.",
                "Enfocar sesiones en conducción orientada y confianza en el uno contra uno.",
            ),
            build_trait(
                "Liderazgo comunicacional",
                (player.vision + player.determination + player.passing) / 3,
                "Influye en sus compañeros y ordena fases ofensivas.",
                "Definir responsabilidades puntuales dentro del equipo para ganar protagonismo progresivo.",
                "Establecer mentoría con referentes del plantel y dinámicas de comunicación en cancha.",
            ),
        ]

        development_focus = deps.compute_suggestions(player, threshold=15, top_n=3)

        return render_template(
            "player_detail.html",
            player=player,
            player_photo_url=player_photo_url,
            technical_attributes=technical_attributes,
            radar_labels=[attr["label"] for attr in technical_attributes],
            radar_values=[attr["value"] for attr in technical_attributes],
            psychological_profile=psychological_profile,
            development_focus=development_focus,
            recent_stats=recent_stats,
            stats_summary=stats_summary,
            history_payload=deps.stats_chart_payload(stats),
            attribute_history=recent_attributes,
            attribute_summary=attribute_summary,
            attribute_payload=deps.attribute_chart_payload(attr_history),
            attribute_labels=deps.ATTRIBUTE_LABELS,
            projection=projection,
            best_position=best_position,
            best_position_score=best_position_score,
            current_fit=current_fit,
            position_ranking=position_ranking[:3],
        )

    @bp.route("/player/<int:player_id>/stats", methods=["GET", "POST"])
    @deps.login_required
    def player_stats(player_id: int):
        if request.method == "POST":
            deps.require_csrf()
            if not deps.can_edit_player_data():
                abort(403)

        db = Session()
        player = db.query(Player).filter(Player.id == player_id).first()
        if not player:
            db.close()
            abort(404)

        if request.method == "POST":
            action = request.form.get("action", "add")
            if action == "recalculate":
                deps.refresh_player_potential(player, db)
                db.commit()
                deps.invalidate_dashboard_cache()
                db.close()
                flash("Listo: se actualizo la proyeccion con los ultimos datos.", "success")
                return redirect(url_for("predict_player", player_id=player_id))
            errors: List[str] = []
            record_date = deps.parse_date_field(request.form.get("record_date"), errors, "La fecha del registro")
            matches_played = deps.validate_non_negative_int_field(
                request.form.get("matches_played"),
                "Partidos jugados",
                errors,
            )
            goals = deps.validate_non_negative_int_field(request.form.get("goals"), "Goles", errors)
            assists = deps.validate_non_negative_int_field(request.form.get("assists"), "Asistencias", errors)
            minutes_played = deps.validate_non_negative_int_field(
                request.form.get("minutes_played"),
                "Minutos jugados",
                errors,
            )
            yellow_cards = deps.validate_non_negative_int_field(
                request.form.get("yellow_cards"),
                "Tarjetas amarillas",
                errors,
            )
            red_cards = deps.validate_non_negative_int_field(request.form.get("red_cards"), "Tarjetas rojas", errors)
            pass_accuracy = deps.validate_optional_float_range(
                request.form.get("pass_accuracy"),
                "Precision de pase",
                errors,
                0,
                100,
            )
            shot_accuracy = deps.validate_optional_float_range(
                request.form.get("shot_accuracy"),
                "Precision de remate",
                errors,
                0,
                100,
            )
            duels_won_pct = deps.validate_optional_float_range(
                request.form.get("duels_won_pct"),
                "Duelos ganados",
                errors,
                0,
                100,
            )
            final_score = deps.validate_optional_float_range(
                request.form.get("final_score"),
                "Valoracion final",
                errors,
                1,
                10,
            )
            if errors:
                for message in errors:
                    flash(message, "danger")
                stats = (
                    db.query(PlayerStat)
                    .filter(PlayerStat.player_id == player_id)
                    .order_by(PlayerStat.record_date.desc(), PlayerStat.id.desc())
                    .all()
                )
                summary = deps.summarize_stats(list(reversed(stats)))
                db.close()
                return render_template(
                    "player_stats.html",
                    player=player,
                    stats=stats,
                    summary=summary,
                )

            stat = PlayerStat(
                player_id=player_id,
                record_date=record_date,
                matches_played=matches_played,
                goals=goals,
                assists=assists,
                minutes_played=minutes_played,
                yellow_cards=yellow_cards,
                red_cards=red_cards,
                pass_accuracy=pass_accuracy,
                shot_accuracy=shot_accuracy,
                duels_won_pct=duels_won_pct,
                final_score=final_score,
                notes=request.form.get("notes") or None,
            )
            if stat.final_score is None:
                stat.final_score = deps.calculate_stats_rating(
                    {
                        "matches": stat.matches_played,
                        "goals": stat.goals,
                        "assists": stat.assists,
                        "minutes": stat.minutes_played,
                        "pass_pct": stat.pass_accuracy,
                        "shot_pct": stat.shot_accuracy,
                        "duels_pct": stat.duels_won_pct,
                    }
                )
            db.add(stat)
            db.commit()
            deps.refresh_player_potential(player, db)
            db.commit()
            deps.invalidate_dashboard_cache()
            db.close()
            flash("Listo: se agrego el registro al historial del jugador.", "success")
            return redirect(url_for("player_stats", player_id=player_id))

        stats = (
            db.query(PlayerStat)
            .filter(PlayerStat.player_id == player_id)
            .order_by(PlayerStat.record_date.desc(), PlayerStat.id.desc())
            .all()
        )
        db.close()
        summary = deps.summarize_stats(list(reversed(stats)))
        return render_template(
            "player_stats.html",
            player=player,
            stats=stats,
            summary=summary,
        )

    @bp.route("/player/<int:player_id>/attributes", methods=["GET", "POST"])
    @deps.login_required
    def player_attributes(player_id: int):
        if request.method == "POST":
            deps.require_csrf()
            if not deps.can_edit_player_data():
                abort(403)

        db = Session()
        player = db.query(Player).filter(Player.id == player_id).first()
        if not player:
            db.close()
            abort(404)

        if request.method == "POST":
            action = request.form.get("action", "add")
            if action == "recalculate":
                deps.refresh_player_potential(player, db)
                db.commit()
                deps.invalidate_dashboard_cache()
                db.close()
                flash("Listo: se actualizo la proyeccion con los nuevos atributos.", "success")
                return redirect(url_for("predict_player", player_id=player_id))
            errors: List[str] = []
            record_date = deps.parse_date_field(request.form.get("record_date"), errors, "La fecha del historial")

            entry = PlayerAttributeHistory(
                player_id=player_id,
                record_date=record_date,
                notes=request.form.get("notes") or None,
            )
            has_any_value = False
            for field in deps.ATTRIBUTE_FIELDS:
                raw = request.form.get(field)
                value = deps.parse_int_field(raw, default=-1) if raw not in (None, "") else None
                if value is not None and not deps.is_valid_attribute(value):
                    errors.append(f"{deps.ATTRIBUTE_LABELS[field]} debe estar entre 0 y 20.")
                    value = None
                setattr(entry, field, value)
                if value is not None:
                    has_any_value = True
                    setattr(player, field, value)
            if not has_any_value:
                errors.append("Debes cargar al menos un atributo para guardar el historial.")
            if errors:
                for message in errors:
                    flash(message, "danger")
                history = (
                    db.query(PlayerAttributeHistory)
                    .filter(PlayerAttributeHistory.player_id == player_id)
                    .order_by(PlayerAttributeHistory.record_date.desc(), PlayerAttributeHistory.id.desc())
                    .all()
                )
                ascending_history = list(reversed(history))
                summary = deps.summarize_attribute_history(ascending_history)
                payload = deps.attribute_chart_payload(ascending_history)
                db.close()
                return render_template(
                    "player_attributes.html",
                    player=player,
                    history=history,
                    summary=summary,
                    attribute_labels=deps.ATTRIBUTE_LABELS,
                    payload=payload,
                )
            db.add(entry)
            deps.refresh_player_potential(player, db)
            db.commit()
            deps.invalidate_dashboard_cache()
            db.close()
            flash("Listo: se guardo el historial de atributos.", "success")
            return redirect(url_for("player_attributes", player_id=player_id))

        history = (
            db.query(PlayerAttributeHistory)
            .filter(PlayerAttributeHistory.player_id == player_id)
            .order_by(PlayerAttributeHistory.record_date.desc(), PlayerAttributeHistory.id.desc())
            .all()
        )
        ascending_history = list(reversed(history))
        summary = deps.summarize_attribute_history(ascending_history)
        payload = deps.attribute_chart_payload(ascending_history)
        db.close()
        return render_template(
            "player_attributes.html",
            player=player,
            history=history,
            summary=summary,
            attribute_labels=deps.ATTRIBUTE_LABELS,
            payload=payload,
        )

    @bp.route("/edit_player/<int:player_id>", methods=["GET", "POST"])
    @deps.roles_required(deps.ROLE_ADMIN, deps.ROLE_SCOUT)
    def edit_player(player_id):
        if request.method == "POST":
            deps.require_csrf()

        db = Session()
        player = db.get(Player, player_id)
        if not player:
            db.close()
            abort(404)
        if request.method == "POST":
            errors: List[str] = []
            name = (request.form.get("name") or "").strip()
            national_id = deps.normalize_identifier(request.form.get("national_id"))
            age = deps.parse_int_field(request.form.get("age"))
            position = deps.normalize_position_choice(request.form.get("position"))
            club = (request.form.get("club") or "").strip() or None
            country = (request.form.get("country") or "").strip() or None
            photo_url = (request.form.get("photo_url") or "").strip() or None
            if not name:
                errors.append("El nombre es obligatorio.")
            if not national_id:
                errors.append("El DNI debe contener solo números.")
            else:
                repeated = (
                    db.query(Player.id)
                    .filter(Player.national_id == national_id, Player.id != player.id)
                    .first()
                )
                if repeated:
                    errors.append("El DNI ingresado pertenece a otro jugador.")
            if not deps.is_valid_eval_age(age):
                errors.append("La edad debe estar entre 12 y 18 años.")
            attr_values: Dict[str, int] = {}
            for field in deps.ATTRIBUTE_FIELDS:
                value = deps.parse_int_field(request.form.get(field), getattr(player, field))
                if not deps.is_valid_attribute(value):
                    errors.append(f"{deps.ATTRIBUTE_LABELS[field]} debe estar entre 0 y 20.")
                attr_values[field] = value
            if errors:
                for message in errors:
                    flash(message, "danger")
                db.close()
                return render_template(
                    "edit_player.html",
                    player=player,
                    form_data=request.form,
                    position_options=deps.POSITION_OPTIONS,
                )
            player.name = name
            player.national_id = national_id
            player.age = age
            player.position = position
            player.club = club
            player.country = country
            player.photo_url = photo_url or deps.default_player_photo_url(
                name=name,
                national_id=national_id,
                fallback=str(player.id),
            )
            for field, value in attr_values.items():
                setattr(player, field, value)
            player.potential_label = True if request.form.get("potential_label") == "1" else False
            deps.sync_player_attribute_history(player, db, note="Actualizacion de ficha")
            deps.refresh_player_potential(player, db)
            db.commit()
            deps.invalidate_dashboard_cache()
            db.close()
            flash("Listo: se actualizo la ficha del jugador.", "success")
            return redirect(url_for("player_detail", player_id=player_id))
        db.close()
        return render_template("edit_player.html", player=player, position_options=deps.POSITION_OPTIONS)

    @bp.route("/delete_player/<int:player_id>", methods=["POST"])
    @deps.roles_required(deps.ROLE_ADMIN, deps.ROLE_SCOUT)
    def delete_player(player_id):
        deps.require_csrf()

        db = Session()
        player = db.get(Player, player_id)
        if not player:
            db.close()
            abort(404)
        db.delete(player)
        db.commit()
        deps.invalidate_dashboard_cache()
        db.close()
        flash("Listo: se elimino el jugador del seguimiento.", "success")
        return redirect(url_for("index"))

    @bp.route("/player/<int:player_id>/predict")
    @deps.login_required
    def predict_player(player_id: int):
        if deps.get_model() is None:
            return "No hay calculo disponible todavia. Actualiza los datos desde Configuracion.", 500

        db = Session()
        player = db.query(Player).filter(Player.id == player_id).first()
        if not player:
            db.close()
            abort(404)
        if deps.can_edit_player_data():
            projection = deps.refresh_player_potential(player, db)
        else:
            projection = deps.compute_projection(player, db_session=db)
        player_view = SimpleNamespace(**player.to_dict())
        player_view.photo_url = player.photo_url or deps.default_player_photo_url(
            name=player.name,
            national_id=player.national_id,
            fallback=str(player.id),
        )
        player_view.potential_label = (
            deps.is_high_potential_probability(projection["combined_prob"])
            if projection else player.potential_label
        )
        suggestions = deps.compute_suggestions(player_view)
        attr_map = deps.player_attribute_map(player)
        best_position, best_position_score = deps.recommend_position_from_attrs(attr_map)
        current_fit = deps.weighted_score_from_attrs(attr_map, player.position)

        if not projection:
            stats_summary = deps.summarize_stats([])
            prob_base = prob_combined = 0.0
            prob_calibrated_combined = None
            category = "Sin datos suficientes"
            stats_history: List[Any] = []
        else:
            prob_base = projection["base_prob"]
            prob_combined = projection["combined_prob"]
            prob_calibrated_combined = projection.get("calibrated_combined_prob")
            category = projection["category"]
            stats_summary = projection["stats_summary"]
            stats_history = projection["history"]
        history_payload = deps.stats_chart_payload(stats_history)
        attribute_payload = deps.attribute_chart_payload(deps.fetch_attribute_history(player_id))
        if deps.can_edit_player_data():
            db.commit()
            deps.invalidate_dashboard_cache()
        else:
            db.rollback()
        db.close()

        return render_template(
            "prediction.html",
            player=player_view,
            probability=f"{prob_combined*100:.1f}%",
            probability_base=f"{prob_base*100:.1f}%",
            probability_calibrated=(
                f"{float(prob_calibrated_combined)*100:.1f}%"
                if prob_calibrated_combined is not None
                else None
            ),
            probability_delta=(prob_combined - prob_base) * 100,
            category=category,
            suggestions=suggestions,
            stats_summary=stats_summary,
            history_payload=history_payload,
            attribute_payload=attribute_payload,
            attribute_labels=deps.ATTRIBUTE_LABELS,
            best_position=best_position,
            best_position_score=best_position_score,
            current_fit=current_fit,
        )

    @bp.route("/players/manage", methods=["GET", "POST"])
    @deps.roles_required(deps.ROLE_ADMIN, deps.ROLE_SCOUT)
    def manage_players():
        if request.method == "POST":
            deps.require_csrf()

        if request.method == "POST":
            mode = request.form.get("mode", "single")
            db = Session()
            created: List[str] = []
            errors: List[str] = []
            try:
                if mode == "single":
                    current_total = db.query(func.count(Player.id)).scalar() or 0
                    if current_total >= deps.EVAL_POOL_MAX:
                        errors.append(
                            f"Se alcanzo el maximo de {deps.EVAL_POOL_MAX} jugadores evaluables. "
                            "Elimina o actualiza jugadores existentes."
                        )
                    name = (request.form.get("name") or "").strip()
                    national_id = deps.normalize_identifier(request.form.get("national_id"))
                    age = deps.parse_int_field(request.form.get("age"))
                    position = deps.normalize_position_choice(request.form.get("position"))
                    club = (request.form.get("club") or "").strip() or None
                    country = (request.form.get("country") or "").strip() or None
                    photo_url = (request.form.get("photo_url") or "").strip() or None
                    if not name:
                        errors.append("El nombre es obligatorio.")
                    if not national_id:
                        errors.append("Ingresa un DNI/ID valido (solo numeros).")
                    else:
                        repeated = (
                            db.query(Player.id)
                            .filter(Player.national_id == national_id)
                            .first()
                        )
                        if repeated:
                            errors.append("Ese DNI ya esta registrado en la base.")
                    if not deps.is_valid_eval_age(age):
                        errors.append("La edad debe estar entre 12 y 18 anos.")
                    attr_values: Dict[str, int] = {}
                    for field in deps.ATTRIBUTE_FIELDS:
                        value = deps.parse_int_field(request.form.get(field), 10)
                        if not deps.is_valid_attribute(value):
                            errors.append(f"{deps.ATTRIBUTE_LABELS[field]} debe ubicarse entre 0 y 20.")
                        attr_values[field] = value
                    if not errors:
                        player = Player(
                            name=name,
                            national_id=national_id,
                            age=age,
                            position=position,
                            club=club,
                            country=country,
                            photo_url=photo_url or deps.default_player_photo_url(name=name, national_id=national_id),
                            potential_label=deps.parse_bool(request.form.get("potential_label")),
                            **attr_values,
                        )
                        db.add(player)
                        db.flush()
                        deps.sync_player_attribute_history(player, db, note="Alta de jugador")
                        deps.refresh_player_potential(player, db)
                        db.commit()
                        deps.invalidate_dashboard_cache()
                        created.append(player.name)
                        flash(f"Listo: se agrego a {player.name} al seguimiento.", "success")
                        return redirect(url_for("manage_players"))
                elif mode == "bulk":
                    bulk_input = request.form.get("bulk_input") or ""
                    lines = [line.strip() for line in bulk_input.splitlines() if line.strip()]
                    current_total = db.query(func.count(Player.id)).scalar() or 0
                    available_slots = max(deps.EVAL_POOL_MAX - current_total, 0)
                    if available_slots <= 0:
                        errors.append(
                            f"No hay cupo disponible. Limite actual: {deps.EVAL_POOL_MAX} jugadores evaluables."
                        )
                    if not lines:
                        errors.append("Ingresa al menos una fila en la carga masiva.")
                    else:
                        seen_ids = set()
                        for idx, line in enumerate(lines, start=1):
                            if len(created) >= available_slots:
                                errors.append(
                                    f"Se alcanzo el limite de {deps.EVAL_POOL_MAX} jugadores. "
                                    "El resto de filas fue omitido."
                                )
                                break
                            parts = [part.strip() for part in line.split(",")]
                            if len(parts) < 17:
                                errors.append(f"Linea {idx}: se esperaban 17 o 18 columnas (recibido {len(parts)}).")
                                continue
                            try:
                                name = parts[0]
                                national_id = deps.normalize_identifier(parts[1])
                                age = int(parts[2])
                                position = deps.normalize_position_choice(parts[3])
                                club = parts[4] or None
                                country = parts[5] or None
                                raw_photo_url = parts[17] if len(parts) > 17 else ""
                                if not national_id:
                                    raise ValueError("DNI no valido.")
                                if national_id in seen_ids:
                                    raise ValueError("DNI repetido en la carga.")
                                exists = (
                                    db.query(Player.id)
                                    .filter(Player.national_id == national_id)
                                    .first()
                                )
                                if exists:
                                    raise ValueError("DNI ya registrado en la base.")
                                if not deps.is_valid_eval_age(age):
                                    raise ValueError("Edad fuera del rango permitido (12-18).")
                                attr_values = {}
                                for offset, field in enumerate(deps.ATTRIBUTE_FIELDS, start=6):
                                    value = int(parts[offset])
                                    if not deps.is_valid_attribute(value):
                                        raise ValueError(f"{deps.ATTRIBUTE_LABELS[field]} fuera de rango.")
                                    attr_values[field] = value
                                potential_flag = deps.parse_bool(parts[16]) if len(parts) > 16 else False
                                player = Player(
                                    name=name,
                                    national_id=national_id,
                                    age=age,
                                    position=position,
                                    club=club,
                                    country=country,
                                    photo_url=(
                                        raw_photo_url.strip()
                                        or deps.default_player_photo_url(name=name, national_id=national_id)
                                    ),
                                    potential_label=potential_flag,
                                    **attr_values,
                                )
                                seen_ids.add(national_id)
                            except (ValueError, IndexError) as exc:
                                errors.append(f"Linea {idx}: {exc}")
                                continue
                            db.add(player)
                            db.flush()
                            deps.sync_player_attribute_history(player, db, note="Alta masiva de jugador")
                            deps.refresh_player_potential(player, db)
                            created.append(player.name)
                        if created and not errors:
                            db.commit()
                            deps.invalidate_dashboard_cache()
                            flash(f"Listo: se agregaron {len(created)} jugadores.", "success")
                        elif created and errors:
                            db.commit()
                            deps.invalidate_dashboard_cache()
                            flash(
                                f"Se agregaron {len(created)} jugadores, pero quedaron advertencias para revisar.",
                                "warning",
                            )
                        else:
                            db.rollback()
                else:
                    errors.append("Modo de carga desconocido.")
            finally:
                db.close()
            if errors:
                for message in errors:
                    flash(message, "danger")
            if created and mode == "bulk":
                return redirect(url_for("manage_players"))
        attribute_sequence = [(field, deps.ATTRIBUTE_LABELS[field]) for field in deps.ATTRIBUTE_FIELDS]
        db_metrics = Session()
        current_total = db_metrics.query(func.count(Player.id)).scalar() or 0
        db_metrics.close()
        return render_template(
            "manage_players.html",
            attribute_labels=deps.ATTRIBUTE_LABELS,
            attribute_sequence=attribute_sequence,
            position_options=deps.POSITION_OPTIONS,
            current_total=current_total,
            max_players=deps.EVAL_POOL_MAX,
        )

    return bp
