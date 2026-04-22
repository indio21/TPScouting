"""Preprocesamiento compartido para entrenamiento e inferencia."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sqlalchemy import select

from models import (
    Match,
    PhysicalAssessment,
    Player,
    PlayerAttributeHistory,
    PlayerAvailability,
    PlayerMatchParticipation,
    PlayerStat,
    ScoutReport,
)
from player_logic import ATTRIBUTE_FIELDS, POSITION_CHOICES, position_weights

STATS_FEATURE_COLUMNS: List[str] = [
    "stats_entry_count",
    "avg_final_score_hist",
    "avg_pass_accuracy_hist",
    "latest_final_score_hist",
]
ATTRIBUTE_HISTORY_FEATURE_COLUMNS: List[str] = [
    "attr_history_entry_count",
    "attr_avg_improvement_90d",
    "attr_avg_improvement_180d",
    "attr_avg_improvement_365d",
    "attr_avg_trend_per_day",
    "attr_weighted_improvement_90d",
    "attr_weighted_improvement_180d",
    "attr_weighted_improvement_365d",
    "attr_weighted_trend_per_day",
    "attr_weighted_volatility",
    "attr_current_vs_recent_gap",
]
MATCH_FEATURE_COLUMNS: List[str] = [
    "match_entry_count",
    "match_avg_final_score",
    "match_recent_final_score",
    "match_avg_minutes",
    "match_minutes_volatility",
    "match_start_rate",
    "match_avg_opponent_level",
    "match_high_difficulty_rate",
    "match_high_difficulty_score",
    "match_natural_position_rate",
]
SCOUT_REPORT_FEATURE_COLUMNS: List[str] = [
    "scout_report_count",
    "scout_avg_decision_making",
    "scout_avg_tactical_reading",
    "scout_avg_mental_profile",
    "scout_avg_adaptability",
    "scout_latest_projection_score",
    "scout_projection_trend",
]
PHYSICAL_FEATURE_COLUMNS: List[str] = [
    "phys_assessment_count",
    "phys_recent_height_cm",
    "phys_recent_weight_kg",
    "phys_recent_speed_score",
    "phys_recent_endurance_score",
    "phys_recent_bmi",
    "phys_height_growth_365d",
    "phys_weight_change_180d",
    "phys_growth_spurt_rate",
    "phys_left_footed_rate",
    "phys_two_footed_rate",
]
AVAILABILITY_FEATURE_COLUMNS: List[str] = [
    "avail_record_count",
    "avail_avg_pct",
    "avail_recent_pct",
    "avail_avg_fatigue",
    "avail_recent_fatigue",
    "avail_avg_training_load",
    "avail_injury_rate",
    "avail_missed_days_avg",
    "avail_availability_trend",
]
HISTORICAL_FEATURE_COLUMNS: List[str] = [
    *STATS_FEATURE_COLUMNS,
    *ATTRIBUTE_HISTORY_FEATURE_COLUMNS,
    *MATCH_FEATURE_COLUMNS,
    *SCOUT_REPORT_FEATURE_COLUMNS,
    *PHYSICAL_FEATURE_COLUMNS,
    *AVAILABILITY_FEATURE_COLUMNS,
]
BASE_NUMERIC_FEATURE_COLUMNS: List[str] = ["age", *ATTRIBUTE_FIELDS]
NUMERIC_FEATURE_COLUMNS: List[str] = [*BASE_NUMERIC_FEATURE_COLUMNS, *HISTORICAL_FEATURE_COLUMNS]
CATEGORICAL_FEATURE_COLUMNS: List[str] = ["position"]
MODEL_FEATURE_COLUMNS: List[str] = [*NUMERIC_FEATURE_COLUMNS, *CATEGORICAL_FEATURE_COLUMNS]
TEMPORAL_TARGET_COLUMN = "temporal_target_label"


def build_preprocessor() -> ColumnTransformer:
    base_numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )
    historical_numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)),
            ("scaler", MinMaxScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    categories=[POSITION_CHOICES],
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("base_numeric", base_numeric_pipeline, BASE_NUMERIC_FEATURE_COLUMNS),
            ("historical_numeric", historical_numeric_pipeline, HISTORICAL_FEATURE_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURE_COLUMNS),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def player_base_dataframe_from_engine(engine) -> pd.DataFrame:
    query = select(
        Player.id.label("player_id"),
        Player.age,
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
        Player.position,
        Player.potential_label,
    )
    with engine.connect() as connection:
        return pd.read_sql(query, connection)


def player_base_dataframe_from_players(players: Iterable[Player]) -> pd.DataFrame:
    rows = []
    for player in players:
        rows.append(
            {
                "player_id": player.id,
                "age": player.age,
                "pace": player.pace,
                "shooting": player.shooting,
                "passing": player.passing,
                "dribbling": player.dribbling,
                "defending": player.defending,
                "physical": player.physical,
                "vision": player.vision,
                "tackling": player.tackling,
                "determination": player.determination,
                "technique": player.technique,
                "position": player.position,
            }
        )
    return pd.DataFrame(rows, columns=["player_id", "age", *ATTRIBUTE_FIELDS, "position"])


def stats_dataframe_from_engine(engine) -> pd.DataFrame:
    query = select(
        PlayerStat.player_id,
        PlayerStat.record_date,
        PlayerStat.pass_accuracy,
        PlayerStat.final_score,
    )
    with engine.connect() as connection:
        return pd.read_sql(query, connection)


def attribute_history_dataframe_from_engine(engine) -> pd.DataFrame:
    query = select(
        PlayerAttributeHistory.player_id,
        PlayerAttributeHistory.record_date,
        PlayerAttributeHistory.pace,
        PlayerAttributeHistory.shooting,
        PlayerAttributeHistory.passing,
        PlayerAttributeHistory.dribbling,
        PlayerAttributeHistory.defending,
        PlayerAttributeHistory.physical,
        PlayerAttributeHistory.vision,
        PlayerAttributeHistory.tackling,
        PlayerAttributeHistory.determination,
        PlayerAttributeHistory.technique,
    )
    with engine.connect() as connection:
        return pd.read_sql(query, connection)


def match_participation_dataframe_from_engine(engine) -> pd.DataFrame:
    query = (
        select(
            PlayerMatchParticipation.player_id,
            Match.match_date,
            Match.opponent_level,
            PlayerMatchParticipation.started,
            PlayerMatchParticipation.position_played,
            PlayerMatchParticipation.minutes_played,
            PlayerMatchParticipation.final_score,
        )
        .join(Match, PlayerMatchParticipation.match_id == Match.id)
    )
    with engine.connect() as connection:
        return pd.read_sql(query, connection)


def scout_report_dataframe_from_engine(engine) -> pd.DataFrame:
    query = select(
        ScoutReport.player_id,
        ScoutReport.report_date,
        ScoutReport.decision_making,
        ScoutReport.tactical_reading,
        ScoutReport.mental_profile,
        ScoutReport.adaptability,
        ScoutReport.observed_projection_score,
    )
    with engine.connect() as connection:
        return pd.read_sql(query, connection)


def physical_assessment_dataframe_from_engine(engine) -> pd.DataFrame:
    query = select(
        PhysicalAssessment.player_id,
        PhysicalAssessment.assessment_date,
        PhysicalAssessment.height_cm,
        PhysicalAssessment.weight_kg,
        PhysicalAssessment.dominant_foot,
        PhysicalAssessment.estimated_speed,
        PhysicalAssessment.endurance,
        PhysicalAssessment.in_growth_spurt,
    )
    with engine.connect() as connection:
        return pd.read_sql(query, connection)


def availability_dataframe_from_engine(engine) -> pd.DataFrame:
    query = select(
        PlayerAvailability.player_id,
        PlayerAvailability.record_date,
        PlayerAvailability.availability_pct,
        PlayerAvailability.fatigue_pct,
        PlayerAvailability.training_load_pct,
        PlayerAvailability.missed_days,
        PlayerAvailability.injury_flag,
    )
    with engine.connect() as connection:
        return pd.read_sql(query, connection)


def training_dataframe_from_engine(engine) -> pd.DataFrame:
    players_df = player_base_dataframe_from_engine(engine)
    stats_df = stats_dataframe_from_engine(engine)
    attribute_history_df = attribute_history_dataframe_from_engine(engine)
    match_participation_df = match_participation_dataframe_from_engine(engine)
    scout_report_df = scout_report_dataframe_from_engine(engine)
    physical_assessment_df = physical_assessment_dataframe_from_engine(engine)
    availability_df = availability_dataframe_from_engine(engine)
    return merge_historical_features(
        players_df,
        stats_df,
        attribute_history_df,
        match_participation_df,
        scout_report_df,
        physical_assessment_df,
        availability_df,
    )


def temporal_training_dataframe_from_engine(engine) -> pd.DataFrame:
    players_df = player_base_dataframe_from_engine(engine)
    stats_df = stats_dataframe_from_engine(engine)
    attribute_history_df = attribute_history_dataframe_from_engine(engine)
    match_participation_df = match_participation_dataframe_from_engine(engine)
    scout_report_df = scout_report_dataframe_from_engine(engine)
    physical_assessment_df = physical_assessment_dataframe_from_engine(engine)
    availability_df = availability_dataframe_from_engine(engine)
    return build_temporal_training_dataframe(
        players_df,
        stats_df,
        attribute_history_df,
        match_participation_df,
        scout_report_df,
        physical_assessment_df,
        availability_df,
    )


def dataframe_from_players(
    players: Iterable[Player],
    stats_feature_map: Optional[Dict[int, Dict[str, Optional[float]]]] = None,
    attribute_feature_map: Optional[Dict[int, Dict[str, Optional[float]]]] = None,
    match_feature_map: Optional[Dict[int, Dict[str, Optional[float]]]] = None,
    scout_report_feature_map: Optional[Dict[int, Dict[str, Optional[float]]]] = None,
    physical_feature_map: Optional[Dict[int, Dict[str, Optional[float]]]] = None,
    availability_feature_map: Optional[Dict[int, Dict[str, Optional[float]]]] = None,
) -> pd.DataFrame:
    base_df = player_base_dataframe_from_players(players)
    rows = []
    stats_feature_map = stats_feature_map or {}
    attribute_feature_map = attribute_feature_map or {}
    match_feature_map = match_feature_map or {}
    scout_report_feature_map = scout_report_feature_map or {}
    physical_feature_map = physical_feature_map or {}
    availability_feature_map = availability_feature_map or {}
    for row in base_df.to_dict(orient="records"):
        player_id = row["player_id"]
        rows.append(
            {
                **row,
                **stats_feature_defaults(stats_feature_map.get(player_id)),
                **attribute_history_feature_defaults(attribute_feature_map.get(player_id)),
                **match_feature_defaults(match_feature_map.get(player_id)),
                **scout_report_feature_defaults(scout_report_feature_map.get(player_id)),
                **physical_feature_defaults(physical_feature_map.get(player_id)),
                **availability_feature_defaults(availability_feature_map.get(player_id)),
            }
        )
    return pd.DataFrame(rows, columns=["player_id", *MODEL_FEATURE_COLUMNS])


def feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, MODEL_FEATURE_COLUMNS].copy()


def fit_transform_features(df: pd.DataFrame, preprocessor: ColumnTransformer | None = None):
    transformer = preprocessor or build_preprocessor()
    transformed = transformer.fit_transform(feature_dataframe(df))
    return to_float32_array(transformed), transformer


def transform_features(df: pd.DataFrame, preprocessor: ColumnTransformer) -> np.ndarray:
    transformed = preprocessor.transform(feature_dataframe(df))
    return to_float32_array(transformed)


def to_float32_array(values) -> np.ndarray:
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values, dtype=np.float32)


def preprocessor_input_dim(preprocessor: ColumnTransformer) -> int:
    return len(preprocessor.get_feature_names_out())


def save_preprocessor(preprocessor: ColumnTransformer, path: str) -> None:
    dump(preprocessor, path)


def load_preprocessor(path: str) -> ColumnTransformer:
    return load(path)


def stats_feature_defaults(values: Optional[Dict[str, Optional[float]]] = None) -> Dict[str, Optional[float]]:
    source = values or {}
    return {
        "stats_entry_count": source.get("stats_entry_count"),
        "avg_final_score_hist": source.get("avg_final_score_hist"),
        "avg_pass_accuracy_hist": source.get("avg_pass_accuracy_hist"),
        "latest_final_score_hist": source.get("latest_final_score_hist"),
    }


def attribute_history_feature_defaults(
    values: Optional[Dict[str, Optional[float]]] = None,
) -> Dict[str, Optional[float]]:
    source = values or {}
    return {
        "attr_history_entry_count": source.get("attr_history_entry_count"),
        "attr_avg_improvement_90d": source.get("attr_avg_improvement_90d"),
        "attr_avg_improvement_180d": source.get("attr_avg_improvement_180d"),
        "attr_avg_improvement_365d": source.get("attr_avg_improvement_365d"),
        "attr_avg_trend_per_day": source.get("attr_avg_trend_per_day"),
        "attr_weighted_improvement_90d": source.get("attr_weighted_improvement_90d"),
        "attr_weighted_improvement_180d": source.get("attr_weighted_improvement_180d"),
        "attr_weighted_improvement_365d": source.get("attr_weighted_improvement_365d"),
        "attr_weighted_trend_per_day": source.get("attr_weighted_trend_per_day"),
        "attr_weighted_volatility": source.get("attr_weighted_volatility"),
        "attr_current_vs_recent_gap": source.get("attr_current_vs_recent_gap"),
    }


def match_feature_defaults(values: Optional[Dict[str, Optional[float]]] = None) -> Dict[str, Optional[float]]:
    source = values or {}
    return {
        "match_entry_count": source.get("match_entry_count"),
        "match_avg_final_score": source.get("match_avg_final_score"),
        "match_recent_final_score": source.get("match_recent_final_score"),
        "match_avg_minutes": source.get("match_avg_minutes"),
        "match_minutes_volatility": source.get("match_minutes_volatility"),
        "match_start_rate": source.get("match_start_rate"),
        "match_avg_opponent_level": source.get("match_avg_opponent_level"),
        "match_high_difficulty_rate": source.get("match_high_difficulty_rate"),
        "match_high_difficulty_score": source.get("match_high_difficulty_score"),
        "match_natural_position_rate": source.get("match_natural_position_rate"),
    }


def scout_report_feature_defaults(
    values: Optional[Dict[str, Optional[float]]] = None,
) -> Dict[str, Optional[float]]:
    source = values or {}
    return {
        "scout_report_count": source.get("scout_report_count"),
        "scout_avg_decision_making": source.get("scout_avg_decision_making"),
        "scout_avg_tactical_reading": source.get("scout_avg_tactical_reading"),
        "scout_avg_mental_profile": source.get("scout_avg_mental_profile"),
        "scout_avg_adaptability": source.get("scout_avg_adaptability"),
        "scout_latest_projection_score": source.get("scout_latest_projection_score"),
        "scout_projection_trend": source.get("scout_projection_trend"),
    }


def physical_feature_defaults(
    values: Optional[Dict[str, Optional[float]]] = None,
) -> Dict[str, Optional[float]]:
    source = values or {}
    return {
        "phys_assessment_count": source.get("phys_assessment_count"),
        "phys_recent_height_cm": source.get("phys_recent_height_cm"),
        "phys_recent_weight_kg": source.get("phys_recent_weight_kg"),
        "phys_recent_speed_score": source.get("phys_recent_speed_score"),
        "phys_recent_endurance_score": source.get("phys_recent_endurance_score"),
        "phys_recent_bmi": source.get("phys_recent_bmi"),
        "phys_height_growth_365d": source.get("phys_height_growth_365d"),
        "phys_weight_change_180d": source.get("phys_weight_change_180d"),
        "phys_growth_spurt_rate": source.get("phys_growth_spurt_rate"),
        "phys_left_footed_rate": source.get("phys_left_footed_rate"),
        "phys_two_footed_rate": source.get("phys_two_footed_rate"),
    }


def availability_feature_defaults(
    values: Optional[Dict[str, Optional[float]]] = None,
) -> Dict[str, Optional[float]]:
    source = values or {}
    return {
        "avail_record_count": source.get("avail_record_count"),
        "avail_avg_pct": source.get("avail_avg_pct"),
        "avail_recent_pct": source.get("avail_recent_pct"),
        "avail_avg_fatigue": source.get("avail_avg_fatigue"),
        "avail_recent_fatigue": source.get("avail_recent_fatigue"),
        "avail_avg_training_load": source.get("avail_avg_training_load"),
        "avail_injury_rate": source.get("avail_injury_rate"),
        "avail_missed_days_avg": source.get("avail_missed_days_avg"),
        "avail_availability_trend": source.get("avail_availability_trend"),
    }


def weighted_score_series(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)

    positions = df["position"].fillna("Mediocampista")
    scores = pd.Series(0.0, index=df.index, dtype=float)
    for pos in POSITION_CHOICES:
        mask = positions == pos
        if not mask.any():
            continue
        weights = position_weights(pos)
        subtotal = pd.Series(0.0, index=df.index[mask], dtype=float)
        for field in ATTRIBUTE_FIELDS:
            subtotal = subtotal.add(df.loc[mask, field].fillna(0).astype(float) * weights[field], fill_value=0.0)
        scores.loc[mask] = subtotal

    fallback_mask = ~positions.isin(POSITION_CHOICES)
    if fallback_mask.any():
        weights = position_weights(None)
        subtotal = pd.Series(0.0, index=df.index[fallback_mask], dtype=float)
        for field in ATTRIBUTE_FIELDS:
            subtotal = subtotal.add(
                df.loc[fallback_mask, field].fillna(0).astype(float) * weights[field],
                fill_value=0.0,
            )
        scores.loc[fallback_mask] = subtotal
    return scores


def aggregate_stats_dataframe(stats_df: pd.DataFrame) -> pd.DataFrame:
    if stats_df.empty:
        return pd.DataFrame(columns=["player_id", *STATS_FEATURE_COLUMNS])

    working_df = stats_df.copy()
    working_df["record_date"] = pd.to_datetime(working_df["record_date"], errors="coerce")
    grouped = (
        working_df.groupby("player_id", as_index=False)
        .agg(
            stats_entry_count=("player_id", "size"),
            avg_final_score_hist=("final_score", "mean"),
            avg_pass_accuracy_hist=("pass_accuracy", "mean"),
        )
    )
    latest_df = (
        working_df.sort_values(["player_id", "record_date"])
        .groupby("player_id", as_index=False)
        .tail(1)
        .loc[:, ["player_id", "final_score"]]
        .rename(columns={"final_score": "latest_final_score_hist"})
    )
    return grouped.merge(latest_df, on="player_id", how="left")


def _delta_for_window(history_df: pd.DataFrame, score_column: str, days: int) -> float:
    latest_date = history_df["record_date"].iloc[-1]
    cutoff = latest_date - pd.Timedelta(days=days)
    eligible = history_df[history_df["record_date"] <= cutoff]
    if eligible.empty:
        return np.nan
    latest_value = float(history_df.iloc[-1][score_column])
    reference_value = float(eligible.iloc[-1][score_column])
    return latest_value - reference_value


def _trend_per_day(history_df: pd.DataFrame, score_column: str) -> float:
    if len(history_df) < 2:
        return np.nan
    days = (history_df["record_date"] - history_df["record_date"].iloc[0]).dt.days.astype(float)
    if float(days.iloc[-1]) <= 0:
        return np.nan
    values = history_df[score_column].astype(float).to_numpy()
    return float(np.polyfit(days.to_numpy(), values, 1)[0])


def aggregate_attribute_history_dataframe(players_df: pd.DataFrame, attribute_history_df: pd.DataFrame) -> pd.DataFrame:
    if attribute_history_df.empty or players_df.empty:
        return pd.DataFrame(columns=["player_id", *ATTRIBUTE_HISTORY_FEATURE_COLUMNS])

    current_df = players_df.loc[:, ["player_id", "position", *ATTRIBUTE_FIELDS]].copy()
    current_df["current_avg_attr_score"] = current_df.loc[:, ATTRIBUTE_FIELDS].astype(float).mean(axis=1)
    current_df["current_weighted_attr_score"] = weighted_score_series(current_df)

    working_df = attribute_history_df.copy()
    working_df["record_date"] = pd.to_datetime(working_df["record_date"], errors="coerce")
    working_df = working_df.merge(
        current_df.loc[:, ["player_id", "position", "current_avg_attr_score", "current_weighted_attr_score"]],
        on="player_id",
        how="left",
    )
    working_df["history_avg_attr_score"] = working_df.loc[:, ATTRIBUTE_FIELDS].astype(float).mean(axis=1)
    working_df["history_weighted_attr_score"] = weighted_score_series(working_df)

    rows = []
    for player_id, history in working_df.groupby("player_id", sort=False):
        ordered = history.sort_values("record_date").reset_index(drop=True)
        recent_window = ordered["history_weighted_attr_score"].tail(min(3, len(ordered)))
        weighted_volatility = recent_window.diff().dropna().std()
        if pd.isna(weighted_volatility):
            weighted_volatility = np.nan
        rows.append(
            {
                "player_id": int(player_id),
                "attr_history_entry_count": int(len(ordered)),
                "attr_avg_improvement_90d": _delta_for_window(ordered, "history_avg_attr_score", 90),
                "attr_avg_improvement_180d": _delta_for_window(ordered, "history_avg_attr_score", 180),
                "attr_avg_improvement_365d": _delta_for_window(ordered, "history_avg_attr_score", 365),
                "attr_avg_trend_per_day": _trend_per_day(ordered, "history_avg_attr_score"),
                "attr_weighted_improvement_90d": _delta_for_window(ordered, "history_weighted_attr_score", 90),
                "attr_weighted_improvement_180d": _delta_for_window(ordered, "history_weighted_attr_score", 180),
                "attr_weighted_improvement_365d": _delta_for_window(ordered, "history_weighted_attr_score", 365),
                "attr_weighted_trend_per_day": _trend_per_day(ordered, "history_weighted_attr_score"),
                "attr_weighted_volatility": weighted_volatility,
                "attr_current_vs_recent_gap": float(ordered.iloc[-1]["current_weighted_attr_score"]) - float(recent_window.mean()),
            }
        )

    return pd.DataFrame(rows, columns=["player_id", *ATTRIBUTE_HISTORY_FEATURE_COLUMNS])


def aggregate_match_participation_dataframe(players_df: pd.DataFrame, participation_df: pd.DataFrame) -> pd.DataFrame:
    if participation_df.empty or players_df.empty:
        return pd.DataFrame(columns=["player_id", *MATCH_FEATURE_COLUMNS])

    current_df = players_df.loc[:, ["player_id", "position"]].copy()
    working_df = participation_df.copy()
    working_df["match_date"] = pd.to_datetime(working_df["match_date"], errors="coerce")
    working_df = working_df.merge(current_df, on="player_id", how="left")
    working_df["started_numeric"] = working_df["started"].fillna(False).astype(int)
    working_df["natural_position_match"] = (
        working_df["position_played"].fillna("") == working_df["position"].fillna("")
    ).astype(int)
    working_df["high_difficulty_match"] = (working_df["opponent_level"].fillna(0) >= 4).astype(int)
    working_df["high_difficulty_final_score"] = np.where(
        working_df["high_difficulty_match"] == 1,
        working_df["final_score"],
        np.nan,
    )

    grouped = (
        working_df.groupby("player_id", as_index=False)
        .agg(
            match_entry_count=("player_id", "size"),
            match_avg_final_score=("final_score", "mean"),
            match_avg_minutes=("minutes_played", "mean"),
            match_minutes_volatility=("minutes_played", "std"),
            match_start_rate=("started_numeric", "mean"),
            match_avg_opponent_level=("opponent_level", "mean"),
            match_high_difficulty_rate=("high_difficulty_match", "mean"),
            match_high_difficulty_score=("high_difficulty_final_score", "mean"),
            match_natural_position_rate=("natural_position_match", "mean"),
        )
    )
    latest_df = (
        working_df.sort_values(["player_id", "match_date"])
        .groupby("player_id", as_index=False)
        .tail(1)
        .loc[:, ["player_id", "final_score"]]
        .rename(columns={"final_score": "match_recent_final_score"})
    )
    merged = grouped.merge(latest_df, on="player_id", how="left")
    return merged.loc[:, ["player_id", *MATCH_FEATURE_COLUMNS]]


def aggregate_scout_report_dataframe(scout_report_df: pd.DataFrame) -> pd.DataFrame:
    if scout_report_df.empty:
        return pd.DataFrame(columns=["player_id", *SCOUT_REPORT_FEATURE_COLUMNS])

    working_df = scout_report_df.copy()
    working_df["report_date"] = pd.to_datetime(working_df["report_date"], errors="coerce")
    grouped = (
        working_df.groupby("player_id", as_index=False)
        .agg(
            scout_report_count=("player_id", "size"),
            scout_avg_decision_making=("decision_making", "mean"),
            scout_avg_tactical_reading=("tactical_reading", "mean"),
            scout_avg_mental_profile=("mental_profile", "mean"),
            scout_avg_adaptability=("adaptability", "mean"),
        )
    )
    latest_df = (
        working_df.sort_values(["player_id", "report_date"])
        .groupby("player_id", as_index=False)
        .tail(1)
        .loc[:, ["player_id", "observed_projection_score"]]
        .rename(columns={"observed_projection_score": "scout_latest_projection_score"})
    )
    trend_rows = []
    for player_id, group in working_df.groupby("player_id", sort=False):
        ordered = group.sort_values("report_date").dropna(subset=["observed_projection_score"])
        if len(ordered) < 2:
            projection_trend = np.nan
        else:
            days = (ordered["report_date"] - ordered["report_date"].iloc[0]).dt.days.astype(float)
            if float(days.iloc[-1]) <= 0:
                projection_trend = np.nan
            else:
                projection_trend = float(np.polyfit(days.to_numpy(), ordered["observed_projection_score"].astype(float).to_numpy(), 1)[0])
        trend_rows.append({"player_id": int(player_id), "scout_projection_trend": projection_trend})
    trend_df = pd.DataFrame(trend_rows, columns=["player_id", "scout_projection_trend"])
    return grouped.merge(latest_df, on="player_id", how="left").merge(trend_df, on="player_id", how="left")


def aggregate_physical_assessment_dataframe(physical_assessment_df: pd.DataFrame) -> pd.DataFrame:
    if physical_assessment_df.empty:
        return pd.DataFrame(columns=["player_id", *PHYSICAL_FEATURE_COLUMNS])

    working_df = physical_assessment_df.copy()
    working_df["assessment_date"] = pd.to_datetime(working_df["assessment_date"], errors="coerce")
    working_df["dominant_foot"] = working_df["dominant_foot"].fillna("").astype(str).str.lower()
    working_df["phys_left_footed_flag"] = working_df["dominant_foot"].isin(["izquierda", "left"]).astype(int)
    working_df["phys_two_footed_flag"] = working_df["dominant_foot"].isin(["ambos", "both"]).astype(int)
    working_df["in_growth_spurt_numeric"] = working_df["in_growth_spurt"].fillna(False).astype(int)
    working_df["phys_recent_bmi"] = np.where(
        (working_df["height_cm"].fillna(0) > 0) & (working_df["weight_kg"].fillna(0) > 0),
        working_df["weight_kg"].astype(float) / np.power(working_df["height_cm"].astype(float) / 100.0, 2),
        np.nan,
    )

    grouped = (
        working_df.groupby("player_id", as_index=False)
        .agg(
            phys_assessment_count=("player_id", "size"),
            phys_growth_spurt_rate=("in_growth_spurt_numeric", "mean"),
            phys_left_footed_rate=("phys_left_footed_flag", "mean"),
            phys_two_footed_rate=("phys_two_footed_flag", "mean"),
        )
    )
    latest_df = (
        working_df.sort_values(["player_id", "assessment_date"])
        .groupby("player_id", as_index=False)
        .tail(1)
        .loc[:, ["player_id", "height_cm", "weight_kg", "estimated_speed", "endurance", "phys_recent_bmi"]]
        .rename(
            columns={
                "height_cm": "phys_recent_height_cm",
                "weight_kg": "phys_recent_weight_kg",
                "estimated_speed": "phys_recent_speed_score",
                "endurance": "phys_recent_endurance_score",
            }
        )
    )
    growth_rows = []
    for player_id, group in working_df.groupby("player_id", sort=False):
        ordered = group.sort_values("assessment_date").reset_index(drop=True).rename(columns={"assessment_date": "record_date"})
        growth_rows.append(
            {
                "player_id": int(player_id),
                "phys_height_growth_365d": _delta_for_window(ordered, "height_cm", 365),
                "phys_weight_change_180d": _delta_for_window(ordered, "weight_kg", 180),
            }
        )
    growth_df = pd.DataFrame(growth_rows, columns=["player_id", "phys_height_growth_365d", "phys_weight_change_180d"])
    return (
        grouped.merge(latest_df, on="player_id", how="left")
        .merge(growth_df, on="player_id", how="left")
        .loc[:, ["player_id", *PHYSICAL_FEATURE_COLUMNS]]
    )


def aggregate_availability_dataframe(availability_df: pd.DataFrame) -> pd.DataFrame:
    if availability_df.empty:
        return pd.DataFrame(columns=["player_id", *AVAILABILITY_FEATURE_COLUMNS])

    working_df = availability_df.copy()
    working_df["record_date"] = pd.to_datetime(working_df["record_date"], errors="coerce")
    working_df["injury_numeric"] = working_df["injury_flag"].fillna(False).astype(int)

    grouped = (
        working_df.groupby("player_id", as_index=False)
        .agg(
            avail_record_count=("player_id", "size"),
            avail_avg_pct=("availability_pct", "mean"),
            avail_avg_fatigue=("fatigue_pct", "mean"),
            avail_avg_training_load=("training_load_pct", "mean"),
            avail_injury_rate=("injury_numeric", "mean"),
            avail_missed_days_avg=("missed_days", "mean"),
        )
    )
    latest_df = (
        working_df.sort_values(["player_id", "record_date"])
        .groupby("player_id", as_index=False)
        .tail(1)
        .loc[:, ["player_id", "availability_pct", "fatigue_pct"]]
        .rename(
            columns={
                "availability_pct": "avail_recent_pct",
                "fatigue_pct": "avail_recent_fatigue",
            }
        )
    )
    trend_rows = []
    for player_id, group in working_df.groupby("player_id", sort=False):
        ordered = group.sort_values("record_date").dropna(subset=["availability_pct"])
        if len(ordered) < 2:
            availability_trend = np.nan
        else:
            days = (ordered["record_date"] - ordered["record_date"].iloc[0]).dt.days.astype(float)
            if float(days.iloc[-1]) <= 0:
                availability_trend = np.nan
            else:
                availability_trend = float(np.polyfit(days.to_numpy(), ordered["availability_pct"].astype(float).to_numpy(), 1)[0])
        trend_rows.append({"player_id": int(player_id), "avail_availability_trend": availability_trend})
    trend_df = pd.DataFrame(trend_rows, columns=["player_id", "avail_availability_trend"])
    return (
        grouped.merge(latest_df, on="player_id", how="left")
        .merge(trend_df, on="player_id", how="left")
        .loc[:, ["player_id", *AVAILABILITY_FEATURE_COLUMNS]]
    )


def _timestamp_cutoff_map(
    attribute_history_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    physical_assessment_df: Optional[pd.DataFrame] = None,
    availability_df: Optional[pd.DataFrame] = None,
) -> Dict[int, pd.Timestamp]:
    date_frames: List[pd.DataFrame] = []
    if not attribute_history_df.empty:
        attr_dates = attribute_history_df.loc[:, ["player_id", "record_date"]].copy()
        attr_dates["event_date"] = pd.to_datetime(attr_dates["record_date"], errors="coerce")
        date_frames.append(attr_dates.loc[:, ["player_id", "event_date"]])
    if not stats_df.empty:
        stat_dates = stats_df.loc[:, ["player_id", "record_date"]].copy()
        stat_dates["event_date"] = pd.to_datetime(stat_dates["record_date"], errors="coerce")
        date_frames.append(stat_dates.loc[:, ["player_id", "event_date"]])
    if physical_assessment_df is not None and not physical_assessment_df.empty:
        physical_dates = physical_assessment_df.loc[:, ["player_id", "assessment_date"]].copy()
        physical_dates["event_date"] = pd.to_datetime(physical_dates["assessment_date"], errors="coerce")
        date_frames.append(physical_dates.loc[:, ["player_id", "event_date"]])
    if availability_df is not None and not availability_df.empty:
        availability_dates = availability_df.loc[:, ["player_id", "record_date"]].copy()
        availability_dates["event_date"] = pd.to_datetime(availability_dates["record_date"], errors="coerce")
        date_frames.append(availability_dates.loc[:, ["player_id", "event_date"]])
    if not date_frames:
        return {}

    combined_dates = pd.concat(date_frames, ignore_index=True).dropna(subset=["event_date"])
    if combined_dates.empty:
        return {}

    cutoff_map: Dict[int, pd.Timestamp] = {}
    for player_id, group in combined_dates.groupby("player_id", sort=False):
        dates = sorted(pd.Series(group["event_date"]).dropna().unique())
        if len(dates) < 2:
            continue
        future_window = min(3, max(1, len(dates) // 3))
        cutoff_index = max(0, len(dates) - future_window - 1)
        cutoff_map[int(player_id)] = pd.Timestamp(dates[cutoff_index])
    return cutoff_map


def _filter_rows_by_cutoff(
    df: pd.DataFrame,
    player_id_col: str,
    date_col: str,
    cutoff_map: Dict[int, pd.Timestamp],
    keep_observed: bool,
) -> pd.DataFrame:
    if df.empty or not cutoff_map:
        return df.iloc[0:0].copy()
    cutoff_df = pd.DataFrame(
        [(player_id, cutoff) for player_id, cutoff in cutoff_map.items()],
        columns=[player_id_col, "_cutoff_date"],
    )
    working_df = df.copy()
    working_df[date_col] = pd.to_datetime(working_df[date_col], errors="coerce")
    working_df = working_df.merge(cutoff_df, on=player_id_col, how="inner")
    if keep_observed:
        filtered = working_df[working_df[date_col] <= working_df["_cutoff_date"]]
    else:
        filtered = working_df[working_df[date_col] > working_df["_cutoff_date"]]
    return filtered.drop(columns=["_cutoff_date"])


def _sigmoid_component(value: float, center: float = 0.0, scale: float = 1.0) -> float:
    if pd.isna(value):
        return 0.0
    safe_scale = scale if abs(scale) > 1e-6 else 1.0
    return float(1.0 / (1.0 + np.exp(-((float(value) - center) / safe_scale))))


def _future_match_target_metrics(
    players_df: pd.DataFrame,
    match_participation_df: pd.DataFrame,
    cutoff_map: Dict[int, pd.Timestamp],
) -> pd.DataFrame:
    if players_df.empty or match_participation_df.empty or not cutoff_map:
        return pd.DataFrame(
            columns=[
                "player_id",
                "future_match_entry_count",
                "future_avg_match_score",
                "future_minutes_per_match",
                "future_start_rate",
                "future_avg_opponent_level",
                "future_high_difficulty_rate",
                "future_high_difficulty_score",
                "future_natural_position_rate",
            ]
        )

    future_match_df = _filter_rows_by_cutoff(match_participation_df, "player_id", "match_date", cutoff_map, False)
    if future_match_df.empty:
        return pd.DataFrame(
            columns=[
                "player_id",
                "future_match_entry_count",
                "future_avg_match_score",
                "future_minutes_per_match",
                "future_start_rate",
                "future_avg_opponent_level",
                "future_high_difficulty_rate",
                "future_high_difficulty_score",
                "future_natural_position_rate",
            ]
        )

    current_df = players_df.loc[:, ["player_id", "position"]].copy()
    future_match_df = future_match_df.merge(current_df, on="player_id", how="left")
    future_match_df["started_numeric"] = future_match_df["started"].fillna(False).astype(int)
    future_match_df["natural_position_match"] = (
        future_match_df["position_played"].fillna("") == future_match_df["position"].fillna("")
    ).astype(int)
    future_match_df["high_difficulty_match"] = (future_match_df["opponent_level"].fillna(0) >= 4).astype(int)
    future_match_df["high_difficulty_final_score"] = np.where(
        future_match_df["high_difficulty_match"] == 1,
        future_match_df["final_score"],
        np.nan,
    )
    grouped = (
        future_match_df.groupby("player_id", as_index=False)
        .agg(
            future_match_entry_count=("player_id", "size"),
            future_avg_match_score=("final_score", "mean"),
            future_minutes_per_match=("minutes_played", "mean"),
            future_start_rate=("started_numeric", "mean"),
            future_avg_opponent_level=("opponent_level", "mean"),
            future_high_difficulty_rate=("high_difficulty_match", "mean"),
            future_high_difficulty_score=("high_difficulty_final_score", "mean"),
            future_natural_position_rate=("natural_position_match", "mean"),
        )
    )
    return grouped


def _future_availability_target_metrics(
    availability_df: pd.DataFrame,
    cutoff_map: Dict[int, pd.Timestamp],
) -> pd.DataFrame:
    if availability_df.empty or not cutoff_map:
        return pd.DataFrame(
            columns=[
                "player_id",
                "future_availability_pct",
                "future_fatigue_pct",
                "future_training_load_pct",
                "future_injury_rate",
                "future_missed_days_avg",
            ]
        )

    future_availability_df = _filter_rows_by_cutoff(availability_df, "player_id", "record_date", cutoff_map, False)
    if future_availability_df.empty:
        return pd.DataFrame(
            columns=[
                "player_id",
                "future_availability_pct",
                "future_fatigue_pct",
                "future_training_load_pct",
                "future_injury_rate",
                "future_missed_days_avg",
            ]
        )

    future_availability_df["injury_numeric"] = future_availability_df["injury_flag"].fillna(False).astype(int)
    return (
        future_availability_df.groupby("player_id", as_index=False)
        .agg(
            future_availability_pct=("availability_pct", "mean"),
            future_fatigue_pct=("fatigue_pct", "mean"),
            future_training_load_pct=("training_load_pct", "mean"),
            future_injury_rate=("injury_numeric", "mean"),
            future_missed_days_avg=("missed_days", "mean"),
        )
    )


def _temporal_anchor_players_dataframe(
    players_df: pd.DataFrame,
    attribute_history_df: pd.DataFrame,
    cutoff_map: Dict[int, pd.Timestamp],
) -> pd.DataFrame:
    if players_df.empty:
        return players_df.copy()

    attr_working_df = attribute_history_df.copy()
    if not attr_working_df.empty:
        attr_working_df["record_date"] = pd.to_datetime(attr_working_df["record_date"], errors="coerce")

    latest_attr_df = pd.DataFrame()
    latest_history_date_map: Dict[int, pd.Timestamp] = {}
    if not attr_working_df.empty:
        latest_history_rows = (
            attr_working_df.sort_values(["player_id", "record_date"])
            .groupby("player_id", as_index=False)
            .tail(1)
            .loc[:, ["player_id", "record_date"]]
        )
        latest_history_date_map = {
            int(row["player_id"]): pd.Timestamp(row["record_date"])
            for row in latest_history_rows.to_dict(orient="records")
            if pd.notna(row["record_date"])
        }
        observed_attr_df = _filter_rows_by_cutoff(attr_working_df, "player_id", "record_date", cutoff_map, True)
        if not observed_attr_df.empty:
            latest_attr_df = (
                observed_attr_df.sort_values(["player_id", "record_date"])
                .groupby("player_id", as_index=False)
                .tail(1)
            )

    latest_attr_map = {
        int(row["player_id"]): row
        for row in latest_attr_df.to_dict(orient="records")
    }

    rows = []
    for player in players_df.to_dict(orient="records"):
        player_id = int(player["player_id"])
        anchor_row = latest_attr_map.get(player_id)
        cutoff_date = cutoff_map.get(player_id)
        latest_history_date = latest_history_date_map.get(player_id)
        if anchor_row is not None:
            anchor_attrs = {field: float(anchor_row[field]) for field in ATTRIBUTE_FIELDS}
        else:
            anchor_attrs = {field: float(player[field]) for field in ATTRIBUTE_FIELDS}

        age_value = float(player["age"])
        if cutoff_date is not None and latest_history_date is not None and latest_history_date > cutoff_date:
            age_value = max(12.0, age_value - ((latest_history_date - cutoff_date).days / 365.25))

        rows.append(
            {
                "player_id": player_id,
                "age": round(age_value, 2),
                **anchor_attrs,
                "position": player["position"],
                "potential_label": player.get("potential_label"),
            }
        )
    return pd.DataFrame(rows, columns=["player_id", "age", *ATTRIBUTE_FIELDS, "position", "potential_label"])


def _temporal_target_dataframe(
    players_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    attribute_history_df: pd.DataFrame,
    cutoff_map: Dict[int, pd.Timestamp],
    match_participation_df: Optional[pd.DataFrame] = None,
    availability_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if players_df.empty or not cutoff_map:
        return pd.DataFrame(
            columns=[
                "player_id",
                TEMPORAL_TARGET_COLUMN,
                "progression_score",
                "temporal_target_threshold",
                "temporal_future_score_threshold",
                "temporal_consolidation_path",
                "temporal_breakout_path",
                "observed_weighted_score",
                "future_weighted_score",
                "weighted_score_growth",
                "observed_final_score",
                "future_final_score",
                "final_score_growth",
                "future_availability_pct",
                "future_fatigue_pct",
                "future_training_load_pct",
                "future_injury_rate",
                "future_missed_days_avg",
            ]
        )

    attr_working_df = attribute_history_df.copy()
    attr_working_df["record_date"] = pd.to_datetime(attr_working_df["record_date"], errors="coerce")
    attr_working_df = attr_working_df.merge(
        players_df.loc[:, ["player_id", "position"]],
        on="player_id",
        how="left",
    )
    attr_working_df["history_weighted_attr_score"] = weighted_score_series(attr_working_df)

    observed_attr_df = _filter_rows_by_cutoff(attr_working_df, "player_id", "record_date", cutoff_map, True)
    future_attr_df = _filter_rows_by_cutoff(attr_working_df, "player_id", "record_date", cutoff_map, False)

    stats_working_df = stats_df.copy()
    stats_working_df["record_date"] = pd.to_datetime(stats_working_df["record_date"], errors="coerce")
    observed_stats_df = _filter_rows_by_cutoff(stats_working_df, "player_id", "record_date", cutoff_map, True)
    future_stats_df = _filter_rows_by_cutoff(stats_working_df, "player_id", "record_date", cutoff_map, False)
    future_match_metrics_df = _future_match_target_metrics(
        players_df,
        match_participation_df if match_participation_df is not None else pd.DataFrame(),
        cutoff_map,
    )
    future_match_metrics_map = {
        int(row["player_id"]): row
        for row in future_match_metrics_df.to_dict(orient="records")
    }
    future_availability_metrics_df = _future_availability_target_metrics(
        availability_df if availability_df is not None else pd.DataFrame(),
        cutoff_map,
    )
    future_availability_metrics_map = {
        int(row["player_id"]): row
        for row in future_availability_metrics_df.to_dict(orient="records")
    }

    rows = []
    for player_id in players_df["player_id"].astype(int).tolist():
        observed_attr = observed_attr_df[observed_attr_df["player_id"] == player_id].sort_values("record_date")
        future_attr = future_attr_df[future_attr_df["player_id"] == player_id].sort_values("record_date")
        if observed_attr.empty or future_attr.empty:
            continue

        observed_weighted_score = float(observed_attr["history_weighted_attr_score"].tail(min(2, len(observed_attr))).mean())
        future_weighted_score = float(future_attr["history_weighted_attr_score"].mean())
        weighted_score_growth = future_weighted_score - observed_weighted_score

        observed_stats = observed_stats_df[observed_stats_df["player_id"] == player_id].sort_values("record_date")
        future_stats = future_stats_df[future_stats_df["player_id"] == player_id].sort_values("record_date")
        observed_final_score = (
            float(observed_stats["final_score"].tail(min(2, len(observed_stats))).mean())
            if not observed_stats.empty and observed_stats["final_score"].notna().any()
            else np.nan
        )
        future_final_score = (
            float(future_stats["final_score"].mean())
            if not future_stats.empty and future_stats["final_score"].notna().any()
            else np.nan
        )
        final_score_growth = (
            future_final_score - observed_final_score
            if not np.isnan(observed_final_score) and not np.isnan(future_final_score)
            else np.nan
        )
        recent_window = observed_attr["history_weighted_attr_score"].tail(min(3, len(observed_attr)))
        observed_weighted_volatility = float(recent_window.diff().dropna().std()) if len(recent_window) >= 2 else np.nan

        match_metrics = future_match_metrics_map.get(player_id, {})
        future_minutes_per_match = float(match_metrics.get("future_minutes_per_match")) if match_metrics.get("future_minutes_per_match") is not None else np.nan
        future_start_rate = float(match_metrics.get("future_start_rate")) if match_metrics.get("future_start_rate") is not None else np.nan
        future_avg_opponent_level = float(match_metrics.get("future_avg_opponent_level")) if match_metrics.get("future_avg_opponent_level") is not None else np.nan
        future_high_difficulty_score = float(match_metrics.get("future_high_difficulty_score")) if match_metrics.get("future_high_difficulty_score") is not None else np.nan
        future_natural_position_rate = float(match_metrics.get("future_natural_position_rate")) if match_metrics.get("future_natural_position_rate") is not None else np.nan
        future_match_entry_count = float(match_metrics.get("future_match_entry_count")) if match_metrics.get("future_match_entry_count") is not None else np.nan
        availability_metrics = future_availability_metrics_map.get(player_id, {})
        future_availability_pct = float(availability_metrics.get("future_availability_pct")) if availability_metrics.get("future_availability_pct") is not None else np.nan
        future_fatigue_pct = float(availability_metrics.get("future_fatigue_pct")) if availability_metrics.get("future_fatigue_pct") is not None else np.nan
        future_training_load_pct = float(availability_metrics.get("future_training_load_pct")) if availability_metrics.get("future_training_load_pct") is not None else np.nan
        future_injury_rate = float(availability_metrics.get("future_injury_rate")) if availability_metrics.get("future_injury_rate") is not None else np.nan
        future_missed_days_avg = float(availability_metrics.get("future_missed_days_avg")) if availability_metrics.get("future_missed_days_avg") is not None else np.nan

        growth_component = _sigmoid_component(weighted_score_growth, center=0.14, scale=0.22)
        future_level_component = _sigmoid_component(future_weighted_score, center=12.7, scale=0.65)
        performance_component = _sigmoid_component(
            final_score_growth if not np.isnan(final_score_growth) else (future_final_score - 6.2 if not np.isnan(future_final_score) else np.nan),
            center=0.08,
            scale=0.28,
        )
        challenge_component = _sigmoid_component(future_high_difficulty_score, center=6.7, scale=0.32) * _sigmoid_component(
            future_avg_opponent_level,
            center=3.4,
            scale=0.45,
        )
        consistency_component = (
            _sigmoid_component(future_minutes_per_match, center=58.0, scale=12.0)
            * _sigmoid_component(future_start_rate, center=0.46, scale=0.14)
            * _sigmoid_component(future_match_entry_count, center=2.2, scale=0.8)
        )
        role_component = (
            0.75 * _sigmoid_component(future_natural_position_rate, center=0.62, scale=0.18)
            + 0.25 * _sigmoid_component(future_natural_position_rate, center=0.88, scale=0.08)
        )
        availability_component = (
            _sigmoid_component(future_availability_pct, center=78.0, scale=7.5)
            * _sigmoid_component(
                100.0 - future_fatigue_pct if not np.isnan(future_fatigue_pct) else np.nan,
                center=45.0,
                scale=14.0,
            )
            * _sigmoid_component(
                1.0 - future_injury_rate if not np.isnan(future_injury_rate) else np.nan,
                center=0.82,
                scale=0.12,
            )
        )
        breakout_component = growth_component * max(performance_component, challenge_component) * max(consistency_component, 0.15)
        stability_penalty = _sigmoid_component(observed_weighted_volatility, center=0.24, scale=0.08)
        progression_score = (
            0.18 * growth_component
            + 0.15 * future_level_component
            + 0.17 * performance_component
            + 0.14 * challenge_component
            + 0.10 * consistency_component
            + 0.08 * role_component
            + 0.09 * availability_component
            + 0.19 * breakout_component
            - 0.10 * stability_penalty
        )

        rows.append(
            {
                "player_id": player_id,
                TEMPORAL_TARGET_COLUMN: False,
                "progression_score": progression_score,
                "temporal_target_threshold": np.nan,
                "temporal_future_score_threshold": np.nan,
                "temporal_consolidation_path": False,
                "temporal_breakout_path": False,
                "observed_weighted_score": observed_weighted_score,
                "future_weighted_score": future_weighted_score,
                "weighted_score_growth": weighted_score_growth,
                "observed_final_score": observed_final_score,
                "future_final_score": future_final_score,
                "final_score_growth": final_score_growth,
                "future_availability_pct": future_availability_pct,
                "future_fatigue_pct": future_fatigue_pct,
                "future_training_load_pct": future_training_load_pct,
                "future_injury_rate": future_injury_rate,
                "future_missed_days_avg": future_missed_days_avg,
                "future_minutes_per_match": future_minutes_per_match,
                "future_start_rate": future_start_rate,
                "future_avg_opponent_level": future_avg_opponent_level,
                "future_high_difficulty_score": future_high_difficulty_score,
                "future_natural_position_rate": future_natural_position_rate,
                "observed_weighted_volatility": observed_weighted_volatility,
            }
        )
    target_df = pd.DataFrame(rows)
    if target_df.empty:
        return target_df

    progression_quantile_threshold = float(target_df["progression_score"].quantile(0.86))
    future_final_quantile_threshold = float(target_df["future_final_score"].dropna().quantile(0.93))
    volatility_threshold = float(target_df["observed_weighted_volatility"].dropna().quantile(0.70))
    breakout_growth_threshold = float(target_df["weighted_score_growth"].dropna().quantile(0.88))
    breakout_final_growth_threshold = float(target_df["final_score_growth"].dropna().quantile(0.88))
    breakout_difficulty_threshold = float(target_df["future_high_difficulty_score"].dropna().quantile(0.80))
    availability_threshold = float(target_df["future_availability_pct"].dropna().quantile(0.48))
    low_fatigue_threshold = float(target_df["future_fatigue_pct"].dropna().quantile(0.58))
    injury_threshold = float(target_df["future_injury_rate"].dropna().quantile(0.70))

    target_threshold = max(progression_quantile_threshold, 0.37)
    future_score_threshold = max(future_final_quantile_threshold, 4.60)
    target_df["temporal_target_threshold"] = target_threshold
    target_df["temporal_future_score_threshold"] = future_score_threshold
    consolidation_mask = (
        (target_df["progression_score"] >= target_threshold)
        & (target_df["future_final_score"].fillna(0.0) >= future_score_threshold)
        & (target_df["future_natural_position_rate"].fillna(0.0) >= 0.65)
        & (target_df["observed_weighted_volatility"].fillna(99.0) <= volatility_threshold)
        & (target_df["future_availability_pct"].fillna(0.0) >= max(availability_threshold, 70.0))
        & (target_df["future_fatigue_pct"].fillna(100.0) <= min(low_fatigue_threshold, 40.0))
        & (target_df["future_injury_rate"].fillna(1.0) <= max(injury_threshold, 0.28))
        & (
            (target_df["future_minutes_per_match"].fillna(0.0) >= 42.0)
            | (target_df["future_start_rate"].fillna(0.0) >= 0.40)
        )
    )
    breakout_mask = (
        (target_df["weighted_score_growth"].fillna(-99.0) >= breakout_growth_threshold)
        & (target_df["final_score_growth"].fillna(-99.0) >= breakout_final_growth_threshold)
        & (target_df["future_high_difficulty_score"].fillna(0.0) >= breakout_difficulty_threshold)
        & (target_df["future_start_rate"].fillna(0.0) >= 0.45)
        & (target_df["future_availability_pct"].fillna(0.0) >= 62.0)
        & (target_df["future_injury_rate"].fillna(1.0) <= 0.28)
    )
    target_df["temporal_consolidation_path"] = consolidation_mask
    target_df["temporal_breakout_path"] = breakout_mask
    target_df[TEMPORAL_TARGET_COLUMN] = consolidation_mask | breakout_mask
    return target_df


def build_temporal_training_dataframe(
    players_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    attribute_history_df: pd.DataFrame,
    match_participation_df: Optional[pd.DataFrame] = None,
    scout_report_df: Optional[pd.DataFrame] = None,
    physical_assessment_df: Optional[pd.DataFrame] = None,
    availability_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    cutoff_map = _timestamp_cutoff_map(
        attribute_history_df,
        stats_df,
        physical_assessment_df=physical_assessment_df,
        availability_df=availability_df,
    )
    if not cutoff_map:
        empty_columns = [
            "player_id",
            "age",
            *ATTRIBUTE_FIELDS,
            "position",
            "potential_label",
            *HISTORICAL_FEATURE_COLUMNS,
            TEMPORAL_TARGET_COLUMN,
            "progression_score",
            "temporal_target_threshold",
            "temporal_future_score_threshold",
            "temporal_consolidation_path",
            "temporal_breakout_path",
            "observed_weighted_score",
            "future_weighted_score",
            "weighted_score_growth",
            "observed_final_score",
            "future_final_score",
            "final_score_growth",
            "future_availability_pct",
            "future_fatigue_pct",
            "future_training_load_pct",
            "future_injury_rate",
            "future_missed_days_avg",
            "future_minutes_per_match",
            "future_start_rate",
            "future_avg_opponent_level",
            "future_high_difficulty_score",
            "future_natural_position_rate",
            "observed_weighted_volatility",
        ]
        return pd.DataFrame(columns=empty_columns)

    anchor_players_df = _temporal_anchor_players_dataframe(players_df, attribute_history_df, cutoff_map)
    observed_stats_df = _filter_rows_by_cutoff(stats_df, "player_id", "record_date", cutoff_map, True)
    observed_attr_df = _filter_rows_by_cutoff(attribute_history_df, "player_id", "record_date", cutoff_map, True)
    observed_match_df = _filter_rows_by_cutoff(
        match_participation_df if match_participation_df is not None else pd.DataFrame(),
        "player_id",
        "match_date",
        cutoff_map,
        True,
    )
    observed_scout_df = _filter_rows_by_cutoff(
        scout_report_df if scout_report_df is not None else pd.DataFrame(),
        "player_id",
        "report_date",
        cutoff_map,
        True,
    )
    observed_physical_df = _filter_rows_by_cutoff(
        physical_assessment_df if physical_assessment_df is not None else pd.DataFrame(),
        "player_id",
        "assessment_date",
        cutoff_map,
        True,
    )
    observed_availability_df = _filter_rows_by_cutoff(
        availability_df if availability_df is not None else pd.DataFrame(),
        "player_id",
        "record_date",
        cutoff_map,
        True,
    )
    features_df = merge_historical_features(
        anchor_players_df,
        observed_stats_df,
        observed_attr_df,
        observed_match_df,
        observed_scout_df,
        observed_physical_df,
        observed_availability_df,
    )
    target_df = _temporal_target_dataframe(
        players_df,
        stats_df,
        attribute_history_df,
        cutoff_map,
        match_participation_df=match_participation_df,
        availability_df=availability_df,
    )
    if target_df.empty:
        return target_df
    merged_df = features_df.merge(target_df, on="player_id", how="inner")
    return merged_df


def merge_historical_features(
    players_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    attribute_history_df: Optional[pd.DataFrame] = None,
    match_participation_df: Optional[pd.DataFrame] = None,
    scout_report_df: Optional[pd.DataFrame] = None,
    physical_assessment_df: Optional[pd.DataFrame] = None,
    availability_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    merged = players_df.copy()
    stats_features_df = aggregate_stats_dataframe(stats_df)
    merged = merged.merge(stats_features_df, on="player_id", how="left")

    attribute_history_df = attribute_history_df if attribute_history_df is not None else pd.DataFrame()
    attr_features_df = aggregate_attribute_history_dataframe(players_df, attribute_history_df)
    merged = merged.merge(attr_features_df, on="player_id", how="left")

    match_participation_df = match_participation_df if match_participation_df is not None else pd.DataFrame()
    match_features_df = aggregate_match_participation_dataframe(players_df, match_participation_df)
    merged = merged.merge(match_features_df, on="player_id", how="left")

    scout_report_df = scout_report_df if scout_report_df is not None else pd.DataFrame()
    scout_features_df = aggregate_scout_report_dataframe(scout_report_df)
    merged = merged.merge(scout_features_df, on="player_id", how="left")

    physical_assessment_df = physical_assessment_df if physical_assessment_df is not None else pd.DataFrame()
    physical_features_df = aggregate_physical_assessment_dataframe(physical_assessment_df)
    merged = merged.merge(physical_features_df, on="player_id", how="left")

    availability_df = availability_df if availability_df is not None else pd.DataFrame()
    availability_features_df = aggregate_availability_dataframe(availability_df)
    merged = merged.merge(availability_features_df, on="player_id", how="left")

    for column in HISTORICAL_FEATURE_COLUMNS:
        if column not in merged.columns:
            merged[column] = np.nan
    return merged
