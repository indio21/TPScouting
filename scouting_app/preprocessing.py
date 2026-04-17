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

from models import Player, PlayerAttributeHistory, PlayerStat
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
HISTORICAL_FEATURE_COLUMNS: List[str] = [*STATS_FEATURE_COLUMNS, *ATTRIBUTE_HISTORY_FEATURE_COLUMNS]
BASE_NUMERIC_FEATURE_COLUMNS: List[str] = ["age", *ATTRIBUTE_FIELDS]
NUMERIC_FEATURE_COLUMNS: List[str] = [*BASE_NUMERIC_FEATURE_COLUMNS, *HISTORICAL_FEATURE_COLUMNS]
CATEGORICAL_FEATURE_COLUMNS: List[str] = ["position"]
MODEL_FEATURE_COLUMNS: List[str] = [*NUMERIC_FEATURE_COLUMNS, *CATEGORICAL_FEATURE_COLUMNS]


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


def training_dataframe_from_engine(engine) -> pd.DataFrame:
    players_df = player_base_dataframe_from_engine(engine)
    stats_df = stats_dataframe_from_engine(engine)
    attribute_history_df = attribute_history_dataframe_from_engine(engine)
    return merge_historical_features(players_df, stats_df, attribute_history_df)


def dataframe_from_players(
    players: Iterable[Player],
    stats_feature_map: Optional[Dict[int, Dict[str, Optional[float]]]] = None,
    attribute_feature_map: Optional[Dict[int, Dict[str, Optional[float]]]] = None,
) -> pd.DataFrame:
    base_df = player_base_dataframe_from_players(players)
    rows = []
    stats_feature_map = stats_feature_map or {}
    attribute_feature_map = attribute_feature_map or {}
    for row in base_df.to_dict(orient="records"):
        player_id = row["player_id"]
        rows.append(
            {
                **row,
                **stats_feature_defaults(stats_feature_map.get(player_id)),
                **attribute_history_feature_defaults(attribute_feature_map.get(player_id)),
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


def merge_historical_features(
    players_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    attribute_history_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    merged = players_df.copy()
    stats_features_df = aggregate_stats_dataframe(stats_df)
    merged = merged.merge(stats_features_df, on="player_id", how="left")

    attribute_history_df = attribute_history_df if attribute_history_df is not None else pd.DataFrame()
    attr_features_df = aggregate_attribute_history_dataframe(players_df, attribute_history_df)
    merged = merged.merge(attr_features_df, on="player_id", how="left")

    for column in HISTORICAL_FEATURE_COLUMNS:
        if column not in merged.columns:
            merged[column] = np.nan
    return merged
