"""Auditoría reproducible de los artefactos históricos sin reentrenarlos."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from joblib import load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "scouting_app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from evaluate_saved_model import _select_split_dataframe  # noqa: E402
from preprocessing import (  # noqa: E402
    MODEL_FEATURE_COLUMNS,
    TEMPORAL_TARGET_COLUMN,
    load_preprocessor,
    preprocessor_input_dim,
    transform_features,
)
from train_model import (  # noqa: E402
    DEFAULT_DROPOUT,
    PlayerNet,
    apply_probability_calibrator,
    load_model_checkpoint,
    load_split_artifact,
    sigmoid_numpy,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_bootstrap(
    y_true: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    roc_differences = []
    pr_differences = []
    for _ in range(iterations):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        sampled_y = y_true[indices]
        if np.unique(sampled_y).size < 2:
            continue
        first_sample = first[indices]
        second_sample = second[indices]
        roc_differences.append(
            roc_auc_score(sampled_y, first_sample) - roc_auc_score(sampled_y, second_sample)
        )
        pr_differences.append(
            average_precision_score(sampled_y, first_sample)
            - average_precision_score(sampled_y, second_sample)
        )

    def interval(values):
        array = np.asarray(values, dtype=float)
        return {
            "mean_difference": float(array.mean()),
            "ci_95_percentile": [
                float(np.quantile(array, 0.025)),
                float(np.quantile(array, 0.975)),
            ],
        }

    return {
        "comparison": "pytorch_raw_minus_logistic_regression",
        "iterations_requested": iterations,
        "iterations_used": len(roc_differences),
        "seed": seed,
        "roc_auc": interval(roc_differences),
        "pr_auc": interval(pr_differences),
        "method": "paired nonparametric bootstrap over the persisted test split",
    }


def audit(app_dir: Path, iterations: int, bootstrap_seed: int) -> dict:
    metadata_path = app_dir / "training_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cache_artifact = load(app_dir / "temporal_training_dataframe.joblib")
    features_df = cache_artifact["dataframe"]
    splits = load_split_artifact(str(app_dir / "training_splits.json"))

    train_df = _select_split_dataframe(features_df, splits["train_player_ids"])
    validation_df = _select_split_dataframe(features_df, splits["validation_player_ids"])
    test_df = _select_split_dataframe(features_df, splits["test_player_ids"])
    y_train = train_df[TEMPORAL_TARGET_COLUMN].astype(np.float32).to_numpy()
    y_test = test_df[TEMPORAL_TARGET_COLUMN].astype(np.float32).to_numpy()

    preprocessor = load_preprocessor(str(app_dir / "preprocessor.joblib"))
    X_train = transform_features(train_df, preprocessor)
    X_test = transform_features(test_df, preprocessor)
    input_dim = preprocessor_input_dim(preprocessor)
    model = PlayerNet(
        input_dim=input_dim,
        dropout=float(metadata.get("config", {}).get("dropout", DEFAULT_DROPOUT)),
    )
    checkpoint = load_model_checkpoint(
        str(app_dir / "model.pt"), expected_input_dim=input_dim, map_location="cpu"
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with torch.no_grad():
        raw_probability = sigmoid_numpy(
            model(torch.tensor(X_test, dtype=torch.float32)).numpy().reshape(-1)
        )
    calibrator = load(app_dir / "probability_calibrator.joblib")
    calibrated_probability = apply_probability_calibrator(calibrator, raw_probability)

    baseline = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=int(splits["seed"]),
    )
    baseline.fit(X_train, y_train)
    baseline_probability = baseline.predict_proba(X_test)[:, 1]

    split_sets = {
        name: set(map(int, splits[f"{name}_player_ids"]))
        for name in ("train", "validation", "test")
    }
    dataframe_columns = set(features_df.columns)
    forbidden = {
        "potential_label",
        TEMPORAL_TARGET_COLUMN,
        "combined_prob",
        "calibrated_combined_prob",
        "raw_probability",
        "calibrated_probability",
    }
    model_features = set(MODEL_FEATURE_COLUMNS)

    artifacts = [
        "model.pt",
        "preprocessor.joblib",
        "probability_calibrator.joblib",
        "training_metadata.json",
        "training_splits.json",
        "temporal_training_dataframe.joblib",
    ]
    return {
        "scope": {
            "historical_run_timestamp": metadata.get("timestamp"),
            "audit_does_not_retrain_or_overwrite_artifacts": True,
        },
        "model_counts": {
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "all_parameters": sum(p.numel() for p in model.parameters()),
            "buffers": sum(b.numel() for b in model.buffers()),
            "state_dict_elements": sum(value.numel() for value in model.state_dict().values()),
        },
        "features_and_labels": {
            "target_column": TEMPORAL_TARGET_COLUMN,
            "model_feature_count_before_encoding": len(MODEL_FEATURE_COLUMNS),
            "transformed_input_dimension": input_dim,
            "forbidden_columns_present_in_dataframe": sorted(forbidden & dataframe_columns),
            "forbidden_columns_used_as_model_features": sorted(forbidden & model_features),
            "potential_label_is_persisted_but_not_a_model_feature": (
                "potential_label" in dataframe_columns and "potential_label" not in model_features
            ),
        },
        "splits": {
            "seed": int(splits["seed"]),
            "counts": {name: len(values) for name, values in split_sets.items()},
            "overlap": {
                "train_validation": len(split_sets["train"] & split_sets["validation"]),
                "train_test": len(split_sets["train"] & split_sets["test"]),
                "validation_test": len(split_sets["validation"] & split_sets["test"]),
            },
        },
        "probability_semantics": {
            "raw": "sigmoid output from PlayerNet",
            "calibrated": "isotonic transform of the raw probability",
            "combined": "application score that mixes raw probability, history and position fit; no saved test predictions",
            "visual_bands": {"medium": 0.60, "high": 0.80},
            "validation_thresholds": {
                "raw": metadata["pytorch"].get("raw_validation_threshold"),
                "calibrated": metadata["pytorch"].get("selected_threshold"),
                "logistic": metadata["baselines"]["logistic_regression_balanced"].get(
                    "selected_threshold"
                ),
            },
        },
        "test_metrics_recomputed": {
            "pytorch_raw": {
                "roc_auc": float(roc_auc_score(y_test, raw_probability)),
                "pr_auc": float(average_precision_score(y_test, raw_probability)),
            },
            "pytorch_calibrated": {
                "roc_auc": float(roc_auc_score(y_test, calibrated_probability)),
                "pr_auc": float(average_precision_score(y_test, calibrated_probability)),
            },
            "logistic_regression": {
                "roc_auc": float(roc_auc_score(y_test, baseline_probability)),
                "pr_auc": float(average_precision_score(y_test, baseline_probability)),
            },
        },
        "paired_bootstrap": paired_bootstrap(
            y_test,
            raw_probability,
            baseline_probability,
            iterations=iterations,
            seed=bootstrap_seed,
        ),
        "artifact_sha256": {name: sha256(app_dir / name) for name in artifacts},
        "known_historical_gaps": {
            "training_duration": None,
            "library_versions_at_training_time": None,
            "validation_loss_history": None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-dir", type=Path, default=APP_DIR)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261005)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = audit(args.app_dir.resolve(), args.iterations, args.bootstrap_seed)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
