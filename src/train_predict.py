from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

RANDOM_SEEDS = [42, 123, 777]
NORMALIZATION_OFFSET = 200.0
CAT_FEATURES = ["pickup", "delivery", "equipment", "route"]


def add_features(frame: pd.DataFrame, *, reduced: bool = False) -> pd.DataFrame:
    """Feature engineering shared by validation and final training.

    reduced=True is used for the fixed December chart because that input file does not
    contain coordinates, market_index, or quote_signal.
    """
    x = frame.copy()
    x["date"] = pd.to_datetime(x["date"], errors="raise")
    x["route"] = x["pickup"].astype(str) + "__" + x["delivery"].astype(str)

    dt = x.pop("date")
    x["day_of_year"] = dt.dt.dayofyear.astype(int)
    x["day_of_week"] = dt.dt.dayofweek.astype(int)
    x["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    x["month"] = dt.dt.month.astype(int)
    x["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    x["sin_day_of_year"] = np.sin(2 * np.pi * x["day_of_year"] / 365.25)
    x["cos_day_of_year"] = np.cos(2 * np.pi * x["day_of_year"] / 365.25)
    x["sin_day_of_week"] = np.sin(2 * np.pi * x["day_of_week"] / 7.0)
    x["cos_day_of_week"] = np.cos(2 * np.pi * x["day_of_week"] / 7.0)
    x["log_distance"] = np.log1p(x["distance"].astype(float))
    x["sqrt_distance"] = np.sqrt(x["distance"].astype(float).clip(lower=0))
    x["weight_per_mile"] = x["weight"].astype(float) / x["distance"].astype(float).replace(0, np.nan)

    if not reduced:
        x["lat_diff"] = x["delivery_lat"] - x["pickup_lat"]
        x["lon_diff"] = x["delivery_lon"] - x["pickup_lon"]
        x["abs_lat_diff"] = x["lat_diff"].abs()
        x["abs_lon_diff"] = x["lon_diff"].abs()
        x["mid_lat"] = (x["delivery_lat"] + x["pickup_lat"]) / 2.0
        x["mid_lon"] = (x["delivery_lon"] + x["pickup_lon"]) / 2.0
        x["distance_x_market"] = x["distance"] * x["market_index"]
        x["distance_x_quote"] = x["distance"] * x["quote_signal"]
        x["market_x_quote"] = x["market_index"] * x["quote_signal"]

    return x


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


def make_model(seed: int, iterations: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=iterations,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=seed,
        l2_leaf_reg=6,
        random_strength=0.15,
        bagging_temperature=0.5,
        verbose=False,
        allow_writing_files=False,
    )


def fit_temporal_ensemble(train: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[list[int], dict[str, float]]:
    dev = train[train["date"] < cutoff].copy()
    holdout = train[train["date"] >= cutoff].copy()

    x_dev = add_features(dev.drop(columns=["load_id", "posted_rate"]), reduced=False)
    x_holdout = add_features(holdout.drop(columns=["load_id", "posted_rate"]), reduced=False)

    y_dev = dev["posted_rate"].to_numpy() / (dev["distance"].to_numpy() + NORMALIZATION_OFFSET)
    y_holdout_norm = holdout["posted_rate"].to_numpy() / (
        holdout["distance"].to_numpy() + NORMALIZATION_OFFSET
    )

    iteration_counts: list[int] = []
    holdout_predictions: list[np.ndarray] = []

    for seed in RANDOM_SEEDS:
        model = make_model(seed, iterations=600)
        model.fit(
            x_dev,
            y_dev,
            cat_features=CAT_FEATURES,
            eval_set=(x_holdout, y_holdout_norm),
            early_stopping_rounds=80,
            verbose=False,
        )
        best_iter = max(60, int(model.get_best_iteration()) + 1)
        iteration_counts.append(best_iter)
        pred = model.predict(x_holdout) * (holdout["distance"].to_numpy() + NORMALIZATION_OFFSET)
        holdout_predictions.append(pred)

    ensemble_pred = np.mean(holdout_predictions, axis=0)
    return iteration_counts, metrics(holdout["posted_rate"].to_numpy(), ensemble_pred)


def fit_final_ensemble(train: pd.DataFrame, iterations: list[int], model_dir: Path) -> list[CatBoostRegressor]:
    x = add_features(train.drop(columns=["load_id", "posted_rate"]), reduced=False)
    y = train["posted_rate"].to_numpy() / (train["distance"].to_numpy() + NORMALIZATION_OFFSET)

    models: list[CatBoostRegressor] = []
    model_dir.mkdir(parents=True, exist_ok=True)
    for seed, n_iter in zip(RANDOM_SEEDS, iterations, strict=True):
        model = make_model(seed, iterations=n_iter)
        model.fit(x, y, cat_features=CAT_FEATURES, verbose=False)
        model.save_model(model_dir / f"catboost_seed_{seed}.cbm")
        models.append(model)
    return models


def predict_ensemble(models: list[CatBoostRegressor], frame: pd.DataFrame) -> np.ndarray:
    x = add_features(frame.drop(columns=["load_id"]), reduced=False)
    scale = frame["distance"].to_numpy() + NORMALIZATION_OFFSET
    predictions = [model.predict(x) * scale for model in models]
    return np.mean(predictions, axis=0)


def fit_december_model(train: pd.DataFrame) -> CatBoostRegressor:
    """Train a reduced-feature model for the December chart input schema.

    The provided chart file intentionally omits market_index, quote_signal and coordinates,
    so a separate model is used rather than fabricating those values.
    """
    x = add_features(
        train[["pickup", "delivery", "distance", "equipment", "weight", "date"]],
        reduced=True,
    )
    y = train["posted_rate"].to_numpy() / (train["distance"].to_numpy() + NORMALIZATION_OFFSET)
    model = make_model(seed=123, iterations=260)
    model.fit(x, y, cat_features=CAT_FEATURES, verbose=False)
    return model


def baseline_metrics(train: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, float]:
    dev = train[train["date"] < cutoff].copy()
    holdout = train[train["date"] >= cutoff].copy()
    # Simple benchmark: median rate per mile by equipment, learned only from historical rows.
    rpm_by_equipment = (dev["posted_rate"] / dev["distance"]).groupby(dev["equipment"]).median()
    global_rpm = float((dev["posted_rate"] / dev["distance"]).median())
    rpm = holdout["equipment"].map(rpm_by_equipment).fillna(global_rpm).to_numpy()
    pred = rpm * holdout["distance"].to_numpy()
    return metrics(holdout["posted_rate"].to_numpy(), pred)


def run(args: argparse.Namespace) -> None:
    project = Path(args.project_dir).resolve()
    data_dir = project / "data"
    output_dir = project / "outputs"
    model_dir = project / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_dir / "train_test.csv")
    validation = pd.read_csv(data_dir / "validation.csv")
    template = pd.read_csv(data_dir / "validation_predictions_template.csv")
    december = pd.read_csv(data_dir / "december_chart_inputs.csv")

    train["date"] = pd.to_datetime(train["date"])
    validation["date"] = pd.to_datetime(validation["date"])

    cutoff = pd.Timestamp("2025-09-01")
    baseline = baseline_metrics(train, cutoff)
    iteration_counts, validation_metrics = fit_temporal_ensemble(train, cutoff)

    models = fit_final_ensemble(train, iteration_counts, model_dir)
    final_pred = np.clip(predict_ensemble(models, validation), 1.0, None)

    pred_frame = pd.DataFrame({"load_id": validation["load_id"], "predicted_rate": final_pred})
    completed = template[["load_id"]].merge(pred_frame, on="load_id", how="left", validate="one_to_one")
    if completed["predicted_rate"].isna().any():
        raise RuntimeError("Some validation load_id values did not receive predictions")
    validation_out = project / "validation_predictions.csv"
    completed.to_csv(validation_out, index=False, float_format="%.2f")

    december_model = fit_december_model(train)
    dec_features = add_features(
        december[["pickup", "delivery", "distance", "equipment", "weight", "date"]], reduced=True
    )
    dec_pred = december_model.predict(dec_features) * (december["distance"].to_numpy() + NORMALIZATION_OFFSET)
    december["predicted_rate"] = np.clip(dec_pred, 1.0, None)
    december_out = data_dir / "december_chart_inputs.csv"
    december.to_csv(december_out, index=False, float_format="%.2f")

    audit = {
        "train_shape": list(train.shape),
        "validation_shape": list(validation.shape),
        "train_date_min": str(train["date"].min().date()),
        "train_date_max": str(train["date"].max().date()),
        "validation_date_min": str(validation["date"].min().date()),
        "validation_date_max": str(validation["date"].max().date()),
        "train_missing": {k: int(v) for k, v in train.isna().sum().items() if int(v) > 0},
        "validation_missing": {k: int(v) for k, v in validation.isna().sum().items() if int(v) > 0},
        "unseen_pickup_cities": sorted(set(validation["pickup"]) - set(train["pickup"])),
        "unseen_delivery_cities": sorted(set(validation["delivery"]) - set(train["delivery"])),
        "temporal_split": {
            "train": "2025-01-01 through 2025-08-31",
            "holdout": "2025-09-01 through 2025-10-31",
        },
        "baseline_metrics": baseline,
        "catboost_ensemble_metrics": validation_metrics,
        "final_iterations_by_seed": dict(zip(map(str, RANDOM_SEEDS), iteration_counts, strict=True)),
        "normalization_offset": NORMALIZATION_OFFSET,
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)

    if args.run_scorer:
        cmd = [
            sys.executable,
            str(project / "score.py"),
            "--predictions",
            str(validation_out),
            "--december-predictions",
            str(december_out),
            "--output-dir",
            str(project / "scorer_results"),
        ]
        subprocess.run(cmd, check=True)

    print("Temporal holdout metrics:")
    print(json.dumps(validation_metrics, indent=2))
    print(f"Saved: {validation_out}")
    print(f"Saved: {december_out}")
    print(f"Saved: {output_dir / 'run_summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--run-scorer", action="store_true")
    run(parser.parse_args())
