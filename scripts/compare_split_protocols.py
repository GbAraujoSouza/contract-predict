"""Compara modelos com e sem split temporal (mesmos hiperparâmetros do notebook 05)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, TimeSeriesSplit, train_test_split
from sklearn.tree import DecisionTreeClassifier

CUTOFF = datetime(2026, 3, 10, tzinfo=timezone.utc)
MODEL_NAMES = ["LightGBM", "Decision Tree", "Random Forest"]
FEATURES = [
    "close_mid",
    "depth_imbalance",
    "mean_spread",
    "close_spread",
    "bar_volatility",
    "ofi_corrected",
    "close_mid_lag1",
    "depth_imbalance_lag1",
    "mean_spread_lag1",
    "close_spread_lag1",
    "bar_volatility_lag1",
    "ofi_corrected_lag1",
]
LAG_COLS = FEATURES[6:]


def build_v3_from_v2(raw: pl.DataFrame) -> pl.DataFrame:
    df = raw.sort(["market_id", "minute_bar"])
    df = df.with_columns(
        pl.when(pl.col("total_volume") > 0)
        .then((pl.col("buy_volume") - pl.col("sell_volume")) / pl.col("total_volume"))
        .otherwise(0.0)
        .alias("ofi_corrected")
    )
    base = FEATURES[:6]
    return df.with_columns(
        [pl.col(c).shift(1).over("market_id").alias(f"{c}_lag1") for c in base]
    )


def load_xy() -> tuple:
    candidates = [
        Path("features/ml_features_v3.parquet"),
        Path("features/ml_features_1m_v2.parquet"),
        Path("ml_features_1m_v2.parquet"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        raw = pl.read_parquet(path)
        df = raw if "close_mid_lag1" in raw.columns else build_v3_from_v2(raw)
        df = df.drop_nulls(LAG_COLS)
        x = df.select(FEATURES).to_pandas()
        y = df["target"].to_numpy().ravel()
        bars = df["minute_bar"].to_numpy()
        return x, y, bars, df
    raise FileNotFoundError("Parquet de features não encontrado.")


def make_model(name: str):
    if name == "LightGBM":
        return lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
            verbose=-1,
        )
    if name == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=10, random_state=42, class_weight="balanced"
        )
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
    raise ValueError(name)


def eval_holdout(model, x_test, y_test) -> dict:
    pred = model.predict(x_test)
    proba = model.predict_proba(x_test)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, pred)),
    }


def cv_summary(scores: list[float]) -> dict:
    return {"mean": float(np.mean(scores)), "std": float(np.std(scores))}


def protocol_temporal(x, y, bars, *, run_cv: bool = True) -> dict:
    """Holdout por data + TimeSeriesSplit no treino (notebook 05)."""
    cutoff = np.datetime64(CUTOFF.replace(tzinfo=None))
    train_mask = bars < cutoff
    test_mask = ~train_mask
    x_tr, y_tr = x.iloc[train_mask], y[train_mask]
    x_te, y_te = x.iloc[test_mask], y[test_mask]

    out: dict = {"holdout": {}, "cv": {}}
    if run_cv:
        tscv = TimeSeriesSplit(n_splits=5)
        for name in MODEL_NAMES:
            aucs = []
            for tr_idx, va_idx in tscv.split(x_tr):
                m = make_model(name)
                m.fit(x_tr.iloc[tr_idx], y_tr[tr_idx])
                proba = m.predict_proba(x_tr.iloc[va_idx])[:, 1]
                aucs.append(roc_auc_score(y_tr[va_idx], proba))
            out["cv"][name] = cv_summary(aucs)

    for name in MODEL_NAMES:
        m = make_model(name)
        m.fit(x_tr, y_tr)
        out["holdout"][name] = eval_holdout(m, x_te, y_te)
    out["meta"] = {
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "test_period": "2026-03-10 até 2026-03-11",
    }
    return out


def protocol_chrono_8020(x, y) -> dict:
    """Split cronológico 80/20 global (lightdm.ipynb) — sem holdout por data."""
    split_idx = int(len(x) * 0.8)
    x_tr, x_te = x.iloc[:split_idx], x.iloc[split_idx:]
    y_tr, y_te = y[:split_idx], y[split_idx:]

    out: dict = {"holdout": {}, "cv": {}}
    for name in MODEL_NAMES:
        m = make_model(name)
        m.fit(x_tr, y_tr)
        out["holdout"][name] = eval_holdout(m, x_te, y_te)
    out["meta"] = {
        "train_rows": split_idx,
        "test_rows": len(x) - split_idx,
        "test_period": "últimos 20% das linhas (cronológico global)",
    }
    return out


def protocol_random_shuffle(x, y) -> dict:
    """Split aleatório 80/20 — vaza informação temporal entre treino e teste."""
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.2, shuffle=True, random_state=42, stratify=y
    )
    out: dict = {"holdout": {}, "cv": {}}
    for name in MODEL_NAMES:
        m = make_model(name)
        m.fit(x_tr, y_tr)
        out["holdout"][name] = eval_holdout(m, x_te, y_te)
    out["meta"] = {
        "train_rows": len(x_tr),
        "test_rows": len(x_te),
        "test_period": "20% aleatório (stratified)",
    }
    return out


def main() -> None:
    x, y, bars, _ = load_xy()
    report = {
        "com_split_temporal": protocol_temporal(x, y, bars),
        "sem_split_cronologico_8020": protocol_chrono_8020(x, y),
        "sem_split_aleatorio": protocol_random_shuffle(x, y),
    }
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "split_comparison.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Relatório salvo em {out_path}")

    print("\n=== HOLDOUT ROC-AUC ===")
    for protocol, label in [
        ("com_split_temporal", "Com split temporal (holdout por data)"),
        ("sem_split_cronologico_8020", "Sem split temporal (80/20 cronológico)"),
        ("sem_split_aleatorio", "Sem split temporal (80/20 aleatório)"),
    ]:
        print(f"\n{label}:")
        for name in MODEL_NAMES:
            h = report[protocol]["holdout"][name]
            print(f"  {name:16s} AUC={h['roc_auc']:.4f}  F1={h['f1']:.4f}")


if __name__ == "__main__":
    main()
