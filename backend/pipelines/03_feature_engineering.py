"""
Etapa 03 — Engenharia de features e preparação do dataset modelável.

Lê features/ml_features_1m_v2.parquet (Kaggle), aplica filtros de qualidade,
corrige OFI normalizado e salva data/processed/model_dataset.parquet.

Por padrão mantém minutos sem negociação: volume=0 não é dado ausente, é regime
de baixa atividade (ex.: madrugada UTC). Use --drop-inactive para variantes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = REPO_ROOT / "features" / "ml_features_1m_v2.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "model_dataset.parquet"
DEFAULT_SUMMARY = REPO_ROOT / "data" / "processed" / "feature_engineering_summary.json"

MIN_MARKET_OBSERVATIONS = 1440

BASE_FEATURE_COLUMNS = [
    "close_mid",
    "mean_spread",
    "close_spread",
    "bar_volatility",
    "total_volume",
    "buy_volume",
    "sell_volume",
    "trade_count",
    "ofi_normalized",
    "return_1m",
    "bid_depth",
    "ask_depth",
    "depth_imbalance",
]

DERIVED_FEATURE_COLUMNS = [
    "has_trade",
    "hour_utc",
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + DERIVED_FEATURE_COLUMNS
METADATA_COLUMNS = ["market_id", "minute_bar", "target"]


def add_ofi_normalized(df: pl.DataFrame) -> pl.DataFrame:
    """OFI normalizado: (buy - sell) / total_volume; 0 quando não há negociação."""
    return df.with_columns(
        pl.when(pl.col("total_volume") > 0)
        .then((pl.col("buy_volume") - pl.col("sell_volume")) / pl.col("total_volume"))
        .otherwise(0.0)
        .alias("ofi_normalized")
    )


def add_activity_features(df: pl.DataFrame) -> pl.DataFrame:
    """Codifica inatividade e horário como sinal explícito para o modelo."""
    return df.with_columns(
        ((pl.col("total_volume") > 0) | (pl.col("trade_count") > 0))
        .cast(pl.Int8)
        .alias("has_trade"),
        pl.col("minute_bar").dt.hour().cast(pl.Int8).alias("hour_utc"),
    )


def filter_active_minutes(df: pl.DataFrame) -> pl.DataFrame:
    """Remove barras sem negociação (variante experimental)."""
    return df.filter(pl.col("has_trade") == 1)


def filter_markets_by_history(
    df: pl.DataFrame, min_observations: int = MIN_MARKET_OBSERVATIONS
) -> pl.DataFrame:
    """Mantém mercados com histórico mínimo (>= 1440 barras totais, incluindo inativas)."""
    market_counts = df.group_by("market_id").len().filter(pl.col("len") >= min_observations)
    return df.join(market_counts.select("market_id"), on="market_id", how="inner")


def build_model_dataset(
    df: pl.DataFrame,
    *,
    min_market_observations: int = MIN_MARKET_OBSERVATIONS,
    filter_short_markets: bool = True,
    drop_inactive: bool = False,
) -> pl.DataFrame:
    """Pipeline completo de feature engineering."""
    if filter_short_markets:
        df = filter_markets_by_history(df, min_observations=min_market_observations)

    df = add_ofi_normalized(df)
    df = add_activity_features(df)

    if drop_inactive:
        df = filter_active_minutes(df)

    output_columns = METADATA_COLUMNS + FEATURE_COLUMNS
    missing = [col for col in output_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no dataset de entrada: {missing}")

    return df.select(output_columns)


def summarize(df: pl.DataFrame, label: str) -> dict:
    target_stats = (
        df.group_by("target")
        .len()
        .sort("target")
        .with_columns((pl.col("len") / df.height).alias("proportion"))
    )
    inactive_pct = None
    if "has_trade" in df.columns:
        inactive_pct = float((df["has_trade"] == 0).mean())

    return {
        "label": label,
        "rows": df.height,
        "markets": df.select(pl.col("market_id").n_unique()).item(),
        "inactive_pct": inactive_pct,
        "date_min": str(df.select(pl.col("minute_bar").min()).item()),
        "date_max": str(df.select(pl.col("minute_bar").max()).item()),
        "target_distribution": target_stats.to_dicts(),
    }


def run(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY,
    *,
    min_market_observations: int = MIN_MARKET_OBSERVATIONS,
    filter_short_markets: bool = True,
    drop_inactive: bool = False,
) -> pl.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Arquivo de entrada não encontrado: {input_path}\n"
            "Baixe o dataset Kaggle e coloque em features/ml_features_1m_v2.parquet"
        )

    print(f"Carregando {input_path} ...")
    raw = pl.read_parquet(input_path)
    print(f"  Linhas brutas: {raw.height:,} | Mercados: {raw['market_id'].n_unique():,}")

    after_history = (
        filter_markets_by_history(raw, min_observations=min_market_observations)
        if filter_short_markets
        else raw
    )
    after_features = add_activity_features(add_ofi_normalized(after_history))
    model_df = build_model_dataset(
        raw,
        min_market_observations=min_market_observations,
        filter_short_markets=filter_short_markets,
        drop_inactive=drop_inactive,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_df.write_parquet(output_path)

    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "min_market_observations": min_market_observations,
        "filter_short_markets": filter_short_markets,
        "drop_inactive": drop_inactive,
        "feature_columns": FEATURE_COLUMNS,
        "metadata_columns": METADATA_COLUMNS,
        "stages": [
            summarize(raw, "raw"),
            summarize(after_history, "after_market_history_filter"),
            summarize(after_features, "after_feature_engineering"),
            summarize(model_df, "final"),
        ],
        "rows_removed": {
            "short_markets": raw.height - after_history.height,
            "inactive_minutes": 0 if not drop_inactive else after_features.height - model_df.height,
            "total": raw.height - model_df.height,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDataset modelavel salvo em {output_path}")
    print(f"  Linhas finais: {model_df.height:,}")
    print(f"  Mercados finais: {model_df['market_id'].n_unique():,}")
    print(f"  drop_inactive: {drop_inactive}")
    print(f"  Resumo: {summary_path}")
    return model_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature engineering — Polymarket ML pipeline")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--min-market-observations",
        type=int,
        default=MIN_MARKET_OBSERVATIONS,
        help="Minimo de barras de 1 min por mercado (padrao: 1440 ≈ 1 dia)",
    )
    parser.add_argument(
        "--keep-short-markets",
        action="store_true",
        help="Nao filtrar mercados com historico curto",
    )
    parser.add_argument(
        "--drop-inactive",
        action="store_true",
        help="Remover minutos sem negociacao (variante active-only para comparacao)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        input_path=args.input,
        output_path=args.output,
        summary_path=args.summary,
        min_market_observations=args.min_market_observations,
        filter_short_markets=not args.keep_short_markets,
        drop_inactive=args.drop_inactive,
    )
