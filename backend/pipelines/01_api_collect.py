"""
Coleta barras de 1 minuto via API Polymarket e avalia viabilidade de estender o dataset.

A API pública NÃO fornece histórico de orderbook (spread, depth). Este pipeline
monta as colunas disponíveis e documenta lacunas em relação ao parquet Kaggle.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from polymarket_client import (
    DEFAULT_SLEEP,
    discover_liquid_markets,
    fetch_price_history,
    fetch_trades_page,
    get_market_by_condition_id,
    get_yes_token_id,
    parse_clob_token_ids,
)

KAGGLE_COLUMNS = [
    "market_id",
    "minute_bar",
    "close_mid",
    "mean_spread",
    "close_spread",
    "bar_volatility",
    "total_volume",
    "buy_volume",
    "sell_volume",
    "trade_count",
    "order_flow_imbalance",
    "target",
    "return_1m",
    "bid_depth",
    "ask_depth",
    "depth_imbalance",
]

API_ONLY_COLUMNS = [
    "market_id",
    "minute_bar",
    "close_mid",
    "bar_volatility",
    "total_volume",
    "buy_volume",
    "sell_volume",
    "trade_count",
    "return_1m",
    "target",
]

MISSING_FROM_API = [
    "mean_spread",
    "close_spread",
    "bid_depth",
    "ask_depth",
    "depth_imbalance",
    "order_flow_imbalance",
]


def _minute_ts(ts: int) -> datetime:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.replace(second=0, microsecond=0)


def price_bars(token_id: str, start_ts: int | None, end_ts: int | None) -> dict[datetime, list[float]]:
    # interval=max + startTs/endTs costuma retornar 400; filtramos localmente.
    history = fetch_price_history(token_id, fidelity=60)
    buckets: dict[datetime, list[float]] = defaultdict(list)
    for point in history:
        ts = point["t"]
        if start_ts and ts < start_ts:
            continue
        if end_ts and ts > end_ts:
            continue
        buckets[_minute_ts(ts)].append(float(point["p"]))
    return buckets


def trade_bars(
    condition_id: str, start_ts: int | None, end_ts: int | None, max_pages: int = 100
) -> dict[datetime, dict]:
    buckets: dict[datetime, dict] = defaultdict(
        lambda: {"buy_volume": 0.0, "sell_volume": 0.0, "trade_count": 0}
    )
    offset = 0
    page_size = 1000
    pages = 0

    while pages < max_pages and offset <= 10_000:
        batch = fetch_trades_page(
            condition_ids=[condition_id], limit=page_size, offset=offset
        )
        if not batch:
            break

        stop = False
        for trade in batch:
            ts = int(trade["timestamp"])
            if end_ts and ts > end_ts:
                continue
            if start_ts and ts < start_ts:
                stop = True
                break
            minute = _minute_ts(ts)
            size = float(trade.get("size") or 0)
            side = (trade.get("side") or "").upper()
            if side == "BUY":
                buckets[minute]["buy_volume"] += size
            elif side == "SELL":
                buckets[minute]["sell_volume"] += size
            buckets[minute]["trade_count"] += 1

        if stop or len(batch) < page_size:
            break
        offset += page_size
        pages += 1

    for minute, vals in buckets.items():
        vals["total_volume"] = vals["buy_volume"] + vals["sell_volume"]
    return buckets


def build_market_minute_bars(
    condition_id: str,
    token_id: str,
    *,
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> pl.DataFrame:
    prices = price_bars(token_id, start_ts, end_ts)
    trades = trade_bars(condition_id, start_ts, end_ts)
    minutes = sorted(set(prices) | set(trades))

    rows: list[dict] = []
    prev_close: float | None = None
    for minute in minutes:
        pts = prices.get(minute, [])
        close_mid = pts[-1] if pts else None
        if close_mid is None:
            continue

        vol = trades.get(minute, {})
        bar_vol = float(pl.Series(pts).std()) if len(pts) > 1 else 0.0
        ret = (close_mid - prev_close) / prev_close if prev_close else 0.0
        prev_close = close_mid

        rows.append(
            {
                "market_id": condition_id,
                "minute_bar": minute,
                "close_mid": close_mid,
                "bar_volatility": bar_vol if bar_vol == bar_vol else 0.0,
                "total_volume": vol.get("total_volume", 0.0),
                "buy_volume": vol.get("buy_volume", 0.0),
                "sell_volume": vol.get("sell_volume", 0.0),
                "trade_count": vol.get("trade_count", 0),
                "return_1m": ret,
            }
        )

    if not rows:
        return pl.DataFrame()

    df = pl.DataFrame(rows).sort("minute_bar")
    df = df.with_columns(
        (pl.col("close_mid").shift(-15).over("market_id") > pl.col("close_mid"))
        .cast(pl.Int8)
        .alias("target")
    )
    return df


def load_existing_market_ids(parquet_path: Path, limit: int | None = None) -> list[str]:
    df = pl.read_parquet(parquet_path, columns=["market_id"])
    ids = df["market_id"].unique().to_list()
    if limit:
        return ids[:limit]
    return ids


def collect_from_catalog(
    markets: list[dict],
    *,
    start_ts: int | None,
    end_ts: int | None,
) -> tuple[pl.DataFrame, list[str]]:
    frames: list[pl.DataFrame] = []
    errors: list[str] = []

    for i, market in enumerate(markets, 1):
        condition_id = market.get("conditionId") or market.get("condition_id")
        if not condition_id:
            continue
        try:
            token_id = get_yes_token_id(market)
            df = build_market_minute_bars(
                condition_id, token_id, start_ts=start_ts, end_ts=end_ts
            )
            if df.height:
                df = df.with_columns(
                    pl.lit(market.get("question", "")).alias("question"),
                    pl.lit(float(market.get("volume") or 0)).alias("market_volume"),
                )
                frames.append(df)
        except Exception as exc:  # noqa: BLE001 — log por mercado
            errors.append(f"{condition_id}: {exc}")

        if i % 10 == 0:
            print(f"  processados {i}/{len(markets)} mercados...")

    if not frames:
        return pl.DataFrame(), errors

    return pl.concat(frames, how="diagonal_relaxed"), errors


def feasibility_report(api_df: pl.DataFrame, kaggle_path: Path) -> dict:
    kaggle = pl.read_parquet(kaggle_path, columns=["minute_bar"])
    report = {
        "full_schema_reconstructible": False,
        "missing_columns": MISSING_FROM_API,
        "api_columns_built": API_ONLY_COLUMNS,
        "kaggle_required": KAGGLE_COLUMNS,
        "reason": (
            "A API pública não expõe histórico de orderbook (spread, bid/ask depth). "
            "Sem essas colunas o parquet v2/v3 não pode ser reproduzido integralmente."
        ),
    }
    if api_df.is_empty():
        report["api_rows"] = 0
        return report

    report["api_rows"] = api_df.height
    report["api_markets"] = api_df["market_id"].n_unique()
    report["api_date_min"] = str(api_df["minute_bar"].min())
    report["api_date_max"] = str(api_df["minute_bar"].max())
    report["kaggle_date_min"] = str(kaggle["minute_bar"].min())
    report["kaggle_date_max"] = str(kaggle["minute_bar"].max())
    return report


def run(
    *,
    mode: str = "discover",
    parquet_path: Path = Path("features/ml_features_1m_v2.parquet"),
    output_dir: Path = Path("data/raw/api_collection"),
    max_markets: int = 50,
    min_volume: float = 250_000.0,
    start: str | None = None,
    end: str | None = None,
    sample_existing: int = 20,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    start_ts = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp()) if start else None
    end_ts = int(datetime.fromisoformat(end).replace(tzinfo=timezone.utc).timestamp()) if end else None

    if mode == "discover":
        print(f"Descobrindo até {max_markets} mercados com volume >= ${min_volume:,.0f}...")
        markets = discover_liquid_markets(
            min_volume=min_volume, max_markets=max_markets, active=True
        )
        catalog_path = output_dir / "liquid_markets_catalog.json"
        catalog_path.write_text(json.dumps(markets, indent=2), encoding="utf-8")
        print(f"Catálogo salvo: {catalog_path} ({len(markets)} mercados)")
    elif mode == "extend_existing":
        ids = load_existing_market_ids(parquet_path, limit=sample_existing)
        markets = []
        for cid in ids:
            m = get_market_by_condition_id(cid)
            if m:
                markets.append(m)
        print(f"Resolvendo {len(markets)} mercados do parquet via Gamma API...")
    else:
        raise ValueError(f"Modo desconhecido: {mode}")

    print("Coletando barras de 1 min (preço + trades)...")
    api_df, errors = collect_from_catalog(markets, start_ts=start_ts, end_ts=end_ts)

    if api_df.is_empty():
        print("Nenhuma barra coletada.")
    else:
        out_parquet = output_dir / "api_minute_bars.parquet"
        api_df.write_parquet(out_parquet)
        print(f"Parquet parcial salvo: {out_parquet} ({api_df.height:,} linhas)")

    report = feasibility_report(api_df if not api_df.is_empty() else pl.DataFrame(), parquet_path)
    report["collection_errors"] = errors[:20]
    report["markets_requested"] = len(markets)
    report_path = output_dir / "feasibility_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Relatório de viabilidade: {report_path}")
    print(f"Schema completo reconstruível: {report['full_schema_reconstructible']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Coleta API Polymarket (schema parcial)")
    parser.add_argument(
        "--mode",
        choices=["discover", "extend_existing"],
        default="discover",
    )
    parser.add_argument("--max-markets", type=int, default=30)
    parser.add_argument("--min-volume", type=float, default=250_000.0)
    parser.add_argument("--start", help="ISO date início, ex: 2026-03-06")
    parser.add_argument("--end", help="ISO date fim, ex: 2026-06-26")
    parser.add_argument("--sample-existing", type=int, default=15)
    args = parser.parse_args()

    run(
        mode=args.mode,
        max_markets=args.max_markets,
        min_volume=args.min_volume,
        start=args.start,
        end=args.end,
        sample_existing=args.sample_existing,
    )


if __name__ == "__main__":
    main()
