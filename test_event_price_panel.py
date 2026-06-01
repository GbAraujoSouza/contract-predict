"""Teste: painel temporal multi-mercado para análise de correlação."""

import json
from pathlib import Path

import pandas as pd

from polymarket_client import (
    fetch_batch_price_history,
    get_event_by_slug,
    get_yes_token_id,
)

OUTPUT_DIR = Path("data/raw/panels")
EVENT_SLUG = "2026-nba-champion"
BATCH_SIZE = 20
FIDELITY = 60


def build_price_panel(event: dict) -> pd.DataFrame:
    markets = event["markets"]
    rows: list[dict] = []

    for start in range(0, len(markets), BATCH_SIZE):
        chunk = markets[start : start + BATCH_SIZE]
        token_map = {get_yes_token_id(m): m.get("groupItemTitle") or m.get("question") for m in chunk}
        histories = fetch_batch_price_history(
            list(token_map.keys()),
            interval="max",
            fidelity=FIDELITY,
        )

        for token_id, history in histories.items():
            outcome = token_map[token_id]
            for point in history:
                rows.append(
                    {
                        "timestamp": point["t"],
                        "outcome": outcome,
                        "token_id": token_id,
                        "price": point["p"],
                    }
                )

    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel

    panel["datetime"] = pd.to_datetime(panel["timestamp"], unit="s", utc=True)
    return panel.sort_values(["timestamp", "outcome"]).reset_index(drop=True)


def correlation_preview(panel: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    wide = panel.pivot_table(index="timestamp", columns="outcome", values="price", aggfunc="last")
    top_outcomes = (
        panel.groupby("outcome")["price"]
        .count()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )
    return wide[top_outcomes].corr()


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    event = get_event_by_slug(EVENT_SLUG)
    print(f"Evento: {event['title']}")
    print(f"Mercados: {len(event['markets'])}")
    print("Coletando histórico em lotes de 20 tokens...")
    print()

    panel = build_price_panel(event)
    print(f"Linhas no painel: {len(panel):,}")
    print(f"Outcomes únicos: {panel['outcome'].nunique()}")
    print(f"Período: {panel['datetime'].min()} -> {panel['datetime'].max()}")
    print()

    corr = correlation_preview(panel)
    print("=== Correlação (top 8 outcomes por cobertura temporal) ===")
    print(corr.round(3).to_string())
    print()

    csv_path = OUTPUT_DIR / f"{EVENT_SLUG}_price_panel.csv"
    corr_path = OUTPUT_DIR / f"{EVENT_SLUG}_correlation_preview.csv"
    panel.to_csv(csv_path, index=False)
    corr.to_csv(corr_path)
    print(f"Painel salvo em: {csv_path}")
    print(f"Correlação salva em: {corr_path}")

    meta_path = OUTPUT_DIR / f"{EVENT_SLUG}_panel_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "event_slug": EVENT_SLUG,
                "markets_count": len(event["markets"]),
                "rows": len(panel),
                "outcomes": int(panel["outcome"].nunique()),
                "start": str(panel["datetime"].min()),
                "end": str(panel["datetime"].max()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
