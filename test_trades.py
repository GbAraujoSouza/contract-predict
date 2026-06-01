"""Teste: coleta de trades individuais (microestrutura + baleias)."""

import json
from collections import Counter
from pathlib import Path

from polymarket_client import get_event_by_slug, get_market_for_team, iter_trades

OUTPUT_DIR = Path("data/raw/trades")
EVENT_SLUG = "2026-nba-champion"
TEAM = "Oklahoma City Thunder"
WHALE_MIN_USD = 1000
MAX_PAGES = 5


def trade_notional(trade: dict) -> float:
    return float(trade["size"]) * float(trade["price"])


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    event = get_event_by_slug(EVENT_SLUG)
    market = get_market_for_team(event, TEAM)
    condition_id = market["conditionId"]

    print(f"Mercado: {market['question']}")
    print(f"conditionId: {condition_id}")
    print()

    print("=== Trades recentes (todas) ===")
    all_trades = list(
        iter_trades(
            condition_ids=[condition_id],
            page_size=500,
            max_pages=MAX_PAGES,
        )
    )
    print(f"Trades coletados: {len(all_trades)}")
    if all_trades:
        notionals = [trade_notional(t) for t in all_trades]
        print(f"Notional médio: ${sum(notionals)/len(notionals):,.2f}")
        print(f"Maior trade: ${max(notionals):,.2f}")
        print(f"Exemplo: {all_trades[0]}")

    print("\n=== Trades filtradas (baleias >= $1000) ===")
    whale_trades = list(
        iter_trades(
            condition_ids=[condition_id],
            page_size=100,
            max_pages=3,
            min_cash_usd=WHALE_MIN_USD,
        )
    )
    print(f"Baleias encontradas: {len(whale_trades)}")

    wallets = Counter(t["proxyWallet"] for t in whale_trades)
    print("Top wallets em trades grandes:")
    for wallet, count in wallets.most_common(5):
        pseudonym = next(t["pseudonym"] for t in whale_trades if t["proxyWallet"] == wallet)
        print(f"  {wallet[:10]}... ({pseudonym}): {count} trades")

    payload = {
        "event_slug": EVENT_SLUG,
        "team": TEAM,
        "condition_id": condition_id,
        "sample_trades": all_trades[:20],
        "whale_trades": whale_trades,
    }
    output_path = OUTPUT_DIR / f"{EVENT_SLUG}_{TEAM.replace(' ', '_').lower()}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nAmostra salva em: {output_path}")
