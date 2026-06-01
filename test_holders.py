"""Teste: holders/top holders (identificação de baleias por posição)."""

import json
from pathlib import Path

from polymarket_client import (
    fetch_holders,
    fetch_open_interest,
    get_event_by_slug,
    get_market_for_team,
)

OUTPUT_DIR = Path("data/raw/holders")
EVENT_SLUG = "2026-nba-champion"
TEAM = "Oklahoma City Thunder"


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    event = get_event_by_slug(EVENT_SLUG)
    market = get_market_for_team(event, TEAM)
    condition_id = market["conditionId"]

    print(f"Mercado: {market['question']}")
    print(f"conditionId: {condition_id}")
    print()

    holders_by_token = fetch_holders(condition_id, limit=20)
    oi = fetch_open_interest(condition_id)

    print("=== Open Interest ===")
    for item in oi:
        print(f"  OI: ${float(item.get('value', 0)):,.2f}")

    print("\n=== Top holders por token ===")
    snapshot = []
    for group in holders_by_token:
        token = group["token"]
        holders = group["holders"]
        print(f"\nToken {token[:16]}... ({len(holders)} holders)")
        for holder in holders[:5]:
            print(
                f"  {holder.get('name') or holder.get('pseudonym')}: "
                f"{holder['amount']:,.2f} tokens | wallet={holder['proxyWallet'][:10]}..."
            )
        snapshot.append({"token": token, "holders": holders})

    output_path = OUTPUT_DIR / f"{EVENT_SLUG}_{TEAM.replace(' ', '_').lower()}_holders.json"
    output_path.write_text(
        json.dumps(
            {
                "event_slug": EVENT_SLUG,
                "team": TEAM,
                "condition_id": condition_id,
                "open_interest": oi,
                "holders_by_token": snapshot,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSnapshot salvo em: {output_path}")
