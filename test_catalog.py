"""Teste: descoberta e catalogação de eventos/mercados."""

import json
from pathlib import Path

from polymarket_client import get_event_by_slug, get_yes_token_id, list_events, search_events

OUTPUT_DIR = Path("data/raw/catalog")
EVENT_SLUG = "2026-nba-champion"


def summarize_event(event: dict) -> dict:
    markets = event.get("markets", [])
    return {
        "id": event.get("id"),
        "slug": event.get("slug"),
        "title": event.get("title"),
        "active": event.get("active"),
        "closed": event.get("closed"),
        "volume": event.get("volume"),
        "liquidity": event.get("liquidity"),
        "endDate": event.get("endDate"),
        "markets_count": len(markets),
        "markets": [
            {
                "question": m.get("question"),
                "outcome": m.get("groupItemTitle"),
                "conditionId": m.get("conditionId"),
                "volume": m.get("volume"),
                "liquidity": m.get("liquidity"),
                "yesTokenId": get_yes_token_id(m),
            }
            for m in markets
        ],
    }


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Busca por texto ===")
    search_hits = search_events("NBA champion 2026")
    for event in search_hits:
        print(f"- {event.get('title')} ({event.get('slug')})")

    print("\n=== Evento alvo ===")
    event = get_event_by_slug(EVENT_SLUG)
    summary = summarize_event(event)
    print(f"Título: {summary['title']}")
    print(f"Mercados: {summary['markets_count']}")
    print(f"Volume total do evento: ${float(summary.get('volume') or 0):,.2f}")

    output_path = OUTPUT_DIR / f"{EVENT_SLUG}.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nCatálogo salvo em: {output_path}")

    print("\n=== Eventos com tag 'nba' (amostra) ===")
    nba_events = list_events(tag_slug="nba", limit=5, active=True)
    for item in nba_events:
        print(f"- {item.get('title')} | volume={item.get('volume')}")
