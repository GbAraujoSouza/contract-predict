from polymarket_client import (
    fetch_price_history,
    get_event_by_slug,
    get_market_for_team,
    get_yes_token_id,
)

EVENT_SLUG = "2026-nba-champion"
TEAM = "Oklahoma City Thunder"


if __name__ == "__main__":
    event = get_event_by_slug(EVENT_SLUG)
    market = get_market_for_team(event, TEAM)
    token_id = get_yes_token_id(market)

    print(f"Evento: {EVENT_SLUG}")
    print(f"Pergunta: {market['question']}")
    print(f"Token ID (Yes): {token_id}")
    print(f"Volume: ${float(market.get('volume') or 0):,.2f}")
    print()

    history = fetch_price_history(token_id)

    print("Status: OK")
    print(f"Pontos de histórico: {len(history)}")

    if history:
        print(f"Primeiro: t={history[0]['t']}, preço={history[0]['p']:.4f}")
        print(f"Último:   t={history[-1]['t']}, preço={history[-1]['p']:.4f}")
        print()
        print("Amostra (5 primeiros pontos):")
        for point in history[:5]:
            print(f"  {point}")
