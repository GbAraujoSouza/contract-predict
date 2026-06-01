"""Cliente mínimo para as APIs públicas da Polymarket."""

from __future__ import annotations

import json
import time
from typing import Any, Iterator

import requests

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
DATA_API = "https://data-api.polymarket.com"

DEFAULT_TIMEOUT = 30
DEFAULT_SLEEP = 0.2


def _get(url: str, params: dict | None = None) -> Any:
    response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json()


def _post(url: str, payload: dict) -> Any:
    response = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json()


def parse_clob_token_ids(market: dict) -> list[str]:
    token_ids = market["clobTokenIds"]
    if isinstance(token_ids, str):
        token_ids = json.loads(token_ids)
    return token_ids


def get_yes_token_id(market: dict) -> str:
    return parse_clob_token_ids(market)[0]


def search_events(query: str, limit_per_type: int = 5) -> list[dict]:
    data = _get(f"{GAMMA_API}/public-search", {"q": query})
    return data.get("events", [])[:limit_per_type]


def get_event_by_slug(slug: str) -> dict:
    return _get(f"{GAMMA_API}/events/slug/{slug}")


def list_events(
    *,
    limit: int = 100,
    offset: int = 0,
    tag_slug: str | None = None,
    active: bool | None = None,
    closed: bool | None = None,
) -> list[dict]:
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if tag_slug:
        params["tag_slug"] = tag_slug
    if active is not None:
        params["active"] = str(active).lower()
    if closed is not None:
        params["closed"] = str(closed).lower()
    return _get(f"{GAMMA_API}/events", params)


def get_market_for_team(event: dict, team: str) -> dict:
    for market in event["markets"]:
        if market.get("groupItemTitle") == team:
            return market
    available = [m.get("groupItemTitle") for m in event["markets"]]
    raise ValueError(f"Time '{team}' não encontrado. Disponíveis: {available[:5]}...")


def fetch_price_history(
    token_id: str,
    *,
    interval: str = "max",
    fidelity: int = 60,
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> list[dict]:
    params: dict[str, Any] = {
        "market": token_id,
        "interval": interval,
        "fidelity": fidelity,
    }
    if start_ts is not None:
        params["startTs"] = start_ts
    if end_ts is not None:
        params["endTs"] = end_ts
    data = _get(f"{CLOB_API}/prices-history", params)
    return data.get("history", [])


def fetch_batch_price_history(
    token_ids: list[str],
    *,
    interval: str = "max",
    fidelity: int = 60,
) -> dict[str, list[dict]]:
    if not token_ids:
        return {}
    if len(token_ids) > 20:
        raise ValueError("batch-prices-history aceita no máximo 20 tokens por chamada")

    data = _post(
        f"{CLOB_API}/batch-prices-history",
        {"markets": token_ids, "interval": interval, "fidelity": fidelity},
    )
    return data.get("history", {})


def fetch_trades_page(
    *,
    condition_ids: list[str] | None = None,
    user: str | None = None,
    limit: int = 100,
    offset: int = 0,
    min_cash_usd: float | None = None,
    side: str | None = None,
) -> list[dict]:
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if condition_ids:
        params["market"] = ",".join(condition_ids)
    if user:
        params["user"] = user
    if side:
        params["side"] = side
    if min_cash_usd is not None:
        params["filterType"] = "CASH"
        params["filterAmount"] = min_cash_usd
    return _get(f"{DATA_API}/trades", params)


def iter_trades(
    *,
    condition_ids: list[str] | None = None,
    user: str | None = None,
    page_size: int = 1000,
    max_pages: int | None = None,
    min_cash_usd: float | None = None,
    sleep_seconds: float = DEFAULT_SLEEP,
) -> Iterator[dict]:
    offset = 0
    pages = 0

    while True:
        batch = fetch_trades_page(
            condition_ids=condition_ids,
            user=user,
            limit=page_size,
            offset=offset,
            min_cash_usd=min_cash_usd,
        )
        if not batch:
            break

        yield from batch
        pages += 1

        if len(batch) < page_size:
            break
        if max_pages is not None and pages >= max_pages:
            break
        if offset + page_size > 10000:
            break

        offset += page_size
        time.sleep(sleep_seconds)


def fetch_holders(condition_id: str, limit: int = 20, min_balance: int = 1) -> list[dict]:
    return _get(
        f"{DATA_API}/holders",
        {"market": condition_id, "limit": limit, "minBalance": min_balance},
    )


def fetch_open_interest(condition_id: str) -> list[dict]:
    return _get(f"{DATA_API}/oi", {"market": condition_id})


def fetch_orderbook(token_id: str) -> dict:
    return _get(f"{CLOB_API}/book", {"token_id": token_id})
