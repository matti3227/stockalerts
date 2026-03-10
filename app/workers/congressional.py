"""Congressional stock trade scraper.

Sources:
  House: https://housestockwatcher.com/api  (JSON, no key required)
  Senate: https://senatestockwatcher.com/api (JSON, no key required)

Both endpoints return arrays of recent disclosures from their respective
chambers. Data is derived from public eFD filings.
"""
import hashlib
import logging
from datetime import date, datetime
from typing import Optional

import httpx
from sqlalchemy import select

from app.database import SessionLocal
from app.models import CongressionalTrade

logger = logging.getLogger(__name__)

HOUSE_URL  = "https://housestockwatcher.com/api"
SENATE_URL = "https://senatestockwatcher.com/api"


def _parse_date(val: Optional[str]) -> Optional[date]:
    if not val:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            return datetime.strptime(val[:19], fmt[:len(val[:19])]).date()
        except Exception:
            pass
    # Try trimming to YYYY-MM-DD
    try:
        return date.fromisoformat(val[:10])
    except Exception:
        return None


def _normalize_trade_type(raw: str) -> str:
    if not raw:
        return "unknown"
    r = raw.lower()
    if "purchase" in r or "buy" in r:
        return "buy"
    if "sale" in r or "sell" in r:
        return "sell"
    return raw.lower()


def _make_external_id(chamber: str, politician: str, ticker: str, trade_date: Optional[date]) -> str:
    key = f"{chamber}|{politician}|{ticker}|{trade_date}"
    return hashlib.md5(key.encode()).hexdigest()


def _scrape_house(client: httpx.Client) -> int:
    saved = 0
    try:
        resp = client.get(HOUSE_URL, timeout=20.0)
        resp.raise_for_status()
        records = resp.json()
        if not isinstance(records, list):
            records = records.get("data", []) if isinstance(records, dict) else []

        with SessionLocal() as db:
            for rec in records:
                try:
                    ticker = (rec.get("ticker") or "").strip().upper()
                    if not ticker or ticker in ("--", "N/A", ""):
                        continue

                    politician     = rec.get("representative") or rec.get("name") or "Unknown"
                    trade_date     = _parse_date(rec.get("transaction_date") or rec.get("date"))
                    disclosure_date = _parse_date(rec.get("disclosure_date"))
                    buy_sell       = _normalize_trade_type(rec.get("type") or rec.get("transaction_type") or "")
                    amount         = rec.get("amount") or ""
                    party          = rec.get("party") or ""

                    ext_id = _make_external_id("house", politician, ticker, trade_date)

                    if db.execute(
                        select(CongressionalTrade.id).where(CongressionalTrade.external_id == ext_id)
                    ).scalar_one_or_none():
                        continue

                    db.add(CongressionalTrade(
                        external_id=ext_id,
                        politician=politician,
                        chamber="house",
                        party=party,
                        ticker=ticker,
                        buy_sell=buy_sell,
                        amount=amount,
                        trade_date=trade_date,
                        disclosure_date=disclosure_date,
                    ))
                    saved += 1

                except Exception:
                    logger.exception("House trade persist error: %s", rec)

            db.commit()

    except Exception:
        logger.exception("Failed to fetch House stock data")

    return saved


def _scrape_senate(client: httpx.Client) -> int:
    saved = 0
    try:
        resp = client.get(SENATE_URL, timeout=20.0)
        resp.raise_for_status()
        records = resp.json()
        if not isinstance(records, list):
            records = records.get("data", []) if isinstance(records, dict) else []

        with SessionLocal() as db:
            for rec in records:
                try:
                    ticker = (rec.get("ticker") or "").strip().upper()
                    if not ticker or ticker in ("--", "N/A", ""):
                        continue

                    politician      = rec.get("senator") or rec.get("name") or "Unknown"
                    trade_date      = _parse_date(rec.get("transaction_date") or rec.get("date"))
                    disclosure_date = _parse_date(rec.get("disclosure_date"))
                    buy_sell        = _normalize_trade_type(rec.get("type") or rec.get("transaction_type") or "")
                    amount          = rec.get("amount") or ""
                    party           = rec.get("party") or ""

                    ext_id = _make_external_id("senate", politician, ticker, trade_date)

                    if db.execute(
                        select(CongressionalTrade.id).where(CongressionalTrade.external_id == ext_id)
                    ).scalar_one_or_none():
                        continue

                    db.add(CongressionalTrade(
                        external_id=ext_id,
                        politician=politician,
                        chamber="senate",
                        party=party,
                        ticker=ticker,
                        buy_sell=buy_sell,
                        amount=amount,
                        trade_date=trade_date,
                        disclosure_date=disclosure_date,
                    ))
                    saved += 1

                except Exception:
                    logger.exception("Senate trade persist error: %s", rec)

            db.commit()

    except Exception:
        logger.exception("Failed to fetch Senate stock data")

    return saved


def scrape() -> int:
    """Scrape House and Senate disclosures. Returns count of new records saved."""
    with httpx.Client() as client:
        house  = _scrape_house(client)
        senate = _scrape_senate(client)

    total = house + senate
    logger.info("Congressional scrape done — house=%d senate=%d total=%d", house, senate, total)
    return total
