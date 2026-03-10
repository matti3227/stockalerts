"""Dark pool / off-exchange volume data from FINRA ATS Transparency.

FINRA publishes weekly aggregated ATS (Alternative Trading System) data at
https://otctransparency.finra.org — this is real off-exchange volume, free
and public, but available on a weekly delay (Friday's data appears Monday).

The API returns share quantities per ATS venue per symbol. We sum across all
venues to get the total dark/ATS volume for each ticker.
"""
import logging
from datetime import datetime
from typing import Optional

import httpx
from sqlalchemy import select

from app.database import SessionLocal
from app.models import DarkPool

logger = logging.getLogger(__name__)

# FINRA OTC Transparency — weekly ATS summary
FINRA_ATS_URL = "https://otctransparency.finra.org/otctransparency/api/weeklySummary/ATS"

# Maximum tickers to store (sorted by dark volume descending)
TOP_N = 200


def _fetch_finra_ats(client: httpx.Client) -> list[dict]:
    """Fetch the latest weekly ATS summary from FINRA."""
    try:
        resp = client.get(FINRA_ATS_URL, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        # API may return a wrapper object or a direct list
        if isinstance(data, list):
            return data
        return data.get("data") or data.get("records") or []
    except Exception:
        logger.exception("Failed to fetch FINRA ATS data")
        return []


def scrape() -> int:
    """Download latest FINRA ATS data and persist top dark-pool tickers."""
    with httpx.Client() as client:
        raw_records = _fetch_finra_ats(client)

    if not raw_records:
        logger.warning("FINRA ATS returned no records")
        return 0

    # Aggregate volume by symbol across all ATS venues
    aggregated: dict[str, dict] = {}
    week_start: Optional[datetime] = None

    for rec in raw_records:
        # Handle both camelCase and snake_case field names
        ticker = (
            rec.get("issueSymbolIdentifier")
            or rec.get("symbol")
            or rec.get("ticker")
            or ""
        ).strip().upper()

        if not ticker or len(ticker) > 10:
            continue

        try:
            dark_vol  = int(rec.get("totalWeeklyShareQuantity") or rec.get("ats_volume") or 0)
            short_vol = int(rec.get("shortSaleWeeklyShareQuantity") or rec.get("short_volume") or 0)
            total_vol = int(rec.get("totalWeeklyShareQuantity") or rec.get("total_volume") or dark_vol)
        except (TypeError, ValueError):
            continue

        if ticker not in aggregated:
            aggregated[ticker] = {"dark": 0, "short": 0, "total": 0}

        aggregated[ticker]["dark"]  += dark_vol
        aggregated[ticker]["short"] += short_vol
        aggregated[ticker]["total"] += total_vol

        # Extract week date from first record that has it
        if week_start is None:
            raw_date = (
                rec.get("weekStartDate")
                or rec.get("week_start")
                or rec.get("reportingPeriod")
            )
            if raw_date:
                try:
                    week_start = datetime.fromisoformat(str(raw_date)[:10])
                except Exception:
                    pass

    if week_start is None:
        week_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    # Sort by dark volume and take top N
    top = sorted(aggregated.items(), key=lambda kv: kv[1]["dark"], reverse=True)[:TOP_N]

    saved = 0
    with SessionLocal() as db:
        for ticker, vols in top:
            try:
                dark_vol  = vols["dark"]
                short_vol = vols["short"]
                total_vol = vols["total"] or dark_vol

                # Deduplicate by (ticker, timestamp)
                existing = db.execute(
                    select(DarkPool.id).where(
                        DarkPool.ticker == ticker,
                        DarkPool.timestamp == week_start,
                    )
                ).scalar_one_or_none()

                if existing:
                    continue

                dark_pct = round(dark_vol / total_vol, 4) if total_vol else None

                db.add(DarkPool(
                    ticker=ticker,
                    volume=dark_vol,
                    short_volume=short_vol if short_vol else None,
                    total_volume=total_vol,
                    dark_pct=dark_pct,
                    timestamp=week_start,
                    source="finra",
                ))
                saved += 1

            except Exception:
                logger.exception("DarkPool persist error: ticker=%s", ticker)

        db.commit()

    logger.info("Dark pool scrape done — %d new records (week_start=%s)", saved, week_start.date())
    return saved
