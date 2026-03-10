"""StockTwits scraper — fetches trending tickers and their message streams."""
import logging
from datetime import datetime
from typing import Optional

import httpx
from sqlalchemy import select

from app.config import settings
from app.database import SessionLocal
from app.models import Mention, Post
from app.workers.sentiment import analyze

logger = logging.getLogger(__name__)

STOCKTWITS_BASE = "https://api.stocktwits.com/api/2"

DEFAULT_TICKERS = [
    "SPY", "QQQ", "AAPL", "TSLA", "NVDA",
    "AMD", "MSFT", "AMZN", "META", "GOOGL",
]


def _author_weight(user: dict) -> float:
    if user.get("verified"):
        return 2.0
    if (user.get("followers") or 0) >= 1000:
        return 1.5
    return 1.0


def _get_trending_tickers(client: httpx.Client, headers: dict) -> list[str]:
    try:
        resp = client.get(
            f"{STOCKTWITS_BASE}/trending/symbols.json",
            headers=headers,
            timeout=10.0,
        )
        resp.raise_for_status()
        symbols = resp.json().get("symbols") or []
        tickers = [s["symbol"] for s in symbols[:20] if s.get("symbol")]
        logger.info("StockTwits trending: %d tickers", len(tickers))
        return tickers or DEFAULT_TICKERS
    except Exception:
        logger.warning("Could not fetch StockTwits trending; using defaults")
        return DEFAULT_TICKERS


def _get_symbol_stream(client: httpx.Client, headers: dict, ticker: str) -> list[dict]:
    try:
        resp = client.get(
            f"{STOCKTWITS_BASE}/streams/symbol/{ticker}.json",
            params={"limit": 30},
            headers=headers,
            timeout=10.0,
        )
        resp.raise_for_status()
        messages = resp.json().get("messages") or []
        logger.debug("StockTwits %s: %d messages", ticker, len(messages))
        return messages
    except httpx.HTTPStatusError as exc:
        logger.warning("StockTwits %s HTTP %s", ticker, exc.response.status_code)
        return []
    except Exception:
        logger.exception("StockTwits stream error for %s", ticker)
        return []


def scrape() -> int:
    """Scrape StockTwits trending streams. Returns count of new posts saved."""
    headers: dict[str, str] = {}
    if settings.stocktwits_access_token:
        headers["Authorization"] = f"OAuth {settings.stocktwits_access_token}"

    total = 0

    with httpx.Client() as client:
        tickers = _get_trending_tickers(client, headers)

        for ticker in tickers:
            messages = _get_symbol_stream(client, headers, ticker)
            saved = skipped = errors = 0

            for msg in messages:
                try:
                    external_id = f"stocktwits_{msg['id']}"

                    with SessionLocal() as db:
                        if db.execute(
                            select(Post.id).where(Post.external_id == external_id)
                        ).scalar_one_or_none():
                            skipped += 1
                            continue

                        body = msg.get("body") or ""

                        # Use `or {}` to safely handle JSON null fields
                        entities  = msg.get("entities") or {}
                        st_label: Optional[str] = (entities.get("sentiment") or {}).get("basic")
                        score, label, confidence = analyze(body, st_label)

                        try:
                            created_at = datetime.strptime(msg["created_at"], "%Y-%m-%dT%H:%M:%SZ")
                        except Exception:
                            created_at = datetime.utcnow()

                        user  = msg.get("user") or {}
                        likes = msg.get("likes") or {}
                        raw_engagement = likes.get("total") or 0
                        weight = _author_weight(user)
                        # Store weighted engagement so the trending query benefits
                        weighted_engagement = int(raw_engagement * weight)

                        post = Post(
                            external_id=external_id,
                            text=body[:5000],
                            source="stocktwits",
                            author=user.get("username", "unknown"),
                            engagement=weighted_engagement,
                            url=f"https://stocktwits.com/message/{msg['id']}",
                            created_at=created_at,
                        )
                        db.add(post)
                        db.flush()

                        db.add(Mention(
                            post_id=post.id,
                            ticker=ticker,
                            sentiment_score=score,
                            sentiment_label=label,
                            sentiment_confidence=confidence,
                            created_at=created_at,
                        ))
                        db.commit()
                        saved += 1
                        total += 1

                except Exception:
                    logger.exception("StockTwits persist error msg=%s ticker=%s", msg.get("id"), ticker)
                    errors += 1

            logger.info("StockTwits %s: saved=%d skipped=%d errors=%d", ticker, saved, skipped, errors)

    logger.info("StockTwits scrape done — %d new posts", total)
    return total
