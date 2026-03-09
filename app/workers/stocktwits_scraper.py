import logging
from datetime import datetime
from typing import Optional

import httpx
from sqlalchemy import select
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app.config import settings
from app.database import SessionLocal
from app.models.schemas import Mention, Post

logger = logging.getLogger(__name__)

STOCKTWITS_BASE = "https://api.stocktwits.com/api/2"

# Fallback list if the trending endpoint fails
DEFAULT_TICKERS = [
    "SPY", "QQQ", "AAPL", "TSLA", "NVDA",
    "AMD", "MSFT", "AMZN", "META", "GOOGL",
]

_vader = SentimentIntensityAnalyzer()


def analyze_sentiment(
    text: str, stocktwits_label: Optional[str] = None
) -> tuple[float, str]:
    """
    Prefer the author-supplied StockTwits sentiment when present;
    otherwise fall back to VADER.
    """
    if stocktwits_label:
        normalised = stocktwits_label.lower()
        if normalised == "bullish":
            return 0.5, "bullish"
        if normalised == "bearish":
            return -0.5, "bearish"

    compound = _vader.polarity_scores(text)["compound"]
    if compound >= 0.05:
        return compound, "bullish"
    if compound <= -0.05:
        return compound, "bearish"
    return compound, "neutral"


class StockTwitsScraper:
    def __init__(self) -> None:
        self._headers: dict[str, str] = {}
        if settings.stocktwits_access_token:
            self._headers["Authorization"] = f"OAuth {settings.stocktwits_access_token}"

    def _get_trending_tickers(self, client: httpx.Client) -> list[str]:
        try:
            resp = client.get(
                f"{STOCKTWITS_BASE}/trending/symbols.json",
                headers=self._headers,
                timeout=10.0,
            )
            resp.raise_for_status()
            symbols = resp.json().get("symbols", [])
            tickers = [s["symbol"] for s in symbols[:20]]
            logger.info("StockTwits trending: fetched %d tickers: %s", len(tickers), tickers)
            return tickers
        except Exception:
            logger.warning("Could not fetch trending symbols, using defaults: %s", DEFAULT_TICKERS)
            return DEFAULT_TICKERS

    def _get_symbol_stream(self, client: httpx.Client, ticker: str) -> list[dict]:
        try:
            resp = client.get(
                f"{STOCKTWITS_BASE}/streams/symbol/{ticker}.json",
                params={"limit": 30},
                headers=self._headers,
                timeout=10.0,
            )
            resp.raise_for_status()
            messages = resp.json().get("messages", [])
            logger.info("StockTwits %s: received %d messages", ticker, len(messages))
            return messages
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "StockTwits %s returned HTTP %s: %s",
                ticker, exc.response.status_code, exc.response.text[:200],
            )
            return []
        except Exception:
            logger.exception("Error fetching StockTwits stream for %s", ticker)
            return []

    def scrape(self) -> int:
        """Fetch trending tickers and their message streams. Returns count saved."""
        total = 0

        with httpx.Client() as client:
            tickers = self._get_trending_tickers(client)

            for ticker in tickers:
                messages = self._get_symbol_stream(client, ticker)
                if not messages:
                    continue

                saved = skipped = errors = 0

                for msg in messages:
                    try:
                        external_id = f"stocktwits_{msg['id']}"

                        with SessionLocal() as db:
                            existing = db.execute(
                                select(Post).where(Post.external_id == external_id)
                            ).scalar_one_or_none()

                            if existing:
                                skipped += 1
                                continue

                            body = msg.get("body", "")
                            st_label: Optional[str] = (
                                msg.get("entities", {})
                                .get("sentiment", {})
                                .get("basic")
                            )
                            score, label = analyze_sentiment(body, st_label)

                            try:
                                created_at = datetime.strptime(
                                    msg["created_at"], "%Y-%m-%dT%H:%M:%SZ"
                                )
                            except Exception:
                                created_at = datetime.utcnow()

                            post = Post(
                                external_id=external_id,
                                source="stocktwits",
                                content=body[:5000],
                                author=msg.get("user", {}).get("username", "unknown"),
                                score=msg.get("likes", {}).get("total", 0),
                                url=f"https://stocktwits.com/message/{msg['id']}",
                                sentiment_score=score,
                                sentiment_label=label,
                                created_at=created_at,
                            )
                            db.add(post)
                            db.flush()

                            db.add(
                                Mention(
                                    ticker=ticker,
                                    post_id=post.id,
                                    context=body[:500],
                                )
                            )
                            db.commit()
                            saved += 1
                            total += 1

                    except Exception:
                        logger.exception(
                            "Failed to persist StockTwits message id=%s for %s",
                            msg.get("id"), ticker,
                        )
                        errors += 1

                logger.info(
                    "StockTwits %s: saved=%d skipped=%d errors=%d",
                    ticker, saved, skipped, errors,
                )

        logger.info("StockTwits scrape complete — %d new posts saved", total)
        return total
