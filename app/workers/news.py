"""News RSS scraper — pulls headlines from major financial feeds."""
import logging
import re
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Optional

import feedparser
from sqlalchemy import select

from app.database import SessionLocal
from app.models import Mention, Post
from app.workers.reddit import extract_tickers  # reuse ticker extraction
from app.workers.sentiment import analyze

logger = logging.getLogger(__name__)

RSS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
    "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
    "https://feeds.reuters.com/reuters/businessNews",
    "https://www.investing.com/rss/news.rss",
    "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml",
]


def _parse_date(entry) -> datetime:
    """Best-effort parse of an RSS entry's published date."""
    for attr in ("published", "updated"):
        val = getattr(entry, attr, None)
        if val:
            try:
                return parsedate_to_datetime(val).replace(tzinfo=None)
            except Exception:
                pass
    return datetime.utcnow()


def scrape() -> int:
    """Fetch all configured RSS feeds and save mentions. Returns new post count."""
    total = 0

    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            logger.info("News RSS %s: %d entries", feed_url[:60], len(feed.entries))

            with SessionLocal() as db:
                for entry in feed.entries:
                    try:
                        url = entry.get("link") or ""
                        if not url:
                            continue

                        external_id = f"news_{url[:200]}"

                        if db.execute(
                            select(Post.id).where(Post.external_id == external_id)
                        ).scalar_one_or_none():
                            continue

                        headline = entry.get("title") or ""
                        summary  = entry.get("summary") or ""
                        full_text = f"{headline} {summary}"

                        tickers = extract_tickers(full_text)
                        if not tickers:
                            continue

                        score, label, confidence = analyze(headline or full_text)
                        created_at = _parse_date(entry)

                        post = Post(
                            external_id=external_id,
                            text=full_text[:5000],
                            source="news",
                            author=entry.get("author") or feed.feed.get("title") or "news",
                            engagement=0,
                            url=url,
                            created_at=created_at,
                        )
                        db.add(post)
                        db.flush()

                        for ticker in tickers:
                            db.add(Mention(
                                post_id=post.id,
                                ticker=ticker,
                                sentiment_score=score,
                                sentiment_label=label,
                                sentiment_confidence=confidence,
                                created_at=created_at,
                            ))

                        total += 1

                    except Exception:
                        logger.exception("News: failed to persist entry from %s", feed_url[:60])

                db.commit()

        except Exception:
            logger.exception("News: failed to fetch feed %s", feed_url[:60])

    logger.info("News scrape done — %d new posts", total)
    return total
