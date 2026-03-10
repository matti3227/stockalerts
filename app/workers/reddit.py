"""Reddit scraper — pulls hot posts from finance subreddits via PRAW."""
import logging
import re
from datetime import datetime

import praw
from sqlalchemy import select

from app.config import settings
from app.database import SessionLocal
from app.models import Mention, Post
from app.workers.sentiment import analyze

logger = logging.getLogger(__name__)

SUBREDDITS = ["wallstreetbets", "stocks", "investing", "options", "StockMarket"]

_DOLLAR_TICKER = re.compile(r"\$([A-Z]{1,5})\b")
_WORD_TICKER   = re.compile(r"\b([A-Z]{2,5})\b")

WORD_BLACKLIST = {
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER",
    "WAS", "ONE", "OUR", "OUT", "DAY", "GET", "HAS", "HIM", "HIS", "HOW",
    "MAN", "NEW", "NOW", "OLD", "SEE", "TWO", "WAY", "WHO", "BOY", "DID",
    "ITS", "LET", "PUT", "SAY", "SHE", "TOO", "USE", "THEY", "WITH",
    "HAVE", "THIS", "WILL", "YOUR", "THAT", "WHAT", "WHEN", "MAKE",
    "INTO", "THAN", "MOST", "SOME", "WANT", "VERY", "ALSO", "COME",
    "BACK", "OVER", "KNOW", "LOOK", "NEED", "OPEN", "FEEL", "GIVE",
    "MEAN", "SAME", "MOVE", "BOTH", "MORE", "MUCH", "DOWN", "ONLY",
    "JUST", "LIKE", "FROM", "EVEN", "TAKE", "THEN", "LAST", "NEXT",
    "BEEN", "GOOD", "WELL", "BEST", "HIGH", "LONG",
    "ETF", "CEO", "CFO", "CTO", "IPO", "ATH", "ATL", "EPS", "OTM",
    "ITM", "ATM", "PE", "EV", "DD", "CALL", "PUTS", "BULL", "BEAR",
    "MOON", "PUMP", "DUMP", "SELL", "HOLD", "CASH", "LOSS", "GAIN",
    "COST", "RATE", "DEBT", "RISK", "FUND", "BANK", "TECH", "COMP",
    "CORP", "YEAR", "WEEK", "NEWS", "REAL",
    "WSB", "IMO", "LOL", "WTF", "FUD", "FOMO", "YOLO", "TLDR", "IIRC",
    "USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "SEC", "FDA",
    "FED", "GDP", "CPI", "PPI", "NFP", "FOMC", "QE", "QT",
    "AI", "ML", "IT", "US", "UK", "EU", "UN", "NY", "LA", "SF", "DC",
    "USA", "CNN", "BBC", "WSJ", "NYT", "IMF", "WTO", "NATO",
}


def extract_tickers(text: str) -> list[str]:
    tickers: set[str] = set()
    for m in _DOLLAR_TICKER.finditer(text):
        tickers.add(m.group(1))
    for m in _WORD_TICKER.finditer(text):
        word = m.group(1)
        if word not in WORD_BLACKLIST:
            tickers.add(word)
    return list(tickers)


def scrape() -> int:
    """Scrape Reddit finance subreddits. Returns count of new posts saved."""
    if not settings.reddit_client_id:
        logger.warning("Reddit credentials not configured; skipping Reddit scrape")
        return 0

    reddit = praw.Reddit(
        client_id=settings.reddit_client_id,
        client_secret=settings.reddit_client_secret,
        user_agent=settings.reddit_user_agent,
    )

    total = 0

    for sub_name in SUBREDDITS:
        try:
            logger.info("Scraping r/%s", sub_name)
            subreddit = reddit.subreddit(sub_name)

            with SessionLocal() as db:
                for submission in subreddit.hot(limit=100):
                    try:
                        external_id = f"reddit_{submission.id}"

                        if db.execute(
                            select(Post.id).where(Post.external_id == external_id)
                        ).scalar_one_or_none():
                            continue

                        full_text = f"{submission.title} {submission.selftext}"
                        tickers = extract_tickers(full_text)
                        if not tickers:
                            continue

                        score, label, confidence = analyze(full_text)
                        sub_score   = submission.score or 0
                        engagement  = min(sub_score, 10_000)   # cap to avoid outlier distortion
                        created_at  = datetime.utcfromtimestamp(submission.created_utc)

                        post = Post(
                            external_id=external_id,
                            text=full_text[:5000],
                            source="reddit",
                            author=str(submission.author) if submission.author else "[deleted]",
                            engagement=engagement,
                            url=f"https://reddit.com{submission.permalink}",
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
                        logger.exception("Reddit: failed to persist submission %s", getattr(submission, "id", "?"))

                db.commit()

        except Exception:
            logger.exception("Reddit: failed to scrape r/%s", sub_name)

    logger.info("Reddit scrape done — %d new posts", total)
    return total
