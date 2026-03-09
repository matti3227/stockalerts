import asyncio
import logging
import re
from datetime import datetime

import praw
from sqlalchemy import select
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app.config import settings
from app.database import AsyncSessionLocal
from app.models.schemas import Mention, Post

logger = logging.getLogger(__name__)

SUBREDDITS = ["wallstreetbets", "stocks", "investing", "options", "StockMarket"]

# $TICKER is the most reliable signal; bare UPPERCASE words need blacklisting
_DOLLAR_TICKER = re.compile(r"\$([A-Z]{1,5})\b")
_WORD_TICKER = re.compile(r"\b([A-Z]{2,5})\b")

WORD_BLACKLIST = {
    # Articles / prepositions / conjunctions
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
    # Finance jargon that isn't a ticker
    "ETF", "CEO", "CFO", "CTO", "IPO", "ATH", "ATL", "EPS", "OTM",
    "ITM", "ATM", "PE", "EV", "DD", "CALL", "PUTS", "BULL", "BEAR",
    "MOON", "PUMP", "DUMP", "SELL", "HOLD", "CASH", "LOSS", "GAIN",
    "COST", "RATE", "DEBT", "RISK", "FUND", "BANK", "TECH", "COMP",
    "CORP", "YEAR", "WEEK", "NEWS", "REAL",
    # Social / internet slang
    "WSB", "IMO", "LOL", "WTF", "FUD", "FOMO", "YOLO", "TLDR", "IIRC",
    # Currencies / regulators / geo
    "USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "SEC", "FDA",
    "FED", "GDP", "CPI", "PPI", "NFP", "FOMC", "QE", "QT",
    "AI", "ML", "IT", "US", "UK", "EU", "UN", "NY", "LA", "SF", "DC",
    "USA", "CNN", "BBC", "WSJ", "NYT", "IMF", "WTO", "NATO",
}

_vader = SentimentIntensityAnalyzer()


def extract_tickers(text: str) -> list[str]:
    """Return unique ticker symbols found in *text*."""
    tickers: set[str] = set()
    for m in _DOLLAR_TICKER.finditer(text):
        tickers.add(m.group(1))
    for m in _WORD_TICKER.finditer(text):
        word = m.group(1)
        if word not in WORD_BLACKLIST:
            tickers.add(word)
    return list(tickers)


def analyze_sentiment(text: str) -> tuple[float, str]:
    """Return (compound_score, label) using VADER."""
    compound = _vader.polarity_scores(text)["compound"]
    if compound >= 0.05:
        return compound, "bullish"
    if compound <= -0.05:
        return compound, "bearish"
    return compound, "neutral"


class RedditScraper:
    def __init__(self) -> None:
        self._reddit = praw.Reddit(
            client_id=settings.reddit_client_id,
            client_secret=settings.reddit_client_secret,
            user_agent=settings.reddit_user_agent,
        )

    # PRAW is synchronous — run it in a thread-pool executor
    def _fetch_subreddit(self, subreddit_name: str, limit: int = 100) -> list[dict]:
        results = []
        subreddit = self._reddit.subreddit(subreddit_name)
        for submission in subreddit.hot(limit=limit):
            text = f"{submission.title} {submission.selftext}"
            tickers = extract_tickers(text)
            if not tickers:
                continue
            score, label = analyze_sentiment(text)
            results.append(
                {
                    "external_id": f"reddit_{submission.id}",
                    "source": "reddit",
                    "content": text[:5000],
                    "author": str(submission.author) if submission.author else "[deleted]",
                    "score": submission.score,
                    "url": f"https://reddit.com{submission.permalink}",
                    "sentiment_score": score,
                    "sentiment_label": label,
                    "created_at": datetime.utcfromtimestamp(submission.created_utc),
                    "_tickers": tickers,
                }
            )
        return results

    async def scrape(self) -> int:
        """Scrape all configured subreddits and persist new posts. Returns count saved."""
        total = 0
        loop = asyncio.get_event_loop()

        for subreddit_name in SUBREDDITS:
            try:
                logger.info("Scraping r/%s", subreddit_name)
                posts_data = await loop.run_in_executor(
                    None, self._fetch_subreddit, subreddit_name
                )

                async with AsyncSessionLocal() as db:
                    for data in posts_data:
                        tickers = data.pop("_tickers")

                        result = await db.execute(
                            select(Post).where(Post.external_id == data["external_id"])
                        )
                        if result.scalar_one_or_none():
                            continue

                        post = Post(**data)
                        db.add(post)
                        await db.flush()  # get post.id

                        for ticker in tickers:
                            db.add(
                                Mention(
                                    ticker=ticker,
                                    post_id=post.id,
                                    context=data["content"][:500],
                                )
                            )
                        total += 1

                    await db.commit()

            except Exception:
                logger.exception("Failed to scrape r/%s", subreddit_name)

        logger.info("Reddit scrape complete — %d new posts saved", total)
        return total
