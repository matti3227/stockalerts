import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional

import httpx
import yfinance as yf
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from apscheduler.schedulers.background import BackgroundScheduler

from app.database import get_db, init_db
from app.models.schemas import Alert, Mention, Post, TickerMetric
from app.workers.reddit_scraper import RedditScraper
from app.workers.stocktwits_scraper import StockTwitsScraper

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Price cache  (ticker -> (price, pct_change, fetched_at))
# ---------------------------------------------------------------------------

_price_cache: dict[str, tuple[float, float, datetime]] = {}
_PRICE_TTL = timedelta(minutes=5)


def _get_price(ticker: str) -> tuple[Optional[float], Optional[float]]:
    """Return (price, pct_change) for *ticker*, using a 5-minute in-memory cache."""
    now = datetime.utcnow()
    cached = _price_cache.get(ticker)
    if cached and (now - cached[2]) < _PRICE_TTL:
        return cached[0], cached[1]

    try:
        info = yf.Ticker(ticker).fast_info
        price = float(info.last_price) if info.last_price else None
        prev_close = float(info.previous_close) if info.previous_close else None
        pct = round((price - prev_close) / prev_close * 100, 2) if price and prev_close else None
        _price_cache[ticker] = (price, pct, now)
        return price, pct
    except Exception:
        logger.warning("Could not fetch price for %s", ticker)
        return None, None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("Database tables ready")

    scheduler = BackgroundScheduler()
    scheduler.add_job(_run_stocktwits, "interval", minutes=5, id="stocktwits")
    scheduler.start()
    logger.info("Scheduler started — StockTwits scrape every 5 minutes")

    # Run once immediately so data is available right after deploy
    _run_stocktwits()

    yield

    scheduler.shutdown(wait=False)
    logger.info("Scheduler stopped")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Stock Alerts API",
    version="1.0.0",
    description="Real-time stock sentiment analysis from Reddit and StockTwits.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic schemas (request / response)
# ---------------------------------------------------------------------------

class TrendingItem(BaseModel):
    ticker: str
    mention_count: int
    avg_sentiment: float
    sentiment_label: str
    price: Optional[float]
    percent_change: Optional[float]


class TickerDetail(BaseModel):
    ticker: str
    mention_count: int
    avg_sentiment: float
    bullish_count: int
    bearish_count: int
    neutral_count: int
    calculated_at: datetime


class MentionItem(BaseModel):
    id: int
    ticker: str
    content: str
    source: str
    author: str
    sentiment_score: float
    sentiment_label: str
    url: Optional[str]
    created_at: datetime


class AlertCreate(BaseModel):
    ticker: str
    # mention_spike | sentiment_bullish | sentiment_bearish
    condition: str
    threshold: float
    webhook_url: Optional[str] = None


class AlertResponse(BaseModel):
    id: int
    ticker: str
    condition: str
    threshold: float
    is_active: bool
    webhook_url: Optional[str]
    last_triggered: Optional[datetime]
    created_at: datetime


class ScrapeResult(BaseModel):
    status: str
    message: str


class AlertTestResult(BaseModel):
    ticker: str
    condition: str
    threshold: float
    current_value: float
    triggered: bool
    reason: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sentiment_label(score: float) -> str:
    if score >= 0.05:
        return "bullish"
    if score <= -0.05:
        return "bearish"
    return "neutral"


def _trigger_webhook(url: str, payload: dict) -> None:
    try:
        with httpx.Client(timeout=10.0) as client:
            client.post(url, json=payload)
    except Exception:
        logger.exception("Webhook delivery failed: %s", url)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow()}


@app.get("/api/trending", response_model=list[TrendingItem])
def get_trending(
    limit: int = 10,
    hours: int = 24,
    db: Session = Depends(get_db),
):
    """Top tickers by mention count over the last *hours* hours."""
    since = datetime.utcnow() - timedelta(hours=hours)

    result = db.execute(
        select(
            Mention.ticker,
            func.count(Mention.id).label("mention_count"),
            func.avg(Post.sentiment_score).label("avg_sentiment"),
        )
        .join(Post, Mention.post_id == Post.id)
        .where(Post.created_at >= since)
        .group_by(Mention.ticker)
        .order_by(desc("mention_count"))
        .limit(limit)
    )
    rows = result.all()

    items = []
    for row in rows:
        price, pct = _get_price(row.ticker)
        items.append(TrendingItem(
            ticker=row.ticker,
            mention_count=row.mention_count,
            avg_sentiment=round(row.avg_sentiment or 0.0, 4),
            sentiment_label=_sentiment_label(row.avg_sentiment or 0.0),
            price=price,
            percent_change=pct,
        ))
    return items


@app.get("/api/ticker/{symbol}", response_model=TickerDetail)
def get_ticker(
    symbol: str,
    hours: int = 24,
    db: Session = Depends(get_db),
):
    """Aggregated sentiment metrics for a single ticker."""
    symbol = symbol.upper()
    since = datetime.utcnow() - timedelta(hours=hours)

    result = db.execute(
        select(Post.sentiment_score, Post.sentiment_label)
        .join(Mention, Mention.post_id == Post.id)
        .where(Mention.ticker == symbol)
        .where(Post.created_at >= since)
    )
    rows = result.all()

    if not rows:
        raise HTTPException(
            status_code=404, detail=f"No data found for {symbol} in the last {hours}h"
        )

    mention_count = len(rows)
    avg_sentiment = sum(r.sentiment_score or 0.0 for r in rows) / mention_count
    bullish_count = sum(1 for r in rows if r.sentiment_label == "bullish")
    bearish_count = sum(1 for r in rows if r.sentiment_label == "bearish")
    neutral_count = mention_count - bullish_count - bearish_count

    return TickerDetail(
        ticker=symbol,
        mention_count=mention_count,
        avg_sentiment=round(avg_sentiment, 4),
        bullish_count=bullish_count,
        bearish_count=bearish_count,
        neutral_count=neutral_count,
        calculated_at=datetime.utcnow(),
    )


@app.get("/api/mentions/{symbol}", response_model=list[MentionItem])
def get_mentions(
    symbol: str,
    limit: int = 50,
    hours: int = 24,
    db: Session = Depends(get_db),
):
    """Recent posts mentioning *symbol*."""
    symbol = symbol.upper()
    since = datetime.utcnow() - timedelta(hours=hours)

    result = db.execute(
        select(
            Mention.id,
            Mention.ticker,
            Post.content,
            Post.source,
            Post.author,
            Post.sentiment_score,
            Post.sentiment_label,
            Post.url,
            Post.created_at,
        )
        .join(Post, Mention.post_id == Post.id)
        .where(Mention.ticker == symbol)
        .where(Post.created_at >= since)
        .order_by(desc(Post.created_at))
        .limit(limit)
    )
    rows = result.all()

    return [
        MentionItem(
            id=r.id,
            ticker=r.ticker,
            content=r.content,
            source=r.source,
            author=r.author,
            sentiment_score=round(r.sentiment_score or 0.0, 4),
            sentiment_label=r.sentiment_label or "neutral",
            url=r.url,
            created_at=r.created_at,
        )
        for r in rows
    ]


@app.get("/api/alerts", response_model=list[AlertResponse])
def get_alerts(db: Session = Depends(get_db)):
    """List all alert rules."""
    result = db.execute(select(Alert).order_by(desc(Alert.created_at)))
    alerts = result.scalars().all()
    return [
        AlertResponse(
            id=a.id,
            ticker=a.ticker,
            condition=a.condition,
            threshold=a.threshold,
            is_active=a.is_active,
            webhook_url=a.webhook_url,
            last_triggered=a.last_triggered,
            created_at=a.created_at,
        )
        for a in alerts
    ]


@app.post("/api/alerts", response_model=AlertResponse, status_code=201)
def create_alert(body: AlertCreate, db: Session = Depends(get_db)):
    """Create a new alert rule."""
    valid_conditions = {"mention_spike", "sentiment_bullish", "sentiment_bearish"}
    if body.condition not in valid_conditions:
        raise HTTPException(
            status_code=422,
            detail=f"condition must be one of: {', '.join(sorted(valid_conditions))}",
        )

    alert = Alert(
        ticker=body.ticker.upper(),
        condition=body.condition,
        threshold=body.threshold,
        webhook_url=body.webhook_url,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)

    return AlertResponse(
        id=alert.id,
        ticker=alert.ticker,
        condition=alert.condition,
        threshold=alert.threshold,
        is_active=alert.is_active,
        webhook_url=alert.webhook_url,
        last_triggered=alert.last_triggered,
        created_at=alert.created_at,
    )


@app.post("/api/alerts/test", response_model=AlertTestResult)
def test_alert(
    body: AlertCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Evaluate an alert rule against current data without persisting it.
    Fires the webhook if triggered.
    """
    ticker = body.ticker.upper()
    since = datetime.utcnow() - timedelta(hours=24)

    # ------------------------------------------------------------------
    # Evaluate condition
    # ------------------------------------------------------------------
    triggered = False
    current_value = 0.0
    reason = ""

    if body.condition == "mention_spike":
        result = db.execute(
            select(func.count(Mention.id))
            .join(Post, Mention.post_id == Post.id)
            .where(Mention.ticker == ticker)
            .where(Post.created_at >= since)
        )
        current_value = float(result.scalar() or 0)
        triggered = current_value >= body.threshold
        reason = (
            f"{int(current_value)} mentions in the last 24 h "
            f"(threshold: {int(body.threshold)})"
        )

    elif body.condition in ("sentiment_bullish", "sentiment_bearish"):
        result = db.execute(
            select(func.avg(Post.sentiment_score))
            .join(Mention, Mention.post_id == Post.id)
            .where(Mention.ticker == ticker)
            .where(Post.created_at >= since)
        )
        avg = result.scalar()
        current_value = round(float(avg or 0.0), 4)

        if body.condition == "sentiment_bullish":
            triggered = current_value >= body.threshold
            reason = (
                f"avg sentiment {current_value:.4f} "
                f"(bullish threshold: {body.threshold})"
            )
        else:
            triggered = current_value <= body.threshold
            reason = (
                f"avg sentiment {current_value:.4f} "
                f"(bearish threshold: {body.threshold})"
            )
    else:
        raise HTTPException(status_code=422, detail="Unknown condition")

    # ------------------------------------------------------------------
    # Fire webhook in the background if triggered
    # ------------------------------------------------------------------
    if triggered and body.webhook_url:
        payload = {
            "ticker": ticker,
            "condition": body.condition,
            "threshold": body.threshold,
            "current_value": current_value,
            "reason": reason,
            "triggered_at": datetime.utcnow().isoformat(),
        }
        background_tasks.add_task(_trigger_webhook, body.webhook_url, payload)

    return AlertTestResult(
        ticker=ticker,
        condition=body.condition,
        threshold=body.threshold,
        current_value=current_value,
        triggered=triggered,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Scrape triggers
# ---------------------------------------------------------------------------

def _run_reddit():
    RedditScraper().scrape()


def _run_stocktwits():
    logger.info("Starting StockTwits scraper")
    try:
        count = StockTwitsScraper().scrape()
        logger.info("Scraper completed: %d posts saved", count)
    except Exception:
        logger.exception("StockTwits scraper raised an unhandled exception")


@app.post("/api/scrape/reddit", response_model=ScrapeResult)
def scrape_reddit(background_tasks: BackgroundTasks):
    """Trigger a Reddit scrape in the background."""
    background_tasks.add_task(_run_reddit)
    return ScrapeResult(status="accepted", message="Reddit scrape started")


@app.post("/api/scrape/stocktwits", response_model=ScrapeResult)
def scrape_stocktwits(background_tasks: BackgroundTasks):
    """Trigger a StockTwits scrape in the background."""
    background_tasks.add_task(_run_stocktwits)
    return ScrapeResult(status="accepted", message="StockTwits scrape started")


@app.post("/api/scrape/test")
def scrape_test(db: Session = Depends(get_db)):
    """Run the StockTwits scraper synchronously and report what was stored."""
    posts_before = db.execute(select(func.count(Post.id))).scalar() or 0
    mentions_before = db.execute(select(func.count(Mention.id))).scalar() or 0

    error = None
    new_posts = 0
    try:
        new_posts = StockTwitsScraper().scrape()
    except Exception as exc:
        logger.exception("scrape/test failed")
        error = str(exc)

    posts_after = db.execute(select(func.count(Post.id))).scalar() or 0
    mentions_after = db.execute(select(func.count(Mention.id))).scalar() or 0

    return {
        "status": "error" if error else "ok",
        "new_posts_saved": new_posts,
        "posts_delta": posts_after - posts_before,
        "mentions_delta": mentions_after - mentions_before,
        "total_posts": posts_after,
        "total_mentions": mentions_after,
        "error": error,
    }


@app.get("/api/debug/db-status")
def db_status(db: Session = Depends(get_db)):
    """Return total counts for posts, mentions, and distinct tickers."""
    total_posts = db.execute(select(func.count(Post.id))).scalar() or 0
    total_mentions = db.execute(select(func.count(Mention.id))).scalar() or 0
    total_tickers = db.execute(select(func.count(func.distinct(Mention.ticker)))).scalar() or 0

    return {
        "total_posts": total_posts,
        "total_mentions": total_mentions,
        "total_tickers": total_tickers,
    }
