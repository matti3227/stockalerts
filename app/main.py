"""Stock Alerts API — main FastAPI application."""
import logging
import math
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.database import get_db, init_db
from app.models import Alert, CongressionalTrade, DarkPool, Mention, Post

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

_FILTER_MAP = {
    "15m":  timedelta(minutes=15),
    "1h":   timedelta(hours=1),
    "4h":   timedelta(hours=4),
    "24h":  timedelta(hours=24),
    "7d":   timedelta(days=7),
}


# ---------------------------------------------------------------------------
# Background jobs
# ---------------------------------------------------------------------------

def _run(name: str, fn):
    try:
        count = fn()
        logger.info("%s scrape complete — %d new records", name, count)
    except Exception:
        logger.exception("%s scrape raised an unhandled exception", name)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("Database ready")

    from app.workers import congressional, dark_pools, news, reddit, stocktwits

    scheduler = BackgroundScheduler()
    scheduler.add_job(lambda: _run("StockTwits",   stocktwits.scrape),   "interval", minutes=5,    id="stocktwits")
    scheduler.add_job(lambda: _run("Reddit",       reddit.scrape),       "interval", minutes=20,   id="reddit")
    scheduler.add_job(lambda: _run("News",         news.scrape),         "interval", minutes=10,   id="news")
    scheduler.add_job(lambda: _run("Congressional",congressional.scrape),"interval", hours=6,      id="congressional")
    scheduler.add_job(lambda: _run("DarkPool",     dark_pools.scrape),   "interval", hours=24,     id="dark_pools")
    scheduler.start()
    logger.info("Scheduler started")

    # Seed data immediately on startup
    _run("StockTwits",    stocktwits.scrape)
    _run("Congressional", congressional.scrape)
    _run("DarkPool",      dark_pools.scrape)

    yield

    scheduler.shutdown(wait=False)
    logger.info("Scheduler stopped")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Stock Alerts API",
    version="2.0.0",
    description="Real-time stock sentiment from Reddit, StockTwits, news, congressional trades, and dark pools.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TrendingItem(BaseModel):
    ticker: str
    mention_count: int
    velocity: float                  # mentions per hour in the window
    avg_sentiment: float
    sentiment_label: str
    sentiment_confidence: float
    trending_score: float
    congressional_activity: bool     # any congressional trade in last 30 days


class CongressionalItem(BaseModel):
    politician: str
    chamber: str
    party: str
    ticker: str
    buy_sell: str
    amount: str
    trade_date: Optional[date]
    disclosure_date: Optional[date]


class TickerDetail(BaseModel):
    ticker: str
    mention_count: int
    avg_sentiment: float
    sentiment_label: str
    bullish_count: int
    bearish_count: int
    neutral_count: int
    avg_confidence: float
    sources: list[str]
    congressional_trades: list[CongressionalItem]
    dark_pool_volume: Optional[int]


class MentionItem(BaseModel):
    id: int
    ticker: str
    text: str
    source: str
    author: str
    sentiment_score: float
    sentiment_label: str
    sentiment_confidence: float
    url: Optional[str]
    created_at: datetime


class DarkPoolItem(BaseModel):
    ticker: str
    volume: int
    short_volume: Optional[int]
    total_volume: Optional[int]
    dark_pct: Optional[float]
    timestamp: datetime


class AlertCreate(BaseModel):
    ticker: str
    alert_type: str        # mention_spike | bullish | bearish
    threshold: float
    webhook_url: Optional[str] = None


class AlertResponse(BaseModel):
    id: int
    ticker: str
    alert_type: str
    threshold: float
    status: str
    webhook_url: Optional[str]
    created_at: datetime
    triggered_at: Optional[datetime]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sentiment_label(score: float) -> str:
    if score >= 0.05:
        return "bullish"
    if score <= -0.05:
        return "bearish"
    return "neutral"


def _congressional_tickers_last_30d(db: Session) -> set[str]:
    """Return set of tickers with a congressional trade in the last 30 days."""
    cutoff = datetime.utcnow() - timedelta(days=30)
    rows = db.execute(
        select(CongressionalTrade.ticker)
        .where(CongressionalTrade.trade_date >= cutoff.date())
        .distinct()
    ).scalars().all()
    return set(rows)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health(db: Session = Depends(get_db)):
    posts    = db.execute(select(func.count(Post.id))).scalar() or 0
    mentions = db.execute(select(func.count(Mention.id))).scalar() or 0
    return {
        "status": "ok",
        "timestamp": datetime.utcnow(),
        "db_posts": posts,
        "db_mentions": mentions,
    }


@app.get("/api/trending", response_model=list[TrendingItem])
def get_trending(
    filter: str = "24h",
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """Top tickers ranked by a quality-weighted trending score.

    Score = SUM((1 + log(1+engagement)) * (1 + confidence*0.5)) * (1 + log(1+velocity))
    where velocity = mention_count / window_hours
    """
    window       = _FILTER_MAP.get(filter, timedelta(hours=24))
    since        = datetime.utcnow() - window
    window_hours = max(window.total_seconds() / 3600, 0.25)

    # Per-ticker aggregation
    raw_score_expr = func.sum(
        (1.0 + func.log(func.greatest(Post.engagement, 0) + 1.0))
        * (1.0 + func.coalesce(Mention.sentiment_confidence, 0.5) * 0.5)
    ).label("raw_score")

    rows = db.execute(
        select(
            Mention.ticker,
            func.count(Mention.id).label("mention_count"),
            func.avg(Mention.sentiment_score).label("avg_sentiment"),
            func.avg(Mention.sentiment_confidence).label("avg_confidence"),
            raw_score_expr,
        )
        .join(Post, Mention.post_id == Post.id)
        .where(Mention.created_at >= since)
        .group_by(Mention.ticker)
        .order_by(desc("raw_score"))
        .limit(limit)
    ).all()

    if not rows:
        return []

    cong_active = _congressional_tickers_last_30d(db)

    result = []
    for row in rows:
        mention_count  = row.mention_count
        velocity       = round(mention_count / window_hours, 4)
        raw_score      = row.raw_score or 0.0
        trending_score = round(raw_score * (1.0 + math.log1p(velocity)), 4)
        avg_s          = row.avg_sentiment or 0.0
        avg_conf       = row.avg_confidence or 0.5

        result.append(TrendingItem(
            ticker=row.ticker,
            mention_count=mention_count,
            velocity=velocity,
            avg_sentiment=round(avg_s, 4),
            sentiment_label=_sentiment_label(avg_s),
            sentiment_confidence=round(avg_conf, 4),
            trending_score=trending_score,
            congressional_activity=row.ticker in cong_active,
        ))

    return result


@app.get("/api/ticker/{symbol}", response_model=TickerDetail)
def get_ticker(
    symbol: str,
    hours: int = 24,
    db: Session = Depends(get_db),
):
    symbol = symbol.upper()
    since  = datetime.utcnow() - timedelta(hours=hours)

    rows = db.execute(
        select(
            Mention.sentiment_score,
            Mention.sentiment_label,
            Mention.sentiment_confidence,
            Post.source,
        )
        .join(Post, Mention.post_id == Post.id)
        .where(Mention.ticker == symbol)
        .where(Mention.created_at >= since)
    ).all()

    if not rows:
        raise HTTPException(status_code=404, detail=f"No data for {symbol} in last {hours}h")

    n             = len(rows)
    avg_s         = sum(r.sentiment_score or 0.0 for r in rows) / n
    avg_conf      = sum(r.sentiment_confidence or 0.5 for r in rows) / n
    bullish_count = sum(1 for r in rows if r.sentiment_label == "bullish")
    bearish_count = sum(1 for r in rows if r.sentiment_label == "bearish")
    sources       = list({r.source for r in rows})

    # Congressional trades for this ticker (last 90 days)
    cutoff = datetime.utcnow() - timedelta(days=90)
    cong_rows = db.execute(
        select(CongressionalTrade)
        .where(CongressionalTrade.ticker == symbol)
        .where(CongressionalTrade.trade_date >= cutoff.date())
        .order_by(desc(CongressionalTrade.trade_date))
        .limit(20)
    ).scalars().all()

    cong_items = [
        CongressionalItem(
            politician=c.politician,
            chamber=c.chamber or "",
            party=c.party or "",
            ticker=c.ticker,
            buy_sell=c.buy_sell,
            amount=c.amount or "",
            trade_date=c.trade_date,
            disclosure_date=c.disclosure_date,
        )
        for c in cong_rows
    ]

    # Latest dark pool reading for this ticker
    dp_row = db.execute(
        select(DarkPool.volume)
        .where(DarkPool.ticker == symbol)
        .order_by(desc(DarkPool.timestamp))
        .limit(1)
    ).scalar_one_or_none()

    return TickerDetail(
        ticker=symbol,
        mention_count=n,
        avg_sentiment=round(avg_s, 4),
        sentiment_label=_sentiment_label(avg_s),
        bullish_count=bullish_count,
        bearish_count=bearish_count,
        neutral_count=n - bullish_count - bearish_count,
        avg_confidence=round(avg_conf, 4),
        sources=sources,
        congressional_trades=cong_items,
        dark_pool_volume=dp_row,
    )


@app.get("/api/mentions/{symbol}", response_model=list[MentionItem])
def get_mentions(
    symbol: str,
    hours: int = 24,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    symbol = symbol.upper()
    since  = datetime.utcnow() - timedelta(hours=hours)

    rows = db.execute(
        select(
            Mention.id,
            Mention.ticker,
            Post.text,
            Post.source,
            Post.author,
            Mention.sentiment_score,
            Mention.sentiment_label,
            Mention.sentiment_confidence,
            Post.url,
            Mention.created_at,
        )
        .join(Post, Mention.post_id == Post.id)
        .where(Mention.ticker == symbol)
        .where(Mention.created_at >= since)
        .order_by(desc(Mention.created_at))
        .limit(limit)
    ).all()

    return [
        MentionItem(
            id=r.id,
            ticker=r.ticker,
            text=r.text,
            source=r.source,
            author=r.author or "",
            sentiment_score=round(r.sentiment_score or 0.0, 4),
            sentiment_label=r.sentiment_label or "neutral",
            sentiment_confidence=round(r.sentiment_confidence or 0.5, 4),
            url=r.url,
            created_at=r.created_at,
        )
        for r in rows
    ]


@app.get("/api/congressional", response_model=list[CongressionalItem])
def get_congressional(
    days: int = 30,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    cutoff = datetime.utcnow() - timedelta(days=days)
    rows = db.execute(
        select(CongressionalTrade)
        .where(CongressionalTrade.trade_date >= cutoff.date())
        .order_by(desc(CongressionalTrade.disclosure_date), desc(CongressionalTrade.trade_date))
        .limit(limit)
    ).scalars().all()

    return [
        CongressionalItem(
            politician=r.politician,
            chamber=r.chamber or "",
            party=r.party or "",
            ticker=r.ticker,
            buy_sell=r.buy_sell,
            amount=r.amount or "",
            trade_date=r.trade_date,
            disclosure_date=r.disclosure_date,
        )
        for r in rows
    ]


@app.get("/api/congressional/{ticker}", response_model=list[CongressionalItem])
def get_congressional_ticker(
    ticker: str,
    days: int = 180,
    db: Session = Depends(get_db),
):
    ticker = ticker.upper()
    cutoff = datetime.utcnow() - timedelta(days=days)
    rows = db.execute(
        select(CongressionalTrade)
        .where(CongressionalTrade.ticker == ticker)
        .where(CongressionalTrade.trade_date >= cutoff.date())
        .order_by(desc(CongressionalTrade.trade_date))
        .limit(100)
    ).scalars().all()

    return [
        CongressionalItem(
            politician=r.politician,
            chamber=r.chamber or "",
            party=r.party or "",
            ticker=r.ticker,
            buy_sell=r.buy_sell,
            amount=r.amount or "",
            trade_date=r.trade_date,
            disclosure_date=r.disclosure_date,
        )
        for r in rows
    ]


@app.get("/api/dark-pools", response_model=list[DarkPoolItem])
def get_dark_pools(
    limit: int = 50,
    db: Session = Depends(get_db),
):
    # Return tickers with the highest dark-pool volume from the latest data week
    latest_ts = db.execute(
        select(func.max(DarkPool.timestamp))
    ).scalar_one_or_none()

    if not latest_ts:
        return []

    rows = db.execute(
        select(DarkPool)
        .where(DarkPool.timestamp == latest_ts)
        .order_by(desc(DarkPool.volume))
        .limit(limit)
    ).scalars().all()

    return [
        DarkPoolItem(
            ticker=r.ticker,
            volume=r.volume,
            short_volume=r.short_volume,
            total_volume=r.total_volume,
            dark_pct=r.dark_pct,
            timestamp=r.timestamp,
        )
        for r in rows
    ]


@app.get("/api/alerts", response_model=list[AlertResponse])
def get_alerts(db: Session = Depends(get_db)):
    rows = db.execute(select(Alert).order_by(desc(Alert.created_at))).scalars().all()
    return [
        AlertResponse(
            id=r.id,
            ticker=r.ticker,
            alert_type=r.alert_type,
            threshold=r.threshold,
            status=r.status,
            webhook_url=r.webhook_url,
            created_at=r.created_at,
            triggered_at=r.triggered_at,
        )
        for r in rows
    ]


@app.post("/api/alerts", response_model=AlertResponse, status_code=201)
def create_alert(body: AlertCreate, db: Session = Depends(get_db)):
    alert = Alert(
        ticker=body.ticker.upper(),
        alert_type=body.alert_type,
        threshold=body.threshold,
        webhook_url=body.webhook_url,
        status="active",
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return AlertResponse(
        id=alert.id,
        ticker=alert.ticker,
        alert_type=alert.alert_type,
        threshold=alert.threshold,
        status=alert.status,
        webhook_url=alert.webhook_url,
        created_at=alert.created_at,
        triggered_at=alert.triggered_at,
    )


@app.delete("/api/alerts/{alert_id}", status_code=204)
def delete_alert(alert_id: int, db: Session = Depends(get_db)):
    alert = db.get(Alert, alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    db.delete(alert)
    db.commit()
