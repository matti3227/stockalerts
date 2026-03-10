from datetime import date, datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)

from app.database import Base


class Post(Base):
    """A single post scraped from StockTwits, Reddit, or a news feed."""

    __tablename__ = "posts"

    id          = Column(Integer, primary_key=True)
    external_id = Column(String(255), unique=True, nullable=False, index=True)
    text        = Column(Text, nullable=False)
    source      = Column(String(50), nullable=False)   # stocktwits | reddit | news
    author      = Column(String(255))
    engagement  = Column(Integer, default=0)           # likes / upvotes
    url         = Column(String(500))
    created_at  = Column(DateTime, nullable=False)
    scraped_at  = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_posts_source_created", "source", "created_at"),
    )


class Mention(Base):
    """One ticker mentioned inside a Post, with its sentiment reading."""

    __tablename__ = "mentions"

    id                   = Column(Integer, primary_key=True)
    post_id              = Column(Integer, ForeignKey("posts.id", ondelete="CASCADE"), nullable=False)
    ticker               = Column(String(10), nullable=False)
    sentiment_score      = Column(Float)    # signed: + bullish, - bearish
    sentiment_label      = Column(String(20))
    sentiment_confidence = Column(Float)
    created_at           = Column(DateTime, nullable=False)

    __table_args__ = (
        Index("ix_mentions_ticker_created", "ticker", "created_at"),
        Index("ix_mentions_post_ticker", "post_id", "ticker"),
    )


class Alert(Base):
    """User-defined alert rule."""

    __tablename__ = "alerts"

    id           = Column(Integer, primary_key=True)
    ticker       = Column(String(10), nullable=False, index=True)
    alert_type   = Column(String(50), nullable=False)  # mention_spike | bullish | bearish
    threshold    = Column(Float, nullable=False)
    status       = Column(String(20), default="active", nullable=False)  # active|triggered|paused
    webhook_url  = Column(String(500))
    created_at   = Column(DateTime, default=datetime.utcnow, nullable=False)
    triggered_at = Column(DateTime)


class CongressionalTrade(Base):
    """A stock trade disclosed by a US House or Senate member."""

    __tablename__ = "congressional_trades"

    id               = Column(Integer, primary_key=True)
    external_id      = Column(String(512), unique=True, nullable=False, index=True)
    politician       = Column(String(255), nullable=False)
    chamber          = Column(String(10))    # house | senate
    party            = Column(String(10))
    ticker           = Column(String(10), nullable=False, index=True)
    buy_sell         = Column(String(10), nullable=False)   # buy | sell
    amount           = Column(String(100))   # e.g. "$50,001 - $100,000"
    trade_date       = Column(Date)
    disclosure_date  = Column(Date)
    scraped_at       = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_cong_ticker_date", "ticker", "trade_date"),
    )


class DarkPool(Base):
    """Dark-pool / off-exchange volume data from FINRA ATS transparency."""

    __tablename__ = "dark_pools"

    id           = Column(Integer, primary_key=True)
    ticker       = Column(String(10), nullable=False)
    volume       = Column(BigInteger, nullable=False)   # dark/ATS share volume
    short_volume = Column(BigInteger)                   # FINRA short-sale volume
    total_volume = Column(BigInteger)
    dark_pct     = Column(Float)                        # volume / total_volume
    timestamp    = Column(DateTime, nullable=False)     # week start date
    source       = Column(String(50), default="finra")

    __table_args__ = (
        Index("ix_darkpool_ticker_ts", "ticker", "timestamp"),
    )
