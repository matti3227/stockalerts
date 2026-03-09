from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from app.database import Base


class Post(Base):
    """A single post scraped from Reddit or StockTwits."""

    __tablename__ = "posts"

    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String(255), unique=True, nullable=False, index=True)
    source = Column(String(50), nullable=False)  # 'reddit' | 'stocktwits'
    content = Column(Text, nullable=False)
    author = Column(String(255))
    score = Column(Integer, default=0)
    url = Column(String(500))
    sentiment_score = Column(Float)   # VADER compound: -1.0 to 1.0
    sentiment_label = Column(String(20))  # 'bullish' | 'bearish' | 'neutral'
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    scraped_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    mentions = relationship(
        "Mention", back_populates="post", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_posts_source_created", "source", "created_at"),
    )


class Mention(Base):
    """Links a ticker symbol to the post that mentions it."""

    __tablename__ = "mentions"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(10), nullable=False, index=True)
    post_id = Column(
        Integer, ForeignKey("posts.id", ondelete="CASCADE"), nullable=False
    )
    context = Column(Text)  # short excerpt around the mention

    post = relationship("Post", back_populates="mentions")

    __table_args__ = (
        Index("ix_mentions_ticker_post", "ticker", "post_id"),
    )


class TickerMetric(Base):
    """Aggregated sentiment metrics for a ticker at a point in time."""

    __tablename__ = "ticker_metrics"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(10), nullable=False, index=True)
    mention_count = Column(Integer, default=0)
    avg_sentiment = Column(Float, default=0.0)
    bullish_count = Column(Integer, default=0)
    bearish_count = Column(Integer, default=0)
    neutral_count = Column(Integer, default=0)
    calculated_at = Column(
        DateTime, default=datetime.utcnow, nullable=False, index=True
    )


class Alert(Base):
    """User-defined alert rule for a ticker."""

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(10), nullable=False, index=True)
    # condition: 'mention_spike' | 'sentiment_bullish' | 'sentiment_bearish'
    condition = Column(String(50), nullable=False)
    threshold = Column(Float, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    webhook_url = Column(String(500))
    last_triggered = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
