"""Keyword-based financial sentiment analysis.

Fast, zero-dependency approach using curated bull/bear word lists tuned for
financial social media (StockTwits, Reddit, news headlines).

Returns the same (score, label, confidence) interface as the previous FinBERT
implementation so all callers remain unchanged.
"""
import re
from typing import Optional

_BULL_WORDS = {
    "bullish", "buy", "buying", "bought", "long", "calls", "call",
    "moon", "mooning", "rocket", "rip", "breakout", "surge", "surging",
    "rally", "rallying", "gain", "gains", "winning", "win", "pump",
    "pumping", "up", "upside", "upgrade", "upgraded", "outperform",
    "beat", "beats", "beat expectations", "strong", "strength",
    "growth", "growing", "profit", "profits", "revenue", "revenues",
    "positive", "bull", "bulls", "accumulate", "accumulating",
    "support", "bounce", "bouncing", "green", "higher", "high",
    "cheap", "undervalued", "opportunity", "opportunities", "hold",
    "hodl", "conviction", "confident", "optimistic", "bright",
    "recover", "recovery", "recovered", "bottom", "bottomed",
    "oversold", "rebound", "rebounding", "squeeze", "short squeeze",
    "all time high", "ath", "record", "earnings beat",
}

_BEAR_WORDS = {
    "bearish", "sell", "selling", "sold", "short", "puts", "put",
    "crash", "crashing", "dump", "dumping", "drop", "dropping",
    "fall", "falling", "fell", "decline", "declining", "loss",
    "losses", "losing", "lose", "down", "downside", "downgrade",
    "downgraded", "underperform", "miss", "misses", "missed",
    "weak", "weakness", "shrink", "shrinking", "negative", "bear",
    "bears", "distribution", "distributing", "resistance",
    "red", "lower", "lower", "overvalued", "expensive",
    "concerned", "worry", "worried", "pessimistic", "trouble",
    "troubled", "debt", "risk", "risks", "risky", "warning",
    "bankruptcy", "bankrupt", "collapse", "collapsing",
    "overbought", "bubble", "correction", "corrections",
    "layoffs", "layoff", "miss expectations", "earnings miss",
    "lawsuit", "fraud", "investigation", "sec", "fine", "penalty",
}

_TOKENIZE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def analyze(
    text: str,
    author_label: Optional[str] = None,
) -> tuple[float, str, float]:
    """Analyse sentiment of financial text using keyword matching.

    Returns ``(score, label, confidence)`` where:
    - ``score``      — float in [-1, +1]; positive = bullish, negative = bearish
    - ``label``      — ``'bullish'``, ``'bearish'``, or ``'neutral'``
    - ``confidence`` — signal strength (0.0–1.0)

    If *author_label* is supplied (StockTwits author tag), it is trusted
    directly with ``confidence=1.0``.
    """
    if author_label:
        norm = author_label.lower()
        if norm == "bullish":
            return 1.0, "bullish", 1.0
        if norm == "bearish":
            return -1.0, "bearish", 1.0

    if not text or not text.strip():
        return 0.0, "neutral", 0.5

    tokens = _TOKENIZE.findall(text.lower())
    if not tokens:
        return 0.0, "neutral", 0.5

    bull = sum(1 for t in tokens if t in _BULL_WORDS)
    bear = sum(1 for t in tokens if t in _BEAR_WORDS)
    total = bull + bear

    if total == 0:
        return 0.0, "neutral", 0.5

    score = (bull - bear) / total          # [-1, +1]
    confidence = min(total / 5.0, 1.0)    # saturates at 5 signal words

    if score > 0.1:
        return round(score, 4), "bullish", round(confidence, 4)
    if score < -0.1:
        return round(score, 4), "bearish", round(confidence, 4)
    return round(score, 4), "neutral", round(confidence, 4)
