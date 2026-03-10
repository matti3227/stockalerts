"""FinBERT-based sentiment analysis for financial text.

ProsusAI/finbert is a BERT model fine-tuned on financial news and StockTwits
text. It returns three-class probabilities (positive / negative / neutral).

The model is lazy-loaded on the first call (~440 MB download on first run).
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline as _hf_pipeline  # noqa: PLC0415

        logger.info("Loading FinBERT model (first run may download ~440 MB)…")
        _pipeline = _hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=None,       # return all three class probabilities
            truncation=True,
            max_length=512,
            device=-1,        # CPU
        )
        logger.info("FinBERT ready.")
    return _pipeline


def analyze(
    text: str,
    author_label: Optional[str] = None,
) -> tuple[float, str, float]:
    """Analyse sentiment of financial text.

    Returns ``(score, label, confidence)`` where:
    - ``score``      — signed float: + for bullish, - for bearish, 0 for neutral
    - ``label``      — ``'bullish'``, ``'bearish'``, or ``'neutral'``
    - ``confidence`` — FinBERT max-class probability (0.0–1.0)

    If *author_label* is ``'Bullish'`` / ``'Bearish'`` (StockTwits tag),
    it is trusted directly with ``confidence=1.0``.
    """
    if author_label:
        norm = author_label.lower()
        if norm == "bullish":
            return 1.0, "bullish", 1.0
        if norm == "bearish":
            return -1.0, "bearish", 1.0

    if not text or not text.strip():
        return 0.0, "neutral", 0.5

    try:
        pipe = _get_pipeline()
        results = pipe(text[:512])[0]   # list of {label, score} dicts
        by_label = {r["label"].lower(): r["score"] for r in results}

        pos  = by_label.get("positive", 0.0)
        neg  = by_label.get("negative", 0.0)
        neu  = by_label.get("neutral",  0.0)
        conf = max(pos, neg, neu)

        if pos >= neg and pos >= neu:
            return pos, "bullish", conf
        if neg >= pos and neg >= neu:
            return -neg, "bearish", conf
        return 0.0, "neutral", conf
    except Exception:
        logger.exception("FinBERT analysis failed; returning neutral")
        return 0.0, "neutral", 0.5
