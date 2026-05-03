"""
models/nlp_pipeline.py — News NLP pipeline.

Components:
  1. FinBERT sentiment classifier (fine-tuned on financial text)
  2. T5 summariser (condenses long articles to 2–3 sentences)
  3. Named-entity extraction (ticker/company mentions)
  4. Event classification (earnings, dividend, merger, macro, etc.)

Models are loaded lazily and cached to avoid repeated GPU allocations.
"""
import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import structlog
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    pipeline,
)

logger = structlog.get_logger(__name__)

# Device selection: GPU if available, else CPU
DEVICE = 0 if torch.cuda.is_available() else -1
_executor = ThreadPoolExecutor(max_workers=2)  # CPU-bound inference threads


# ─── Model loaders (cached singletons) ───────────────────────────────────────
@lru_cache(maxsize=1)
def _load_sentiment_pipeline():
    """FinBERT: financial-domain BERT for sentiment classification."""
    logger.info("Loading FinBERT sentiment model")
    return pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        device=DEVICE,
        top_k=None,
        truncation=True,
        max_length=512,
    )


@lru_cache(maxsize=1)
def _load_summarizer():
    """T5-small summariser fine-tuned for news summarisation."""
    logger.info("Loading T5 summariser")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    if DEVICE >= 0:
        model = model.cuda()
    return tokenizer, model


# ─── NLP Pipeline ─────────────────────────────────────────────────────────────
class NLPPipeline:
    """
    Asynchronous NLP pipeline for market news processing.

    All heavy inference runs in a thread-pool executor to avoid
    blocking the FastAPI event loop.
    """

    # Event classification keywords (simple rule-based bootstrap)
    EVENT_PATTERNS = {
        "earnings": r"(lợi nhuận|doanh thu|earnings|revenue|EPS|profit)",
        "dividend": r"(cổ tức|dividend|chia cổ phần)",
        "merger": r"(mua lại|sáp nhập|M&A|merger|acquisition)",
        "leadership": r"(CEO|giám đốc|lãnh đạo|bổ nhiệm|resign|appointed)",
        "macro": r"(lãi suất|tỷ giá|lạm phát|interest rate|inflation|GDP)",
        "regulatory": r"(quy định|luật|regulation|compliance|SEC|SSC)",
        "ipo": r"(IPO|niêm yết|listing|phát hành)",
        "legal": r"(kiện|tố cáo|lawsuit|investigation|vi phạm)",
    }

    async def process_article(self, article: Dict) -> Dict:
        """
        Full NLP enrichment for a news article.

        Input: {"title": ..., "raw_content": ..., "ticker": ..., ...}
        Output: input dict enriched with sentiment, summary, event_tags
        """
        title = article.get("title", "")
        content = article.get("raw_content", "") or ""
        text = f"{title}. {content[:1000]}"   # cap to 1000 chars for inference speed

        # Run inference tasks concurrently in thread pool
        loop = asyncio.get_event_loop()
        sentiment_task = loop.run_in_executor(_executor, self._run_sentiment, text)
        summary_task = loop.run_in_executor(_executor, self._run_summary, content)

        sentiment_result, summary = await asyncio.gather(sentiment_task, summary_task)

        event_tags = self._classify_events(text)
        tickers_mentioned = self._extract_ticker_mentions(text)

        return {
            **article,
            "sentiment_score": sentiment_result["score"],
            "sentiment_label": sentiment_result["label"],
            "summary": summary,
            "event_tags": event_tags,
            "tickers_mentioned": tickers_mentioned,
        }

    def _run_sentiment(self, text: str) -> Dict:
        """
        Run FinBERT and return normalised sentiment score.

        Returns: {"label": "POSITIVE"|"NEGATIVE"|"NEUTRAL", "score": float}
        where score ∈ [-1, 1] (positive for bullish, negative for bearish).
        """
        try:
            clf = _load_sentiment_pipeline()
            results = clf(text[:512])[0]  # list of {label, score}

            # Map FinBERT outputs to a scalar
            label_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
            score = sum(
                label_map.get(r["label"].lower(), 0.0) * r["score"]
                for r in results
            )
            # Dominant label
            dominant = max(results, key=lambda r: r["score"])
            return {
                "label": dominant["label"].upper(),
                "score": round(float(score), 4),
            }
        except Exception as e:
            logger.warning("Sentiment inference failed", error=str(e))
            return {"label": "NEUTRAL", "score": 0.0}

    def _run_summary(self, text: str, max_new_tokens: int = 80) -> str:
        """
        Summarise article content using T5.
        Returns a 1–2 sentence summary.
        """
        if not text or len(text.strip()) < 100:
            return text.strip()[:200]

        try:
            tokenizer, model = _load_summarizer()
            input_text = "summarize: " + text[:1024]
            inputs = tokenizer.encode(
                input_text, return_tensors="pt", max_length=512, truncation=True
            )
            if DEVICE >= 0:
                inputs = inputs.cuda()

            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    min_length=20,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True,
                )
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            logger.warning("Summarisation failed", error=str(e))
            return text[:200]

    def _classify_events(self, text: str) -> List[str]:
        """Rule-based event tag classification (bilingual: Vietnamese + English)."""
        tags = []
        text_lower = text.lower()
        for event, pattern in self.EVENT_PATTERNS.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                tags.append(event)
        return tags

    def _extract_ticker_mentions(self, text: str) -> List[str]:
        """
        Extract Vietnam stock ticker mentions from text.
        Tickers: 3-character uppercase codes (HOSE/HNX format).
        e.g., VNM, VIC, TCB, HPG, MSN
        """
        # Pattern: word boundaries around 2–4 uppercase letter sequences
        raw = re.findall(r"\b([A-Z]{2,4})\b", text)
        # Filter to plausible tickers (not common English acronyms)
        EXCLUDED = {"GDP", "CEO", "CFO", "M&A", "USD", "VND", "EPS", "PE", "IPO", "ETF"}
        return list(set(t for t in raw if t not in EXCLUDED))

    async def batch_process(self, articles: List[Dict]) -> List[Dict]:
        """Process multiple articles concurrently."""
        tasks = [self.process_article(a) for a in articles]
        return await asyncio.gather(*tasks, return_exceptions=False)

    def compute_aggregate_sentiment(
        self,
        articles: List[Dict],
        decay_halflife_days: int = 7,
    ) -> float:
        """
        Compute time-decayed aggregate sentiment score for a stock.
        More recent articles weighted higher (exponential decay).
        """
        from datetime import datetime
        import math

        if not articles:
            return 0.0

        now = datetime.utcnow()
        total_weight = 0.0
        weighted_sum = 0.0

        for article in articles:
            published = article.get("published_at")
            if published:
                if isinstance(published, str):
                    published = datetime.fromisoformat(published.replace("Z", "+00:00"))
                    published = published.replace(tzinfo=None)
                age_days = max(0, (now - published).days)
                weight = math.exp(-age_days * math.log(2) / decay_halflife_days)
            else:
                weight = 0.5

            score = article.get("sentiment_score", 0.0) or 0.0
            weighted_sum += weight * score
            total_weight += weight

        return float(weighted_sum / total_weight) if total_weight > 0 else 0.0
