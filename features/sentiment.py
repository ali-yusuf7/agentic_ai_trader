"""
Sentiment Analysis - Analyze news headlines and articles for market sentiment
"""
import logging
import re
from typing import Dict, List, Optional

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

# Download VADER lexicon if not present
try:
    nltk.data.find("vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)


class SentimentAnalyzer:
    """Analyze sentiment from news headlines and articles."""

    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()

        # Financial-specific sentiment words to boost VADER
        self.financial_positive = {
            "beat", "beats", "exceeds", "exceeded", "upgrade", "upgraded",
            "bull", "bullish", "rally", "rallies", "soar", "soars", "surge",
            "surges", "gain", "gains", "profit", "profits", "growth", "growing",
            "strong", "stronger", "outperform", "outperforms", "buy", "buying",
            "accumulate", "breakout", "breakthrough", "record", "high", "higher"
        }

        self.financial_negative = {
            "miss", "misses", "missed", "downgrade", "downgraded", "bear",
            "bearish", "crash", "crashes", "plunge", "plunges", "drop", "drops",
            "fall", "falls", "loss", "losses", "decline", "declining", "weak",
            "weaker", "underperform", "underperforms", "sell", "selling",
            "bankruptcy", "default", "lawsuit", "investigation", "fraud",
            "recall", "warning", "layoff", "layoffs", "cut", "cuts"
        }

        self.financial_neutral = {
            "hold", "holds", "holding", "steady", "unchanged", "flat",
            "sideways", "consolidate", "consolidating", "range", "ranging"
        }

    def analyze_text(self, text: str) -> dict:
        """
        Analyze sentiment of a text string.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment scores
        """
        if not text:
            return {"compound": 0, "positive": 0, "negative": 0, "neutral": 1}

        # Get VADER scores
        scores = self.vader.polarity_scores(text)

        # Boost based on financial keywords
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))

        positive_count = len(words & self.financial_positive)
        negative_count = len(words & self.financial_negative)

        # Adjust compound score based on financial keywords
        keyword_adjustment = (positive_count - negative_count) * 0.1
        scores["compound"] = max(-1, min(1, scores["compound"] + keyword_adjustment))

        # Add financial keyword counts
        scores["financial_positive_keywords"] = positive_count
        scores["financial_negative_keywords"] = negative_count

        return scores

    def analyze_headline(self, headline: str) -> dict:
        """
        Analyze a news headline with extra weight on key terms.

        Args:
            headline: News headline text

        Returns:
            Dictionary with sentiment analysis
        """
        result = self.analyze_text(headline)
        result["text"] = headline
        result["type"] = "headline"
        return result

    def analyze_article(self, title: str, summary: str = "") -> dict:
        """
        Analyze a full news article.

        Args:
            title: Article title
            summary: Article summary or body

        Returns:
            Combined sentiment analysis
        """
        title_sentiment = self.analyze_text(title)
        summary_sentiment = self.analyze_text(summary) if summary else title_sentiment

        # Weight title more heavily (60% title, 40% summary)
        combined_compound = (
            title_sentiment["compound"] * 0.6 +
            summary_sentiment["compound"] * 0.4
        )

        return {
            "compound": combined_compound,
            "title_sentiment": title_sentiment["compound"],
            "summary_sentiment": summary_sentiment["compound"],
            "positive": (title_sentiment["pos"] + summary_sentiment["pos"]) / 2,
            "negative": (title_sentiment["neg"] + summary_sentiment["neg"]) / 2,
            "neutral": (title_sentiment["neu"] + summary_sentiment["neu"]) / 2,
            "title": title,
            "type": "article"
        }

    def analyze_news_batch(self, articles: List[dict]) -> dict:
        """
        Analyze a batch of news articles and return aggregate sentiment.

        Args:
            articles: List of article dictionaries with 'title' and 'summary' keys

        Returns:
            Aggregate sentiment analysis
        """
        if not articles:
            return {
                "overall_sentiment": 0,
                "sentiment_label": "neutral",
                "article_count": 0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "articles": []
            }

        article_sentiments = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for article in articles:
            title = article.get("title", "")
            summary = article.get("summary", "")

            sentiment = self.analyze_article(title, summary)
            article_sentiments.append(sentiment)

            if sentiment["compound"] > 0.05:
                positive_count += 1
            elif sentiment["compound"] < -0.05:
                negative_count += 1
            else:
                neutral_count += 1

        # Calculate overall sentiment (average of all articles)
        overall = sum(s["compound"] for s in article_sentiments) / len(article_sentiments)

        # Determine label
        if overall > 0.1:
            label = "bullish"
        elif overall > 0.05:
            label = "slightly_bullish"
        elif overall < -0.1:
            label = "bearish"
        elif overall < -0.05:
            label = "slightly_bearish"
        else:
            label = "neutral"

        return {
            "overall_sentiment": overall,
            "sentiment_label": label,
            "article_count": len(articles),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "articles": article_sentiments
        }

    def get_sentiment_features(self, articles: List[dict]) -> dict:
        """
        Extract sentiment features for ML model input.

        Args:
            articles: List of news articles

        Returns:
            Dictionary of sentiment features
        """
        batch_analysis = self.analyze_news_batch(articles)

        total = batch_analysis["article_count"] or 1

        return {
            "sentiment_score": batch_analysis["overall_sentiment"],
            "positive_ratio": batch_analysis["positive_count"] / total,
            "negative_ratio": batch_analysis["negative_count"] / total,
            "neutral_ratio": batch_analysis["neutral_count"] / total,
            "news_volume": batch_analysis["article_count"],
            "sentiment_label": batch_analysis["sentiment_label"]
        }

    def get_sentiment_signal(self, sentiment_score: float) -> str:
        """
        Convert sentiment score to trading signal.

        Args:
            sentiment_score: Sentiment score (-1 to 1)

        Returns:
            Signal string: 'buy', 'sell', or 'hold'
        """
        if sentiment_score > 0.15:
            return "buy"
        elif sentiment_score < -0.15:
            return "sell"
        else:
            return "hold"

    def explain_sentiment(self, text: str) -> str:
        """
        Provide human-readable explanation of sentiment analysis.

        Args:
            text: Text that was analyzed

        Returns:
            Explanation string
        """
        analysis = self.analyze_text(text)

        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))

        found_positive = words & self.financial_positive
        found_negative = words & self.financial_negative

        explanation = []

        if analysis["compound"] > 0.05:
            explanation.append(f"Overall sentiment: POSITIVE ({analysis['compound']:.2f})")
        elif analysis["compound"] < -0.05:
            explanation.append(f"Overall sentiment: NEGATIVE ({analysis['compound']:.2f})")
        else:
            explanation.append(f"Overall sentiment: NEUTRAL ({analysis['compound']:.2f})")

        if found_positive:
            explanation.append(f"Positive keywords: {', '.join(found_positive)}")
        if found_negative:
            explanation.append(f"Negative keywords: {', '.join(found_negative)}")

        return " | ".join(explanation)
