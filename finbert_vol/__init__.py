"""finbert_vol — class-based pipeline: news + BSE announcements + earnings calls → sentiment → daily features → volatility.

    from finbert_vol import Settings, Pipeline
    p = Pipeline(Settings.load())          # settings from finbert_vol.yaml if present
    p.score_news(); p.score_calls(); p.build_features(); p.evaluate(); p.predict_latest()

CLI: python -m finbert_vol {score-news,score-calls,features,evaluate,predict,all}
The repo's older scripts (news/, model/) are reused, not duplicated.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("news", "model", "bert"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from .config import Settings                      # noqa: E402
from .models import SentimentModel                # noqa: E402
from .calls import TranscriptParser, CallFeatures  # noqa: E402
from .sources import NewsCorpus, BSEAnnouncements, Transcripts  # noqa: E402
from .scorers import NewsScorer, CallScorer, AnnouncementScorer  # noqa: E402
from .features import FeatureBuilder              # noqa: E402
from .volatility import VolatilityModel, Evaluator  # noqa: E402
from .pipeline import Pipeline                    # noqa: E402

__all__ = ["ROOT", "Settings", "SentimentModel", "TranscriptParser", "CallFeatures", "NewsCorpus", "BSEAnnouncements",
           "Transcripts", "NewsScorer", "CallScorer", "AnnouncementScorer", "FeatureBuilder", "VolatilityModel",
           "Evaluator", "Pipeline"]
