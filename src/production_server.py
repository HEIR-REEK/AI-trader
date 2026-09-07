"""
Production Server — Continuous Analysis Service.
Runs the master engine at configured intervals, persists results,
and exposes health/status endpoints.
Requires all production environment variables set.
NOTE: This is an analysis service — NOT a trading execution system.
"""

import os
import time
import logging
from datetime import datetime

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Ensure logs/ directory exists before FileHandler tries to open it
os.makedirs("logs", exist_ok=True)

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/production.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

from master_engine import MasterSignalEngine
from realtime_feeds import RealTimeFeeds
from production_db import ProductionDB


class ProductionService:
    """Continuous production analysis service."""

    def __init__(self):
        # Explicit check: all environment variables must be set
        required_vars = ["EXCHANGE_API_KEY", "NEWS_API_KEY", "VIX_API_KEY"]
        missing = [v for v in required_vars if not os.environ.get(v)]
        if missing:
            raise ValueError(f"Missing required environment variables for production: {missing}. Set them in .env or environment.")

        self.engine = MasterSignalEngine()
        self.db = ProductionDB()
        self.interval = int(os.environ.get("ANALYSIS_INTERVAL_MINUTES", "5"))
        logger.info(f"Production service initialized. Analysis interval: {self.interval} minutes.")

    def run_cycle(self, pair: str = "EUR/USD"):
        logger.info(f"Starting analysis cycle for {pair}")
        try:
            result = self.engine.analyze(pair)
            self.db.save_analysis(result)
            logger.info(f"Analysis complete for {pair}: dominant={result.get('dominant_strategy')} score={result.get('dominant_score')}")
            return result
        except Exception as e:
            logger.error(f"Analysis cycle failed for {pair}: {e}")
            # Persist failure for audit
            self.db.save_feed_health({
                "feeds_active": False,
                "rate_feed_source": "error",
                "vol_feed_source": "error",
                "warning": f"Analysis failure: {str(e)}",
            })
            raise

    def health_check(self) -> dict:
        try:
            feeds = RealTimeFeeds()
            health = feeds.feed_health_check()
            latest = self.db.get_latest_analysis()
            return {
                "status": "healthy" if health.get("feeds_active") else "degraded",
                "feeds_active": health.get("feeds_active"),
                "latest_analyses": len(latest),
                "timestamp": datetime.now().isoformat(),
                "message": "Production analysis service running. This provides structured analysis — NOT guaranteed trading predictions.",
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "feeds_active": False,
                "timestamp": datetime.now().isoformat(),
                "message": f"Service unhealthy: {str(e)}",
            }

    def start_continuous(self, pair: str = "EUR/USD"):
        logger.info("Starting continuous production analysis service.")
        logger.info("WARNING: This service provides structured analysis points only. It does NOT guarantee profitable trades.")
        while True:
            try:
                self.run_cycle(pair)
            except Exception as e:
                logger.error(f"Cycle error (service continues): {e}")
            time.sleep(self.interval * 60)


if __name__ == "__main__":
    service = ProductionService()
    # Run a single demonstration cycle (production services run continuously in deployment)
    result = service.run_cycle("EUR/USD")
    print(f"Production cycle complete. Dominant strategy: {result.get('dominant_strategy')}")
    print(f"Master warning embedded: {bool(result.get('master_warning'))}")
    health = service.health_check()
    print(f"Service health: {health['status']}")
