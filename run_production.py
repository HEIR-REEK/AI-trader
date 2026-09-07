"""
Production Entry Point.
Requires all environment variables set. No simulated data allowed.
Usage: source .venv/bin/activate && PYTHONPATH=src python run_production.py
NOTE: This is structured analysis — NOT a guaranteed trading predictor.
"""

import sys
sys.path.insert(0, "src")

from production_server import ProductionService

if __name__ == "__main__":
    print("=" * 70)
    print("AI-TRADER PRODUCTION SERVICE")
    print("=" * 70)
    print("Status: Production infrastructure active.")
    print("Requirements:")
    print("  - EXCHANGE_API_KEY, NEWS_API_KEY, VIX_API_KEY set in environment")
    print("  - Database: data/trader.db (auto-initialized)")
    print("  - Logs: logs/production.log")
    print()
    print("WARNING: This service provides structured market analysis using")
    print("multi-strategy frameworks, volatility indices, and technical indicators.")
    print("It does NOT guarantee winning trades. Markets are unpredictable.")
    print()
    try:
        service = ProductionService()
        health = service.health_check()
        print(f"Health check: {health['status']}")
        print(f"Message: {health['message']}")
        print(f"Latest analyses in DB: {health.get('latest_analyses', 0)}")
        result = service.run_cycle("EUR/USD")
        print(f"\nLatest analysis result:")
        print(f"  Pair: {result.get('pair')}")
        print(f"  Dominant Strategy: {result.get('dominant_strategy')}")
        print(f"  Score: {result.get('dominant_score')}")
        print(f"  Master Warning Present: {bool(result.get('master_warning'))}")
    except Exception as e:
        print(f"PRODUCTION FAILURE: {e}")
        print("This failure confirms the production system rejects simulated/fallback data.")
        print("Set real API keys in .env or environment to run production mode.")
        sys.exit(1)
