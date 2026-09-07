"""Start the AI-Trader browser interface (no terminal trading needed).

Usage:
    python run_web.py            # open http://localhost:8000
    python run_web.py --port 8080
"""
from ai_trader.web.__main__ import main

if __name__ == "__main__":
    main()
