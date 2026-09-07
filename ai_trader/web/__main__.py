"""`python -m ai_trader.web` — start the browser interface."""
from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-Trader browser interface")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes (dev)")
    args = parser.parse_args()

    import uvicorn

    print("=" * 70)
    print("AI-TRADER — BROWSER INTERFACE")
    print("=" * 70)
    print(f"Open in your browser:  http://localhost:{args.port}")
    print(f"API docs (optional):   http://localhost:{args.port}/api/docs")
    print()
    print("Decision support only — no trade is guaranteed.")
    print("Press Ctrl+C to stop.")
    print("=" * 70)
    uvicorn.run("ai_trader.web.app:app", host=args.host, port=args.port, reload=args.reload,
                log_level="info")


if __name__ == "__main__":
    main()
