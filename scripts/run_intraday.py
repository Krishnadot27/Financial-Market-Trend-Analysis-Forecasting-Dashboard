"""
scripts/run_intraday.py
────────────────────────
Live intraday scanner — runs during NSE market hours.

Scans all 50 NIFTY tickers across 5min, 15min, 1hr timeframes.
Only alerts when signal confidence > 65% (strong signals only).

Usage:
    python scripts/run_intraday.py              # scan once
    python scripts/run_intraday.py --loop       # scan every 15 mins
    python scripts/run_intraday.py --ticker RELIANCE.NS  # single ticker

Schedule with Windows Task Scheduler:
    Run every 15 minutes between 9:15 AM and 3:30 PM on weekdays.
"""

from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime
from loguru import logger

from src.utils import load_config, setup_logger, project_path
from src.intraday import IntradayPredictor, STRONG_SIGNAL_THRESHOLD, NSE_OPEN, NSE_CLOSE
from src.alerts import send_daily_alerts


def is_market_open() -> bool:
    now = datetime.now().time()
    return NSE_OPEN <= now <= NSE_CLOSE


def scan_all_tickers(
    tickers:    list[str],
    config:     dict,
    alert_only: bool = True,
) -> list[dict]:
    """
    Scan all tickers and return strong signals only.

    Parameters
    ----------
    tickers    : list of NSE tickers
    config     : loaded config dict
    alert_only : if True, only return signals above threshold

    Returns
    -------
    list of result dicts sorted by confluence score
    """
    results     = []
    strong_buys  = []
    strong_sells = []

    logger.info(f"Scanning {len(tickers)} tickers across 3 timeframes...")
    logger.info(f"Market open: {is_market_open()}")

    for i, ticker in enumerate(tickers, 1):
        logger.info(f"[{i}/{len(tickers)}] {ticker}")
        try:
            ip     = IntradayPredictor(ticker=ticker, config=config)
            result = ip.predict_all_timeframes()

            if result.get("conf_score", 0) == 0:
                continue

            # Check if any timeframe has strong signal
            has_strong = any(
                r.get("is_alert", False)
                for r in result["timeframes"].values()
                if r.get("status") == "ok"
            )

            if alert_only and not has_strong:
                continue

            results.append(result)

            # Categorize
            conf = result["confluence"]
            if "BUY" in conf:
                strong_buys.append(result)
            elif "SELL" in conf:
                strong_sells.append(result)

        except Exception as e:
            logger.warning(f"  {ticker}: scan failed — {e}")

    # Sort by confluence score descending
    results.sort(key=lambda r: r.get("conf_score", 0), reverse=True)

    logger.info(f"\nScan complete:")
    logger.info(f"  Strong Buys:  {len(strong_buys)}")
    logger.info(f"  Strong Sells: {len(strong_sells)}")
    logger.info(f"  Total alerts: {len(results)}")

    return results


def print_summary(results: list[dict]) -> None:
    """Print clean summary table of all signals."""
    if not results:
        print("\n  No strong signals found in this scan.")
        return

    print(f"\n{'═'*80}")
    print(f"  INTRADAY SCAN RESULTS — {datetime.now().strftime('%d %b %Y %H:%M IST')}")
    print(f"{'─'*80}")
    print(f"  {'TICKER':<16} {'CONFLUENCE':<14} {'5MIN':<10} {'15MIN':<10} {'1HR':<10} {'CONF'}")
    print(f"{'─'*80}")

    for r in results:
        ticker = r["ticker"].replace(".NS", "")
        conf   = r["confluence"]
        score  = r["conf_score"]

        def tf_str(tf):
            res = r["timeframes"].get(tf, {})
            if res.get("status") != "ok":
                return "  —  "
            d   = res["direction"]
            sym = "▲" if d == "UP" else "▼"
            pct = f"{res['confidence']:.0%}"
            return f"{sym} {pct}"

        print(
            f"  {ticker:<16} {conf:<14} "
            f"{tf_str('5min'):<10} {tf_str('15min'):<10} "
            f"{tf_str('1hr'):<10} {score:.0%}"
        )

    print(f"{'═'*80}")


def print_detailed(results: list[dict], top_n: int = 5) -> None:
    """Print detailed report for top N signals."""
    for result in results[:top_n]:
        ip = IntradayPredictor(ticker=result["ticker"], config={})
        ip.print_report(result)


def build_alert_predictions(results: list[dict]) -> tuple[dict, dict]:
    """
    Convert scan results to format expected by send_daily_alerts.
    Returns (predictions_dict, multiday_dict)
    """
    predictions = {}
    multiday    = {}

    for r in results:
        ticker = r["ticker"]
        best   = r.get("best")
        if not best:
            continue

        predictions[ticker] = {
            "direction":  best["direction"],
            "score":      best["raw_score"],
            "price":      best["current_price"],
            "confidence": best["confidence"],
            "note":       f"Intraday ({best['timeframe']}) — {r['confluence']}",
        }
        multiday[ticker] = {
            "3d": "—",
            "5d": "—",
        }

    return predictions, multiday


def main():
    parser = argparse.ArgumentParser(description="NSE Intraday Scanner")
    parser.add_argument("--loop",   action="store_true",
                        help="Run continuously every 15 minutes")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Scan single ticker only")
    parser.add_argument("--all",    action="store_true",
                        help="Show all signals (not just strong)")
    parser.add_argument("--alert",  action="store_true",
                        help="Send WhatsApp/Email alerts for strong signals")
    args = parser.parse_args()

    setup_logger(log_dir="logs")
    config  = load_config()
    tickers = [args.ticker] if args.ticker else config["data"]["tickers"]

    def run_scan():
        logger.info("=" * 60)
        logger.info("  NSE INTRADAY SCANNER")
        logger.info(f"  Time: {datetime.now().strftime('%d %b %Y %H:%M:%S IST')}")
        logger.info("=" * 60)

        results = scan_all_tickers(
            tickers    = tickers,
            config     = config,
            alert_only = not args.all,
        )

        print_summary(results)
        print_detailed(results, top_n=3)

        # Send alerts if requested and there are strong signals
        if args.alert and results:
            strong = [r for r in results if r.get("conf_score", 0) >= 0.8]
            if strong:
                logger.info(f"Sending alerts for {len(strong)} very strong signals...")
                preds, multi = build_alert_predictions(strong)
                send_daily_alerts(
                    predictions = preds,
                    config      = config,
                    regime_info = None,
                    multiday    = multi,
                )

        return results

    if args.loop:
        logger.info("Running in loop mode — scanning every 15 minutes")
        logger.info("Press Ctrl+C to stop")
        while True:
            try:
                run_scan()
                logger.info("Next scan in 15 minutes...")
                time.sleep(15 * 60)
            except KeyboardInterrupt:
                logger.info("Scanner stopped.")
                break
            except Exception as e:
                logger.error(f"Scan error: {e}")
                time.sleep(60)
    else:
        run_scan()


if __name__ == "__main__":
    main()