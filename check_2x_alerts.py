#!/usr/bin/env python3
"""
Check: if you had manually aped every alert (signal that crossed your
alert threshold) and set a sell order at 2x market cap, how many would
have hit it?

Uses price_snapshots (recorded continuously by the bot) to see if market
cap ever reached 2x the entry (signal-time) market cap within a lookback
window.

Usage:
    cd ~/zeen
    python3 check_2x_alerts.py
    python3 check_2x_alerts.py --threshold 8 --days 30
    python3 check_2x_alerts.py --multiple 3 --days 30      # check 3x instead
    python3 check_2x_alerts.py --window-hours 24           # max hold time
    python3 check_2x_alerts.py --min-mc 40000 --max-mc 100000   # filter by entry MC range
    python3 check_2x_alerts.py --out results.txt           # custom output file
"""
import argparse
import os
import sqlite3
import time
from contextlib import closing

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DB_PATH = os.getenv("DB_PATH", "monitor.db")


def main():
    ap = argparse.ArgumentParser(description="Check alert -> Nx outcome using recorded snapshots")
    ap.add_argument("--threshold", type=int, default=int(os.getenv("MONITOR_SCORE_THRESHOLD", 7)),
                     help="min score to count as 'would have alerted' (default: your configured threshold)")
    ap.add_argument("--multiple", type=float, default=2.0, help="target multiple (default 2x)")
    ap.add_argument("--days", type=int, default=30, help="lookback window in days")
    ap.add_argument("--window-hours", type=float, default=24.0,
                     help="max hours after signal to count as a hit (default 24h)")
    ap.add_argument("--min-mc", type=float, default=0.0,
                     help="only include signals with entry market cap >= this (default: no min)")
    ap.add_argument("--max-mc", type=float, default=0.0,
                     help="only include signals with entry market cap <= this (default: no max)")
    ap.add_argument("--db", type=str, default=DB_PATH)
    ap.add_argument("--out", type=str, default="check_2x_results.txt",
                     help="file to write the report to (default: check_2x_results.txt)")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        print(f"ERROR: DB not found at {args.db}")
        return

    since_ts = int(time.time()) - args.days * 86400
    window_sec = int(args.window_hours * 3600)

    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    with closing(sqlite3.connect(args.db)) as conn:
        conn.row_factory = sqlite3.Row

        query = """
            SELECT id, mint, name, symbol, score, probability,
                   market_cap_at_signal, created_at
            FROM signals
            WHERE score >= ?
              AND created_at >= ?
              AND market_cap_at_signal > 0
        """
        params = [args.threshold, since_ts]

        if args.min_mc > 0:
            query += " AND market_cap_at_signal >= ?"
            params.append(args.min_mc)
        if args.max_mc > 0:
            query += " AND market_cap_at_signal <= ?"
            params.append(args.max_mc)

        query += " ORDER BY created_at ASC"

        signals = conn.execute(query, params).fetchall()

        mc_range_str = ""
        if args.min_mc > 0 or args.max_mc > 0:
            lo = f"${args.min_mc:,.0f}" if args.min_mc > 0 else "$0"
            hi = f"${args.max_mc:,.0f}" if args.max_mc > 0 else "∞"
            mc_range_str = f", MC {lo}-{hi}"

        if not signals:
            out(f"No signals with score >= {args.threshold}{mc_range_str} in the last {args.days}d.")
            with open(args.out, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            return

        hits = []
        misses = []
        no_data = []

        for s in signals:
            mint = s["mint"]
            entry_mc = s["market_cap_at_signal"]
            entry_ts = s["created_at"]
            target_mc = entry_mc * args.multiple

            row = conn.execute("""
                SELECT market_cap, created_at
                FROM price_snapshots
                WHERE mint = ?
                  AND created_at >= ?
                  AND created_at <= ?
                  AND market_cap >= ?
                ORDER BY created_at ASC
                LIMIT 1
            """, (mint, entry_ts, entry_ts + window_sec, target_mc)).fetchone()

            peak_row = conn.execute("""
                SELECT MAX(market_cap) AS peak
                FROM price_snapshots
                WHERE mint = ? AND created_at >= ? AND created_at <= ?
            """, (mint, entry_ts, entry_ts + window_sec)).fetchone()

            has_any_data = peak_row and peak_row["peak"] is not None

            name = s["name"] or s["symbol"] or mint[:8]
            if row:
                mins_to_hit = (row["created_at"] - entry_ts) / 60.0
                hits.append((name, s["score"], entry_mc, mins_to_hit, mint))
            elif has_any_data:
                peak_mult = peak_row["peak"] / entry_mc if entry_mc else 0
                misses.append((name, s["score"], entry_mc, peak_mult, mint))
            else:
                no_data.append((name, s["score"], mint))

    total = len(signals)
    n_hit = len(hits)
    n_miss = len(misses)
    n_nodata = len(no_data)
    tracked = n_hit + n_miss

    out("=" * 70)
    out(f"  ALERT -> {args.multiple}x CHECK  (score >= {args.threshold}{mc_range_str}, last {args.days}d, "
        f"hold window {args.window_hours}h)")
    out("=" * 70)
    out(f"  Total alerts        : {total}")
    out(f"  Reached {args.multiple}x        : {n_hit}")
    out(f"  Did NOT reach {args.multiple}x  : {n_miss}")
    out(f"  No snapshot data    : {n_nodata}  (bot wasn't tracking / too recent)")
    if tracked:
        out(f"  Hit rate (of tracked): {n_hit/tracked*100:.1f}%")
    out("-" * 70)

    if hits:
        out(f"\n  HIT {args.multiple}x:")
        for name, score, entry_mc, mins, mint in sorted(hits, key=lambda x: x[3]):
            out(f"    {name:<20} score {score}/10  entry ${entry_mc:,.0f}  "
                f"hit in {mins:.0f}m  [{mint[:8]}]")

    if misses:
        out(f"\n  MISSED {args.multiple}x (best peak shown):")
        for name, score, entry_mc, peak_mult, mint in sorted(misses, key=lambda x: -x[3]):
            out(f"    {name:<20} score {score}/10  entry ${entry_mc:,.0f}  "
                f"peak {peak_mult:.2f}x  [{mint[:8]}]")

    out("=" * 70)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nFull report written to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
