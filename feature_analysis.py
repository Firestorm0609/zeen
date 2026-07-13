#!/usr/bin/env python3
"""
Feature separation analysis: for every recorded signal that has a labeled
outcome (via lookbacks), decode its stored feature_vector and check which
individual features actually differ between "good" outcomes (PUMP/MOON) and
"bad" outcomes (RUG / severe collapse), plus correlation with pct_change.

This tells you which of your pre-entry filters (bundle_clean,
mint_auth_safety, holder_distribution, etc.) are actually doing useful work
separating winners from instant rugs, vs which are just noise.

Read-only. Does not modify your DB.

Usage:
    cd ~/zeen
    python3 feature_analysis.py
    python3 feature_analysis.py --window 4hr --days 100
    python3 feature_analysis.py --gap-threshold -70   # define "instant rug" cutoff
"""
import argparse
import json
import math
import os
import sqlite3
import sys
from contextlib import closing

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DB_PATH = os.getenv("DB_PATH", "monitor.db")
DEFAULT_WINDOW = os.getenv("ML_LABEL_WINDOW", "1hr")

# Must match features.py registration order exactly.
FEATURE_NAMES = [
    "mc_log", "mc_zscore", "mc_percentile",
    "reply_log", "reply_percentile", "replies_per_kmc",
    "has_twitter", "has_telegram", "has_website",
    "social_count", "social_completeness",
    "desc_len_log", "desc_word_count_log", "desc_has_url",
    "desc_exclamation_density", "desc_unique_word_ratio", "desc_uppercase_ratio",
    "name_len", "symbol_len", "name_has_numbers", "name_is_allcaps",
    "name_symbol_length_ratio",
    "hour_sin", "hour_cos", "dow_sin",
    "keyword_pump_score", "keyword_positive_hits", "keyword_negative_hits",
    "coin_age_log", "coin_age_capped", "reply_velocity",
    "mc_momentum_pct", "bonding_curve_progress",
    "mint_auth_safety", "freeze_auth_safety", "holder_distribution",
    "bundle_clean", "creator_freshness", "creator_blacklisted",
    "creator_holding_safety",
]


def safe_float(v, default=0.0):
    try:
        f = float(v) if v is not None else 0.0
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def stdev(vals):
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    return math.sqrt(var)


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = mean(xs), mean(ys)
    sx, sy = stdev(xs), stdev(ys)
    if sx == 0 or sy == 0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    return cov / (sx * sy)


def corr_zscore(r, n):
    """Fisher z-transform significance test for a correlation coefficient.
    With large n, even small |r| can be statistically real rather than noise.
    Returns an approximate z-score; |z| >= 1.96 ~ p < 0.05, |z| >= 2.58 ~ p < 0.01.
    """
    if n < 4 or abs(r) >= 1.0:
        return 0.0
    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1.0 / math.sqrt(n - 3)
    return z / se


def load_rows(conn, window, since_ts):
    return conn.execute("""
        SELECT s.feature_vector, s.name, s.symbol, s.score, s.probability,
               lb.outcome, lb.pct_change
        FROM signals s
        JOIN lookbacks lb ON lb.signal_id = s.id
        WHERE lb.window_label = ?
          AND lb.checked = 1
          AND lb.pct_change IS NOT NULL
          AND s.feature_vector IS NOT NULL
          AND s.created_at >= ?
    """, (window, since_ts)).fetchall()


def decode_fvec(raw, n_expected):
    try:
        fvec = json.loads(raw)
        if not isinstance(fvec, list):
            return None
        if len(fvec) < n_expected:
            fvec = fvec + [0.0] * (n_expected - len(fvec))
        elif len(fvec) > n_expected:
            fvec = fvec[:n_expected]
        return [safe_float(v) for v in fvec]
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Analyze which features separate rugs from winners")
    ap.add_argument("--days", type=int, default=100, help="lookback window in days")
    ap.add_argument("--window", type=str, default=DEFAULT_WINDOW,
                    help=f"lookback label to use (default from .env: {DEFAULT_WINDOW})")
    ap.add_argument("--db", type=str, default=DB_PATH)
    ap.add_argument("--gap-threshold", type=float, default=-70.0,
                    help="pct_change at/below this = 'instant rug' bucket (default -70)")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        print(f"ERROR: DB not found at {args.db}", file=sys.stderr)
        sys.exit(1)

    import time
    since_ts = int(time.time()) - args.days * 86400

    with closing(sqlite3.connect(args.db)) as conn:
        conn.row_factory = sqlite3.Row
        rows = load_rows(conn, args.window, since_ts)

    if not rows:
        print(f"No labeled signals found for window='{args.window}' in last {args.days}d.")
        print("Try --window 4hr or --window 24hr if 1hr has too few labeled rows.")
        return

    n_feat = len(FEATURE_NAMES)
    good_vecs, rug_vecs, other_vecs = [], [], []
    all_vecs, all_pct = [], []
    decode_fail = 0

    for row in rows:
        fvec = decode_fvec(row["feature_vector"], n_feat)
        if fvec is None:
            decode_fail += 1
            continue
        pct = safe_float(row["pct_change"])
        outcome = row["outcome"] or ""

        all_vecs.append(fvec)
        all_pct.append(pct)

        is_rug = (outcome == "RUG") or (pct <= args.gap_threshold)
        is_good = outcome in ("PUMP", "MOON")

        if is_rug:
            rug_vecs.append(fvec)
        elif is_good:
            good_vecs.append(fvec)
        else:
            other_vecs.append(fvec)

    print("=" * 72)
    print(f"  FEATURE SEPARATION ANALYSIS  —  window={args.window}  last {args.days}d")
    print("=" * 72)
    print(f"  Total labeled signals : {len(rows)}  ({decode_fail} failed to decode)")
    print(f"  Rug/collapse bucket   : {len(rug_vecs)}  (outcome=RUG or pct<={args.gap_threshold}%)")
    print(f"  Good bucket           : {len(good_vecs)}  (outcome=PUMP/MOON)")
    print(f"  Other/STALE/DOWN      : {len(other_vecs)}")

    if len(rug_vecs) < 3 or len(good_vecs) < 3:
        print("\n  Not enough samples in both buckets for a meaningful comparison yet.")
        print("  (need at least ~3 each — try a longer --days window or different --window label)")
    else:
        print("-" * 72)
        print(f"  {'feature':<26} {'rug avg':>9} {'good avg':>9} {'gap':>8}  {'corr':>7} {'z-score':>9} {'sig?':>5}")
        print("-" * 72)

        results = []
        for i, name in enumerate(FEATURE_NAMES):
            rug_col  = [v[i] for v in rug_vecs]
            good_col = [v[i] for v in good_vecs]
            all_col  = [v[i] for v in all_vecs]

            rug_avg  = mean(rug_col)
            good_avg = mean(good_col)
            gap      = good_avg - rug_avg
            corr     = pearson(all_col, all_pct)
            z        = corr_zscore(corr, len(all_col))

            results.append((name, rug_avg, good_avg, gap, corr, z))

        # Sort by statistical significance (|z-score|), not raw correlation —
        # at this sample size a tiny but consistent correlation can be more
        # trustworthy than a larger one built on noise.
        results.sort(key=lambda r: -abs(r[5]))

        for name, rug_avg, good_avg, gap, corr, z in results:
            sig = "**" if abs(z) >= 2.58 else ("*" if abs(z) >= 1.96 else "")
            print(f"  {name:<26} {rug_avg:9.3f} {good_avg:9.3f} {gap:8.3f}  {corr:7.3f} {z:9.2f} {sig:>5}")

        print("-" * 72)
        print("  Reading this:")
        print("  - 'z-score' tests whether the correlation is distinguishable from")
        print("    zero given your sample size. ** = p<0.01 (very likely real),")
        print("    *  = p<0.05 (likely real), blank = not distinguishable from noise.")
        print("  - Statistical significance != practical usefulness. A feature can")
        print("    be '**' significant with |corr| of only 0.03 — real, but weak.")
        print("    At this scale, focus on features that are BOTH significant AND")
        print("    have the largest |corr| — those are your best formula-weight")
        print("    candidates. Features with no significance mark are prime")
        print("    candidates to strip from FORMULA_WEIGHTS entirely.")
        print("  - 'gap' = good-outcome avg minus rug avg — a quick sanity check")
        print("    that points the same direction as corr.")

    print("=" * 72)


if __name__ == "__main__":
    main()
