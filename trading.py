"""Dynamic exit parameter computation, creator history, and auto-blacklist.

Provides:
  - compute_dynamic_exit_params  (used by real_trading.py)
  - record_creator_token         (used by processor.py)
  - maybe_auto_blacklist_creator (used by lookback.py)
"""
import logging
from contextlib import closing

from .config import (
    REAL_STOP_LOSS_PCT, REAL_TAKE_PROFIT_PCT, REAL_TIME_STOP_SEC,
)
from .db import db_conn, db_write
from .state import blacklist_cache, is_creator_blacklisted
from .utils import now_ts, safe_float, safe_int

log = logging.getLogger(__name__)


# ---------- Dynamic exit parameter computation ----------

def compute_dynamic_exit_params(
    coin: dict,
    result: dict,
    market_ctx: "MarketContext | None" = None,
) -> tuple[float, float, int]:
    """Compute per-trade SL%, TP%, and time-stop seconds from coin features.

    Decisions are based on:
      - Score / probability (high confidence → tighter SL, wider TP)
      - Market-cap percentile (late entries get tighter parameters)
      - Coin age (fresh coins get more room)
      - Reply momentum (high engagement → longer time, wider TP)
    Returns (sl_pct, tp_pct, time_stop_sec).
    """
    score   = safe_int(result.get("score", 0))
    prob    = safe_float(result.get("probability", 0.0))
    mc      = safe_float(coin.get("usd_market_cap"))
    replies = safe_int(coin.get("reply_count", 0))

    base_sl   = REAL_STOP_LOSS_PCT
    base_tp   = REAL_TAKE_PROFIT_PCT
    base_time = REAL_TIME_STOP_SEC

    # --- Score/prob scaling (-1.0 … +1.0) ---
    score_factor = (score - 5) / 5.0
    prob_factor  = (prob - 0.5) * 2.0
    confidence   = max(-1.0, min(1.0, 0.6 * score_factor + 0.4 * prob_factor))

    # --- MC percentile: later entries get tighter exits ---
    mc_pct = 0.5
    if market_ctx is not None:
        try:
            mc_pct = market_ctx.percentile_mc(mc)
        except Exception:
            pass
    mc_factor = 1.0 - (mc_pct - 0.5) * 1.0  # 0.5→1.0, 1.0→0.5

    # --- Coin age: fresh coins are more volatile → wider exit bands ---
    age_factor = 1.0
    created_raw = coin.get("created_timestamp")
    if created_raw:
        created_ts = safe_int(created_raw)
        if created_ts > 1_000_000_000_000:
            created_ts //= 1000
        age_sec    = now_ts() - created_ts
        age_factor = max(0.6, min(1.4, 1.4 - (age_sec / 7200.0) * 0.6))

    # --- Reply momentum: high engagement → give more time/room ---
    mc_k = max(mc, 1.0)
    replies_per_kmc    = replies / (mc_k / 1000.0)
    engagement_factor  = max(0.6, min(1.4, 0.8 + (replies_per_kmc / 50.0) * 0.6))

    # --- SL: high confidence → tighter (smaller %) ---
    sl = base_sl * (1.0 - 0.3 * confidence) * mc_factor * age_factor
    sl = max(base_sl * 0.3, min(base_sl * 1.5, sl))

    # --- TP: high confidence → wider ---
    tp = base_tp * (1.0 + 0.8 * confidence) * mc_factor * engagement_factor
    tp = max(base_tp * 0.75, min(base_tp * 2.5, tp))  # floor: 60% at base 80

    # --- Time stop: more time for high engagement + high confidence ---
    time_mult = (1.0 + 0.5 * confidence) * mc_factor * engagement_factor
    time_sec  = int(base_time * max(0.3, min(2.5, time_mult)))

    return round(sl, 1), round(tp, 1), time_sec


# ---------- Creator history / auto-blacklist ----------

def record_creator_token(creator: str, mint: str) -> None:
    if not creator or not mint:
        return

    def _w():
        with closing(db_conn()) as conn, conn:
            conn.execute(
                "INSERT OR IGNORE INTO creator_history(creator,mint,seen_at) VALUES(?,?,?)",
                (creator, mint, now_ts()))
    db_write(_w)


def maybe_auto_blacklist_creator(mint: str) -> None:
    with closing(db_conn()) as conn:
        cr = conn.execute(
            "SELECT creator FROM creator_history WHERE mint=?", (mint,),
        ).fetchone()
    if not cr:
        return
    creator = cr["creator"]
    if not creator or is_creator_blacklisted(creator):
        return

    with closing(db_conn()) as conn:
        rows = conn.execute("""
            SELECT outcome FROM creator_history
            WHERE creator=? AND outcome IS NOT NULL
            ORDER BY seen_at DESC LIMIT 3
        """, (creator,)).fetchall()

    if len(rows) < 3:
        return

    bad_outcomes = {"RUG", "DOWN"}
    if all(r["outcome"] in bad_outcomes for r in rows):
        def _w():
            with closing(db_conn()) as conn, conn:
                conn.execute(
                    "INSERT OR IGNORE INTO creator_blacklist(creator,reason,added_at,auto_added) "
                    "VALUES(?,?,?,1)",
                    (creator, "3 consecutive rugs", now_ts()))
        db_write(_w)
        blacklist_cache.invalidate()
        log.warning("AUTO-BLACKLIST creator %s (3 consecutive rugs)", creator[:8])
