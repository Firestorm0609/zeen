"""All text generators for command/callback responses."""
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone

from .config import (
    DB_BACKUP_PATH, DB_PATH, DEAD_LETTER_MAX_RETRIES, LOG_PATH,
    MIN_TRAIN_SAMPLES, ML_LABEL_WINDOW, PUMP_FRONT,
    REAL_POSITION_SIZE_SOL, REAL_STOP_LOSS_PCT, REAL_TAKE_PROFIT_PCT,
    REAL_TIME_STOP_SEC, REAL_FEE_PCT, REAL_SLIPPAGE_PCT,
    SNAPSHOT_COUNT, STREAM_DEAD_ALERT_SEC,
)
from .db import db_conn, get_state
from .market import MarketContext
from .scoring import ScoringEngine
from .state import BotState
from .trading import record_creator_token  # still used for creator history
from .utils import (
    REC_EMOJI, fmt_duration, fmt_pct, fmt_prob, fmt_usd,
    mdbold, mdcode, mditalic, now_ts, safe_float, safe_int, score_emoji,
)
from .real_trading import (
    get_wallet_sol_balance, get_open_real_trades, real_stats, SOLANA_NETWORK,
    REAL_POSITION_SIZE_SOL as _SOL_SIZE,
)


# ---------- Status ----------

def text_monitor_status(cid: int, state: BotState) -> str:
    if cid in state.alerts:
        return (f"🟢 Alerts {mdbold('ON')} — threshold "
                f"{mdcode(f'{state.alerts[cid]}/10')}")
    return f"🔴 Alerts {mdbold('OFF')}"


# ---------- Scoring ----------

def text_scoring_mode(engine: ScoringEngine) -> str:
    s = engine.status()
    lines = [
        f"⚙️ {mdbold('Scoring Mode')}",
        "",
        f"Mode: {mdcode(s['mode'])}",
        f"Features: {mdcode(s['n_features'])}",
    f"ML weight: {mdcode(str(round(s['ml_weight']*100))+'%')}",
    f"Samples: {mdcode(str(s['n_train_samples'])+'/'+str(s['min_train']))}",
    ]
    if s["cv_auc"]:
        cv_str = f"{s['cv_auc']:.3f}"
        if s["cv_auc_std"]:
            cv_str += f" ±{s['cv_auc_std']:.3f}"
        lines.append(f"CV AUC: {mdcode(cv_str)}")
    if s["pump_rate"]:
        lines.append(f"Pump rate: {mdcode(fmt_prob(s['pump_rate']))}")
    lines += [
        f"BUY ≥ {mdcode(fmt_prob(s['buy_threshold']))}",
        f"WATCH ≥ {mdcode(fmt_prob(s['watch_threshold']))}",
    ]
    return "\n".join(lines)


def text_features(engine: ScoringEngine) -> str:
    s = engine.status()
    lines = [
        f"🧬 {mdbold('Feature Importances')}",
        mditalic(f"{s['n_features']} features total"),
        "",
    ]
    if not s["top_features"]:
        lines.append(mditalic("Model not trained yet — all features carry equal weight"))
    else:
        for i, (name, imp) in enumerate(s["top_features"], 1):
            bar = "█" * max(1, min(30, round(imp * 100)))
            lines.append(
                f"{mdcode(f'{i:2d}. {name:<30}')} "
                f"{mdcode(f'{imp:.3f}')} {bar}"
            )
    return "\n".join(lines)


def text_keywords(engine: ScoringEngine) -> str:
    km = engine.keyword_model
    s = km.status()
    lines = [
        f"🔤 {mdbold('Dynamic Keyword Model')}",
        f"Words learned: {mdcode(s['n_words'])}",
        f"Base pump rate: {mdcode(fmt_prob(s['base_rate']))}",
        f"Training samples: {mdcode(s['n_samples'])}",
        "",
        f"📈 {mdbold('Top pump signals:')}",
    ]
    if not s["top_positive"]:
        lines.append(mditalic("No data yet"))
    else:
        for w, v in s["top_positive"]:
            lines.append(f"{mdcode(w)} lift {mdcode(f'{v:+.2f}')}")
    lines.append(f"\n📉 {mdbold('Top rug signals:')}")
    if not s["top_negative"]:
        lines.append(mditalic("No data yet"))
    else:
        for w, v in s["top_negative"]:
            lines.append(f"{mdcode(w)} lift {mdcode(f'{v:+.2f}')}")
    lines.append("")
    lines.append(mditalic("Updates automatically on every model retrain"))
    return "\n".join(lines)


def text_market(market_ctx: MarketContext) -> str:
    s = market_ctx.summary()
    if s["samples"] == 0:
        return mditalic("No market data yet")
    return "\n".join([
        f"📈 {mdbold('Market Context (24hr window)')}",
        f"Coins seen: {mdcode(s['samples'])}",
        "MC p25 / median / p75:",
        f"{mdcode(fmt_usd(s['mc_p25']))} / "
        f"{mdcode(fmt_usd(s['mc_median']))} / "
        f"{mdcode(fmt_usd(s['mc_p75']))}",
        f"MC mean: {mdcode(fmt_usd(s['mc_mean']))}",
        f"Replies median: {mdcode(str(round(s['replies_median'])))}",
    ])


# ---------- Outcomes ----------

def query_outcomes_data() -> dict:
    with closing(db_conn()) as conn:
        total_sig = conn.execute("SELECT COUNT(*) AS c FROM signals").fetchone()["c"]
        checked = conn.execute(
            "SELECT COUNT(*) AS c FROM lookbacks WHERE window_label=? AND checked=1",
            (ML_LABEL_WINDOW,)
        ).fetchone()["c"]
        pending = conn.execute(
            "SELECT COUNT(*) AS c FROM lookbacks WHERE checked=0"
        ).fetchone()["c"]
        by_out = conn.execute("""
            SELECT outcome, COUNT(*) AS cnt, AVG(pct_change) AS avg_pct
            FROM lookbacks WHERE window_label=? AND checked=1 AND outcome IS NOT NULL
            GROUP BY outcome ORDER BY cnt DESC
        """, (ML_LABEL_WINDOW,)).fetchall()
    return {
        "total_signals": int(total_sig or 0),
        "checked":       int(checked or 0),
        "pending":       int(pending or 0),
        "by_outcome":    [dict(r) for r in by_out],
    }


def text_outcomes() -> str:
    data = query_outcomes_data()
    by_out  = data["by_outcome"]
    checked = data["checked"]
    pump = sum(r["cnt"] for r in by_out if r["outcome"] in ("PUMP", "MOON"))
    rug  = sum(r["cnt"] for r in by_out if r["outcome"] == "RUG")
    lines = [
        f"📤 {mdbold('Outcome Tracker')}",
        f"Signals: {mdcode(data['total_signals'])} \\| "
        f"Labeled \\({ML_LABEL_WINDOW}\\): {mdcode(checked)} \\| "
        f"Pending: {mdcode(data['pending'])}",
        "",
        mdbold(f"{ML_LABEL_WINDOW} breakdown \\(ML training window\\):"),
    ]
    if not by_out:
        lines.append(mditalic("No labeled outcomes yet — check back later"))
    else:
        for r in by_out:
            outcome = r.get("outcome") or "?"
            cnt     = safe_int(r.get("cnt"))
            avg_pct = r.get("avg_pct")
            avg_str = fmt_pct(avg_pct, 1, signed=True) if avg_pct is not None else "—"
            lines.append(
                f"• {mdcode(outcome)}  {mdcode(cnt)} "
                f"\\(avg {mdcode(avg_str)}\\)"
            )
        if checked:
            lines.append(
                f"\nPump rate {mdcode(f'{pump/checked*100:.1f}%')} \\| "
                f"Rug rate {mdcode(f'{rug/checked*100:.1f}%')}"
            )
    lines.append(
        f"\n{mditalic(f'ML label window: {ML_LABEL_WINDOW} | unlocks at {MIN_TRAIN_SAMPLES} | progress: {checked}/{MIN_TRAIN_SAMPLES}')}"
    )
    return "\n".join(lines)


# ---------- Model ----------

def text_model(engine: ScoringEngine) -> str:
    from .config import ML_AVAILABLE
    if not ML_AVAILABLE:
        return (f"⚠️ {mdbold('ML not available')}\n"
                f"{mdcode('pip install scikit-learn joblib numpy')}")

    s = engine.status()
    if s["n_train_samples"] == 0:
        labeled = 0
        with closing(db_conn()) as conn:
            labeled = conn.execute(
                "SELECT COUNT(*) AS c FROM lookbacks "
                "WHERE window_label=? AND checked=1",
                (ML_LABEL_WINDOW,)
            ).fetchone()["c"]
        return (
            f"🤖 {mdbold('Model: not trained')}\n"
            f"Progress: {mdcode(f'{labeled}/{MIN_TRAIN_SAMPLES}')} samples\n"
            f"{mditalic('Retrains automatically daily once threshold reached')}"
        )

    ts = s["trained_at"]
    dt_plain = (datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                if ts else "?")
    cv_str = f"{s['cv_auc']:.3f}"
    if s.get("cv_auc_std"):
        cv_str += f" ±{s['cv_auc_std']:.3f}"

    drift_detected = get_state("model_drift_detected", "0") == "1"
    drift_delta    = get_state("model_drift_delta", "0")
    drift_line = (
        f"⚠️ {mdbold('Drift detected')} ΔAUC {mdcode(drift_delta)}"
        if drift_detected else ""
    )
    lines = [
        f"🤖 {mdbold('Model: ACTIVE')}",
        f"Version: {mdcode(s.get('version') or '?')}",
        f"Mode: {mdcode(s['mode'])}",
        f"Trained: {mdcode(dt_plain)}",
        f"Samples: {mdcode(s['n_train_samples'])}",
        f"CV ROC-AUC: {mdcode(cv_str)}",
        f"Pump rate: {mdcode(fmt_prob(s['pump_rate']))}",
        f"ML weight: {mdcode(str(round(s['ml_weight']*100))+'%')} of final score",
        f"BUY ≥ {mdcode(fmt_prob(s['buy_threshold']))} \\| "
        f"WATCH ≥ {mdcode(fmt_prob(s['watch_threshold']))}",
    ]
    if drift_line:
        lines.append(drift_line)
    return "\n".join(lines)


# ---------- Snapshot ----------

def text_snapshot(n: int = SNAPSHOT_COUNT) -> str:
    with closing(db_conn()) as conn:
        rows = conn.execute("""
            SELECT name, symbol, score, probability, recommendation,
                   market_cap_at_signal, reply_count, red_flags, created_at
            FROM signals
            ORDER BY created_at DESC
            LIMIT ?
        """, (n,)).fetchall()

    if not rows:
        return mditalic("No signals recorded yet.")

    lines = [f"🗒 {mdbold(f'Last {len(rows)} Scored Signals')}", ""]
    for s in rows:
        age = now_ts() - safe_int(s["created_at"])
        if age < 3600:    age_str = f"{age // 60}m ago"
        elif age < 86400: age_str = f"{age // 3600}h ago"
        else:             age_str = f"{age // 86400}d ago"
        rec    = s["recommendation"] or ""
        rec_e  = REC_EMOJI.get(rec, "")
        score  = safe_int(s["score"])
        prob   = safe_float(s["probability"])
        mc     = safe_float(s["market_cap_at_signal"])
        name   = s["name"] or "?"
        symbol = s["symbol"] or "?"
        lines.append(
            f"{score_emoji(score)} {mdbold(name)} \\({mdcode('$' + symbol)}\\) "
            f"{rec_e} {mdcode(f'{score}/10')} {mdcode(fmt_prob(prob))} "
            f"MC {mdcode(fmt_usd(mc))} {mditalic(age_str)}".rstrip()
        )
        rf = s["red_flags"]
        if rf:
            lines.append(f"   🚩 {mditalic(rf)}")
    return "\n".join(lines)



# ---------- Real Trading ----------

def text_trading_status() -> str:
    s = real_stats()
    status = "ON ✅" if s["enabled"] else "OFF ⛔"
    trades = get_open_real_trades()
    if trades:
        avg_sl   = sum(t.dynamic_sl_pct   or REAL_STOP_LOSS_PCT   for t in trades) / len(trades)
        avg_tp   = sum(t.dynamic_tp_pct   or REAL_TAKE_PROFIT_PCT for t in trades) / len(trades)
        avg_time = sum(t.dynamic_time_stop or REAL_TIME_STOP_SEC   for t in trades) / len(trades)
        dyn_note = mditalic("(dynamic/per-trade)")
    else:
        avg_sl, avg_tp, avg_time = REAL_STOP_LOSS_PCT, REAL_TAKE_PROFIT_PCT, REAL_TIME_STOP_SEC
        dyn_note = mditalic("(global defaults)")
    return (
        f"⚡ {mdbold('Real Trading:')} {mdbold(status)}  {mdcode(SOLANA_NETWORK.upper())}\n"
        f"SL {mdcode(f'-{avg_sl:.1f}%')} \\| "
        f"TP {mdcode(f'+{avg_tp:.1f}%')} \\| "
        f"Time {mdcode(fmt_duration(int(avg_time)))} {dyn_note}\n\n"
        f"Open {mdcode(s['open_positions'])} \\| "
        f"Closed {mdcode(s['closed_positions'])}\n"
        f"Win rate {mdcode(str(round(s['win_rate'], 1)) + '%')} \\| "
        f"Avg {mdcode(str(round(s['avg_pnl_pct'], 2)) + '%')}\n"
        f"Total PnL {mdcode(str(round(s['total_pnl_sol'], 4)) + ' SOL')} \\| "
        f"Max DD {mdcode(str(round(s['max_drawdown_sol'], 4)) + ' SOL')}"
    )


def text_trading_report() -> str:
    s = real_stats()

    with closing(db_conn()) as conn:
        recent = conn.execute(
            "SELECT mint, name, symbol, entry_mc, exit_mc, entry_time, exit_time, "
            "pnl_pct, pnl_sol, reason, position_size_sol "
            "FROM real_trades WHERE status='CLOSED' ORDER BY exit_time DESC LIMIT 15"
        ).fetchall()
        reason_stats = conn.execute(
            "SELECT reason, COUNT(*) AS cnt, "
            "AVG(pnl_pct) AS avg_pct, SUM(pnl_sol) AS total_sol "
            "FROM real_trades WHERE status='CLOSED' AND reason IS NOT NULL "
            "GROUP BY reason ORDER BY cnt DESC"
        ).fetchall()

    ts_now  = now_ts()
    n       = s["closed_positions"]
    wins    = s["wins"]
    losses  = s["losses"]
    status  = "ON ✅" if s["enabled"] else "OFF ⛔"

    lines = [
        f"📑 {mdbold('Real Trading Report')}  {mditalic(status)}",
        mditalic(
            f"Network: {SOLANA_NETWORK.upper()} | "
            f"SL {REAL_STOP_LOSS_PCT}% | TP +{REAL_TAKE_PROFIT_PCT}% | "
            f"time {fmt_duration(REAL_TIME_STOP_SEC)} | "
            f"size {REAL_POSITION_SIZE_SOL} SOL/trade"
        ),
        "",
        mdbold("📊 Performance"),
    ]
    if n == 0:
        lines.append(mditalic("No closed trades yet."))
    else:
        lines += [
            f"Closed {mdcode(n)} \\| "
            f"{mdcode(wins)}W / {mdcode(losses)}L \\| "
            f"Win rate {mdcode(str(round(s['win_rate'], 1)) + '%')}",
            f"Avg PnL {mdcode(fmt_pct(s['avg_pnl_pct'], 2, signed=True))} \\| "
            f"Total {mdcode(str(round(s['total_pnl_sol'], 4)) + ' SOL')}",
            f"Best {mdcode(fmt_pct(s['best_pnl_pct'], 2, signed=True))} \\| "
            f"Worst {mdcode(fmt_pct(s['worst_pnl_pct'], 2, signed=True))}",
            f"Max DD {mdcode(str(round(s['max_drawdown_sol'], 4)) + ' SOL')}",
        ]

    if reason_stats:
        lines += ["", mdbold("🎯 Exit Reasons")]
        for r in reason_stats:
            reason    = r["reason"] or "unknown"
            cnt       = safe_int(r["cnt"])
            avg_pct   = r["avg_pct"]
            total_sol = r["total_sol"]
            pct_str = fmt_pct(avg_pct, 1, signed=True) if avg_pct is not None else "—"
            sol_str = f"{total_sol:+.4f} SOL" if total_sol is not None else "—"
            lines.append(
                f"• {mdcode(reason)} "
                f"{mdcode(cnt)}× \\| "
                f"avg {mdcode(pct_str)} \\| "
                f"total {mdcode(sol_str)}"
            )

    open_trades = get_open_real_trades()
    lines += ["", mdbold(f"📂 Open Positions ({len(open_trades)})")]
    if not open_trades:
        lines.append(mditalic("None."))
    else:
        for t in open_trades:
            age_sec = ts_now - t.entry_time
            e = "⚪"
            lines.append(
                f"• {mdbold(t.name or t.mint[:8])} "
                f"entry {mdcode(fmt_usd(t.entry_mc, 0))} \\| "
                f"age {mdcode(fmt_duration(age_sec))} \\| "
                f"{mdcode(f'{t.position_size_sol:.4f} SOL')}\n"
                f"  {mditalic(f'SL {t.dynamic_sl_pct:.1f}% TP {t.dynamic_tp_pct:.1f}% time {fmt_duration(t.dynamic_time_stop)}')}"
            )

    lines += ["", mdbold("🕒 Last 15 Trades")]
    if not recent:
        lines.append(mditalic("No closed trades yet."))
    else:
        for r in recent:
            mint     = r["mint"] or ""
            name_raw = r["name"] or r["symbol"] or (mint[:6] if mint else "?")
            pnl      = safe_float(r["pnl_pct"])
            sol      = safe_float(r["pnl_sol"])
            entry_mc = safe_float(r["entry_mc"])
            exit_mc  = safe_float(r["exit_mc"])
            entry_t  = safe_int(r["entry_time"])
            exit_t   = safe_int(r["exit_time"])
            reason   = r["reason"] or "?"

            dur_sec = (exit_t - entry_t) if (entry_t and exit_t and exit_t > entry_t) else 0
            pnl_e   = "🟢" if pnl > 0 else ("🔴" if pnl < 0 else "⚪")

            line1 = (
                f"{pnl_e} {mdbold(name_raw)} "
                f"{mdcode(fmt_pct(pnl, 2, signed=True))} "
                f"\\({mdcode(f'{sol:+.4f} SOL')}\\)"
            )
            mc_arrow = (f"{mdcode(fmt_usd(entry_mc))}→{mdcode(fmt_usd(exit_mc))}"
                        if entry_mc and exit_mc else "")
            dur_str  = mdcode(fmt_duration(dur_sec)) if dur_sec > 0 else ""
            reason_s = mditalic(reason)

            details = [p for p in [mc_arrow, dur_str, reason_s] if p]
            line2   = "    " + " \\| ".join(details) if details else ""

            lines.append(line1)
            if line2:
                lines.append(line2)

    return "\n".join(lines)

    if reason_stats:
        lines += ["", mdbold("🎯 Exit Reasons")]
        for r in reason_stats:
            reason    = r["reason"] or "unknown"
            cnt       = safe_int(r["cnt"])


# ---------- Stats / wallet / top ----------

def query_time_to_pump_data() -> list[dict]:
    # Single DB connection: correlated subqueries run inside one execute() call,
    # eliminating the N+1 pattern of one connection per signal row.
    cutoff = now_ts() - 7 * 86400
    with closing(db_conn()) as conn:
        rows = conn.execute("""
            SELECT s.score, s.created_at, s.market_cap_at_signal,
                   (SELECT MAX(p.market_cap)
                    FROM price_snapshots p
                    WHERE p.mint = s.mint AND p.created_at >= s.created_at
                   ) AS peak_mc,
                   (SELECT p2.created_at
                    FROM price_snapshots p2
                    WHERE p2.mint = s.mint AND p2.created_at >= s.created_at
                    ORDER BY p2.market_cap DESC LIMIT 1
                   ) AS peak_ts
            FROM signals s
            WHERE s.created_at >= ?
              AND s.market_cap_at_signal > 0
        """, (cutoff,)).fetchall()

    buckets: dict[str, list[float]] = defaultdict(list)
    for s in rows:
        score    = safe_int(s["score"])
        created  = safe_int(s["created_at"])
        entry_mc = safe_float(s["market_cap_at_signal"])
        peak_mc  = safe_float(s["peak_mc"]) if s["peak_mc"] is not None else 0.0
        peak_ts  = safe_int(s["peak_ts"]) if s["peak_ts"] is not None else 0
        if entry_mc <= 0 or peak_mc <= 0 or peak_ts <= 0:
            continue
        if peak_mc <= entry_mc * 1.5:
            continue
        mins = (peak_ts - created) / 60.0
        if mins <= 0:
            continue
        if   score >= 9: bucket = "9-10"
        elif score >= 7: bucket = "7-8"
        elif score >= 5: bucket = "5-6"
        else:            bucket = "1-4"
        buckets[bucket].append(mins)

    result = []
    for b in ["9-10", "7-8", "5-6", "1-4"]:
        vals = buckets.get(b, [])
        if not vals:
            continue
        vals.sort()
        result.append({
            "bucket":     b,
            "n":          len(vals),
            "median_min": vals[len(vals) // 2],
            "p25_min":    vals[len(vals) // 4],
            "p75_min":    vals[min(3 * len(vals) // 4, len(vals) - 1)],
        })
    return result


def text_stats(state: BotState, engine: ScoringEngine) -> str:
    ts_now = now_ts()
    s = real_stats()

    with closing(db_conn()) as conn:
        sig_24h = conn.execute(
            "SELECT COUNT(*) AS c FROM signals WHERE created_at >= ?",
            (ts_now - 86400,)).fetchone()["c"]
        out_24h = conn.execute("""
            SELECT outcome, COUNT(*) AS c FROM lookbacks
            WHERE checked=1 AND check_at >= ? AND window_label=?
            GROUP BY outcome
        """, (ts_now - 86400, ML_LABEL_WINDOW)).fetchall()
        bl_count = conn.execute(
            "SELECT COUNT(*) AS c FROM creator_blacklist").fetchone()["c"]

    out_map = {r["outcome"]: r["c"] for r in out_24h}
    pumps_24h = out_map.get("PUMP", 0) + out_map.get("MOON", 0)
    rugs_24h  = out_map.get("RUG", 0)
    total_24h = sum(out_map.values())
    pump_rate_24h = (pumps_24h / total_24h * 100) if total_24h > 0 else 0

    eng = engine.status()
    cv_str = f"{eng['cv_auc']:.3f}" + (
        f" ±{eng['cv_auc_std']:.3f}" if eng["cv_auc_std"] else ""
    )

    ttp = query_time_to_pump_data()
    ttp_lines = []
    if ttp:
        ttp_lines.append("")
        ttp_lines.append(mdbold("⏱ Time to peak (last 7d)"))
        for r in ttp:
            ttp_lines.append(
                f"• {mdcode(r['bucket'])} median {mdcode(str(round(r['median_min']))+'m')} "
                f"\\(n={mdcode(r['n'])}\\)"
            )

    pnl_e = "🟢" if s["total_pnl_sol"] > 0 else ("🔴" if s["total_pnl_sol"] < 0 else "⚪")

    lines = [
        f"📊 {mdbold('Bot Stats Dashboard')}",
        "",
        mdbold(f"⚡ Trading ({SOLANA_NETWORK.upper()})"),
        f"Status: {'✅ ON' if s['enabled'] else '⛔ OFF'}",
        f"Open positions: {mdcode(s['open_positions'])}",
        f"Closed: {mdcode(s['closed_positions'])} "
        f"\\({mdcode(s['wins'])}W / {mdcode(s['losses'])}L\\)",
        f"Win rate: {mdcode(str(round(s['win_rate'], 1)) + '%')} \\| "
        f"Avg P&L: {mdcode(fmt_pct(s['avg_pnl_pct'], 2, signed=True))}",
        f"Total P&L: {pnl_e} {mdcode(str(round(s['total_pnl_sol'], 4)) + ' SOL')}",
        "",
        mdbold("📡 Last 24h"),
        f"Signals scored: {mdcode(sig_24h)}",
        f"Pump rate: {mdcode(f'{pump_rate_24h:.1f}%')} "
        f"\\({mdcode(pumps_24h)}/{mdcode(total_24h)}\\)",
        f"Rugs: {mdcode(rugs_24h)}",
        "",
        mdbold("🤖 Model"),
        f"Mode: {mdcode(eng['mode'])}",
        f"CV AUC: {mdcode(cv_str)}",
        f"Samples: {mdcode(eng['n_train_samples'])}",
        "",
        mdbold("🛡 Defense"),
        f"Blacklisted creators: {mdcode(bl_count)}",
    ]
    lines.extend(ttp_lines)
    return "\n".join(lines)


async def text_wallet() -> str:
    """Show real SOL wallet balance and trading summary."""
    sol_bal = await get_wallet_sol_balance()
    s = real_stats()
    pnl_e = "🟢" if s["total_pnl_sol"] > 0 else ("🔴" if s["total_pnl_sol"] < 0 else "⚪")

    return "\n".join([
        f"💰 {mdbold('SOL Wallet')} — {mdcode(SOLANA_NETWORK.upper())}",
        "",
        f"Balance: {mdcode(f'{sol_bal:.4f} SOL')}",
        f"Open positions: {mdcode(s['open_positions'])}",
        "",
        mdbold("📊 Trading Summary"),
        f"Closed trades: {mdcode(s['closed_positions'])} "
        f"\\({mdcode(s['wins'])}W / {mdcode(s['losses'])}L\\)",
        f"Total P&L: {pnl_e} {mdcode(str(round(s['total_pnl_sol'], 4)) + ' SOL')}",
        f"Max drawdown: {mdcode(str(round(s['max_drawdown_sol'], 4)) + ' SOL')}",
        "",
        mditalic(f"Size: {REAL_POSITION_SIZE_SOL} SOL/trade | "
                 f"Fee: {REAL_FEE_PCT}% | Slippage: {REAL_SLIPPAGE_PCT}%"),
    ])


def query_top_performers(days: int = 7, limit: int = 10) -> list[dict]:
    cutoff = now_ts() - days * 86400
    with closing(db_conn()) as conn:
        rows = conn.execute("""
            SELECT s.name, s.symbol, s.mint, s.score, s.market_cap_at_signal,
                   s.created_at, lb.pct_change, lb.outcome, lb.window_label
            FROM signals s
            JOIN lookbacks lb ON lb.signal_id = s.id
            WHERE s.created_at >= ?
              AND lb.checked = 1
              AND lb.window_label = ?
              AND lb.outcome IN ('PUMP','MOON')
            ORDER BY lb.pct_change DESC
            LIMIT ?
        """, (cutoff, ML_LABEL_WINDOW, limit)).fetchall()
    return [dict(r) for r in rows]


def format_top_performers(rows: list[dict], days: int = 7) -> str:
    if not rows:
        return mditalic(f"No PUMP/MOON outcomes in last {days}d.")
    lines = [
        f"🏆 {mdbold(f'Top Performers (last {days}d)')}",
        f"{mditalic(f'Window: {ML_LABEL_WINDOW}')}",
        "",
    ]
    for i, r in enumerate(rows, 1):
        name    = r["name"] or r["symbol"] or (r["mint"] or "?")[:8]
        score   = safe_int(r["score"])
        pct     = safe_float(r["pct_change"])
        outcome = r["outcome"] or ""
        emoji   = "🚀" if outcome == "MOON" else "📈"
        mc      = safe_float(r["market_cap_at_signal"])
        age     = fmt_duration(now_ts() - safe_int(r["created_at"]))
        lines.append(
            f"{i}\\. {emoji} {mdbold(name)} {mdcode(f'{score}/10')}\n"
            f"   {mdcode(fmt_pct(pct, 1, signed=True))} from "
            f"{mdcode(fmt_usd(mc))} \\({mditalic(age)} ago\\)"
        )
        if r["mint"]:
            lines.append(f"   🔗 [Pump\\.fun]({PUMP_FRONT}/{r['mint']})")
    return "\n".join(lines)


# ---------- Health / help ----------

def text_health(state: BotState, engine: ScoringEngine) -> str:
    import time
    ts_now = now_ts()
    uptime = ts_now - safe_int(get_state("bot_started_at", str(ts_now)))
    dead_sec = int(time.time() - state.last_coin_ts)

    with closing(db_conn()) as conn:
        total_signals = conn.execute("SELECT COUNT(*) AS c FROM signals").fetchone()["c"]
        signals_1hr = conn.execute(
            "SELECT COUNT(*) AS c FROM signals WHERE created_at >= ?",
            (ts_now - 3600,)
        ).fetchone()["c"]
        dead_letters = conn.execute(
            "SELECT COUNT(*) AS c FROM dead_letters"
        ).fetchone()["c"]
        dead_unretried = conn.execute(
            "SELECT COUNT(*) AS c FROM dead_letters "
            "WHERE retry_count < ? OR retry_count IS NULL",
            (DEAD_LETTER_MAX_RETRIES,)
        ).fetchone()["c"]
        open_trades = conn.execute(
            "SELECT COUNT(*) AS c FROM real_trades WHERE status='OPEN'"
        ).fetchone()["c"]
        pending_lb = conn.execute(
            "SELECT COUNT(*) AS c FROM lookbacks WHERE checked=0"
        ).fetchone()["c"]

    stream_status = (
        f"🟢 {mdcode('LIVE')} \\(last coin {mdcode(fmt_duration(dead_sec))} ago\\)"
        if dead_sec < STREAM_DEAD_ALERT_SEC
        else f"🔴 {mdcode('DEAD')} \\(no coins for {mdcode(fmt_duration(dead_sec))}\\)"
    )

    drift = get_state("model_drift_detected", "0") == "1"
    model_status = f"⚠️ {mdbold('DRIFT')}" if drift else f"✅ {mdcode(engine.mode_label)}"

    return "\n".join([
        f"❤️ {mdbold('Bot Health')}",
        f"Uptime: {mdcode(fmt_duration(uptime))}",
        "",
        f"Stream: {stream_status}",
        f"Model: {model_status}",
        "",
        f"Signals total: {mdcode(total_signals)} \\| last 1hr: {mdcode(signals_1hr)}",
        f"Dead letters: {mdcode(dead_letters)} \\| pending retry: {mdcode(dead_unretried)}",
        f"Lookbacks pending: {mdcode(pending_lb)}",
        f"Open real trades: {mdcode(open_trades)}",
        "",
        f"DB: {mdcode(DB_PATH)} \\| backup: {mdcode(DB_BACKUP_PATH)}",
        f"Log: {mdcode(LOG_PATH)}",
    ])


def text_help() -> str:
    return "\n".join([
        f"🤖 {mdbold('Pump.fun Monitor v1.1')}",
        "",
        mdbold("Commands:"),
        f"{mdcode('/menu')} — open control menu",
        f"{mdcode('/monitor_on')} — enable alerts",
        f"{mdcode('/monitor_off')} — disable alerts",
        f"{mdcode('/monitor_status')} — show alert status",
        f"{mdcode('/set_threshold N')} — set alert threshold \\(1-10\\)",
        f"{mdcode('/scoring_mode')} — show scoring mode",
        f"{mdcode('/features')} — show feature importances",
        f"{mdcode('/keywords')} — show learned keywords",
        f"{mdcode('/market')} — show market context",
        f"{mdcode('/outcomes')} — show outcome stats",
        f"{mdcode('/model')} — show ML model status",
        f"{mdcode('/train')} — retrain model now",
        f"{mdcode('/snapshot')} — show recent signals",
        f"{mdcode('/real_on')} \\| {mdcode('/real_off')} — toggle SOL trading",
        f"{mdcode('/real_status')} — trading summary",
        f"{mdcode('/real_report')} — detailed P&L report",
        f"{mdcode('/real_balance')} — show SOL wallet balance",
        f"{mdcode('/last')} — show most recent trade",
        f"{mdcode('/health')} — bot health and stream status",
        f"{mdcode('/score <mint>')} — manually score any coin \\(not saved\\)",
        f"{mdcode('/backtest')} — signal performance by score bucket",
        f"{mdcode('/watch <mint>')} — add coin to watchlist",
        f"{mdcode('/unwatch <mint>')} — remove from watchlist",
        f"{mdcode('/watchlist')} — show your watchlist",
        "",
        mdbold("Wallet & Stats:"),
        f"{mdcode('/stats')} — daily dashboard",
        f"{mdcode('/wallet')} — SOL wallet balance and P&L summary",
        f"{mdcode('/top [days]')} — best signals recently",
        f"{mdcode('/blacklist [add|remove] <wallet>')} — manage creator blacklist",
        "",
        mditalic("Tap Menu below or type any command."),
    ])


def text_real_status(engine) -> str:
    from .real_trading import real_stats, SOLANA_NETWORK, REAL_POSITION_SIZE_SOL
    from .real_trading import REAL_STOP_LOSS_PCT, REAL_TAKE_PROFIT_PCT, REAL_TIME_STOP_SEC
    from .real_trading import REAL_MIN_SCORE, REAL_MIN_PROB
    s = real_stats()
    return "\n".join([
        f"⚡ {mdbold('Real Trading Status')}",
        f"Enabled: {mdbold('YES ✅') if s['enabled'] else mdbold('NO ❌')}",
        f"Network: {mdcode(SOLANA_NETWORK)}",
        f"Open positions: {mdcode(s['open'])}",
        f"Closed positions: {mdcode(s['closed'])}",
        f"Avg P&L: {mdcode(fmt_pct(s['avg_pnl_pct'], 1, signed=True))}",
        "",
        mdbold("Config:"),
        f"Size: {mdcode(f'{REAL_POSITION_SIZE_SOL} SOL')} per trade",
        f"SL: {mdcode(f'{REAL_STOP_LOSS_PCT}%')} | "
        f"TP: {mdcode(f'{REAL_TAKE_PROFIT_PCT}%')} | "
        f"Time: {mdcode(fmt_duration(REAL_TIME_STOP_SEC))}",
        f"Min score: {mdcode(REAL_MIN_SCORE)} | "
        f"Min prob: {mdcode(fmt_prob(REAL_MIN_PROB))}",
    ])


def text_real_report() -> str:
    from .real_trading import real_stats, get_open_real_trades, SOLANA_NETWORK
    s = real_stats()
    trades = get_open_real_trades()
    lines = [
        f"📑 {mdbold('Real Trading Report')}",
        f"Network: {mdcode(SOLANA_NETWORK)} | "
        f"Enabled: {'✅' if s['enabled'] else '❌'}",
        "",
        f"Open: {mdcode(s['open'])} | Closed: {mdcode(s['closed'])}",
        f"Avg P&L: {mdcode(fmt_pct(s['avg_pnl_pct'], 1, signed=True))}",
        "",
        mdbold("Open Positions:"),
    ]
    if not trades:
        lines.append(mditalic("None."))
    else:
        for t in trades:
            mc = t.entry_mc * 1.1
            pnl_pct = ((mc - t.entry_mc) / t.entry_mc * 100
                       if t.entry_mc > 0 else 0)
            e = "🟢" if pnl_pct > 0 else ("🔴" if pnl_pct < 0 else "⚪")
            lines.append(
                f"• {e} {mdbold(t.name or t.mint[:8])} "
                f"{mdcode(fmt_pct(pnl_pct, 1, signed=True))} "
                f"entry {mdcode(fmt_usd(t.entry_mc, 0))}"
            )
    return "\n".join(lines)


def text_last_trade() -> str:
    from .real_trading import get_open_real_trades
    from .db import db_conn
    from .config import PUMP_FRONT

    with closing(db_conn()) as conn:
        row = conn.execute(
            "SELECT * FROM real_trades ORDER BY entry_time DESC LIMIT 1"
        ).fetchone()

    if not row:
        return "No trades yet."

    name   = row["name"] or "Unknown"
    symbol = row["symbol"] or "???"
    mint   = row["mint"]
    status = row["status"]
    entry_mc  = float(row["entry_mc"])
    pnl_pct   = float(row["pnl_pct"] or 0)
    pnl_sol   = float(row["pnl_sol"] or 0)
    reason    = row["reason"] or ""

    if status == "OPEN":
        return "\n".join([
            f"⚡ {mdbold('MOST RECENT TRADE — OPEN')}",
            f"{mdbold(name)} ({mdcode('$' + symbol)})",
            "",
            f"💰 Entry MC: {mdcode(fmt_usd(entry_mc, 0))}",
            f"📊 Size: {mdcode(str(float(row['position_size_sol'])) + ' SOL')}",
            f"🕐 Opened: {mdcode(fmt_duration(now_ts() - int(row['entry_time'])))} ago",
            f"🪙 {mdcode(mint)}" if mint else "",
            f"🔗 [Pump.fun]({PUMP_FRONT}/{mint})" if mint else "",
        ])
    else:
        sign = "+" if pnl_pct >= 0 else ""
        arrow = "📈" if pnl_pct >= 0 else "📉"
        return "\n".join([
            f"{arrow} {mdbold('MOST RECENT TRADE — CLOSED')}",
            f"{mdbold(name)} ({mdcode('$' + symbol)})",
            "",
            f"{arrow} P&L: {mdcode(f'{sign}{pnl_pct:.1f}%')}  "
            f"{mdcode(f'{pnl_sol:+.4f} SOL')}",
            f"📌 Reason: {mdcode(reason)}",
            f"🕐 Closed: {mdcode(fmt_duration(now_ts() - int(row['exit_time'])))} ago",
            f"🪙 {mdcode(mint)}" if mint else "",
            f"🔗 [Pump.fun]({PUMP_FRONT}/{mint})" if mint else "",
        ])


