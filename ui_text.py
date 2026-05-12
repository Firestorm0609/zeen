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
from .trading import record_creator_token
from .utils import (
    REC_EMOJI, esc, fmt_duration, fmt_pct, fmt_prob, fmt_usd,
    mdbold, mdcode, mditalic, now_ts, safe_float, safe_int, score_emoji,
)
from .real_trading import (
    get_wallet_sol_balance, get_open_real_trades, real_stats, SOLANA_NETWORK,
    REAL_POSITION_SIZE_SOL as _SOL_SIZE,
)


# ── Design tokens ──────────────────────────────────────────────────────────

_DIV  = "━━━━━━━━━━━━━━━━━━━━━━━━"
_DIV_SM = "┄┄┄┄┄┄┄┄┄┄┄┄"
_BULL = "▸"
_DOT  = "●"
_DOT_OFF = "○"


def _bar(ratio: float, width: int = 10) -> str:
    """Render a Unicode fill bar  ████░░░░  clamped 0-1."""
    filled = max(0, min(width, round(ratio * width)))
    return "█" * filled + "░" * (width - filled)


def _pnl_emoji(val: float) -> str:
    if val > 5:   return "🚀"
    if val > 0:   return "🟢"
    if val == 0:  return "⚪"
    if val > -5:  return "🔴"
    return "💀"


def _net_badge(enabled: bool) -> str:
    return f"`{SOLANA_NETWORK.upper()}`"


def _status_dot(on: bool) -> str:
    return f"{_DOT} `ON `" if on else f"{_DOT_OFF} `OFF`"


# ── Monitor status ─────────────────────────────────────────────────────────

def text_monitor_status(cid: int, state: BotState) -> str:
    if cid in state.alerts:
        thr = state.alerts[cid]
        bar = _bar(thr / 10)
        return (
            f"{_DOT} Alerts {mdbold('ACTIVE')}\n"
            f"Threshold `{thr}/10`  `{bar}`"
        )
    return f"{_DOT_OFF} Alerts {mdbold('PAUSED')} — tap *Monitor ON* to enable"


# ── Scoring mode ───────────────────────────────────────────────────────────

def text_scoring_mode(engine: ScoringEngine) -> str:
    s = engine.status()
    prog = min(1.0, s["n_train_samples"] / max(s["min_train"], 1))
    lines = [
        f"⚙️  {mdbold('Scoring Engine')}",
        f"`{_DIV}`",
        f"  {_BULL} Mode          {mdcode(s['mode'])}",
        f"  {_BULL} Features      {mdcode(s['n_features'])}",
        f"  {_BULL} ML weight     {mdcode(str(round(s['ml_weight']*100)) + '%')}",
        f"  {_BULL} Samples       {mdcode(str(s['n_train_samples']) + '/' + str(s['min_train']))}",
        f"  `{_bar(prog)}` {mdcode(f'{prog*100:.0f}%')}",
    ]
    if s["cv_auc"]:
        cv_str = f"{s['cv_auc']:.3f}"
        if s["cv_auc_std"]:
            cv_str += f" ±{s['cv_auc_std']:.3f}"
        lines.append(f"  {_BULL} CV AUC        {mdcode(cv_str)}")
    if s["pump_rate"]:
        lines.append(f"  {_BULL} Pump rate     {mdcode(fmt_prob(s['pump_rate']))}")
    lines += [
        f"`{_DIV_SM}`",
        f"  BUY ≥ {mdcode(fmt_prob(s['buy_threshold']))}   "
        f"WATCH ≥ {mdcode(fmt_prob(s['watch_threshold']))}",
    ]
    return "\n".join(lines)


def text_features(engine: ScoringEngine) -> str:
    s = engine.status()
    lines = [
        f"🧬  {mdbold('Feature Importances')}",
        f"`{_DIV}`",
        mditalic(f"{s['n_features']} features total"),
        "",
    ]
    if not s["top_features"]:
        lines.append(mditalic("Model not trained yet — features carry equal weight"))
    else:
        for i, (name, imp) in enumerate(s["top_features"], 1):
            bar = _bar(imp * 5, 8)
            lines.append(
                f"  `{i:2d}.` {mdcode(f'{name:<28}')} "
                f"`{bar}` {mdcode(f'{imp:.3f}')}"
            )
    return "\n".join(lines)


def text_keywords(engine: ScoringEngine) -> str:
    km = engine.keyword_model
    s = km.status()
    lines = [
        f"🔤  {mdbold('Keyword Intelligence')}",
        f"`{_DIV}`",
        f"  {_BULL} Words learned    {mdcode(s['n_words'])}",
        f"  {_BULL} Base pump rate   {mdcode(fmt_prob(s['base_rate']))}",
        f"  {_BULL} Training samples {mdcode(s['n_samples'])}",
        "",
        f"📈  {mdbold('Top pump signals')}",
    ]
    if not s["top_positive"]:
        lines.append(mditalic("  No data yet"))
    else:
        for w, v in s["top_positive"]:
            bar = _bar(min(1.0, abs(v) / 2), 6)
            lines.append(f"  `{bar}` {mdcode(w)}  `{v:+.2f}`")

    lines += ["", f"📉  {mdbold('Top rug signals')}"]
    if not s["top_negative"]:
        lines.append(mditalic("  No data yet"))
    else:
        for w, v in s["top_negative"]:
            bar = _bar(min(1.0, abs(v) / 2), 6)
            lines.append(f"  `{bar}` {mdcode(w)}  `{v:+.2f}`")

    lines += ["", mditalic("Updates automatically on every model retrain")]
    return "\n".join(lines)


def text_market(market_ctx: MarketContext) -> str:
    s = market_ctx.summary()
    if s["samples"] == 0:
        return mditalic("No market data yet — waiting for coins\\.\\.\\.")
    return "\n".join([
        f"📡  {mdbold('Market Context')}  {mditalic('24hr window')}",
        f"`{_DIV}`",
        f"  {_BULL} Coins seen       {mdcode(s['samples'])}",
        f"  {_BULL} MC p25           {mdcode(fmt_usd(s['mc_p25']))}",
        f"  {_BULL} MC median        {mdcode(fmt_usd(s['mc_median']))}",
        f"  {_BULL} MC p75           {mdcode(fmt_usd(s['mc_p75']))}",
        f"  {_BULL} MC mean          {mdcode(fmt_usd(s['mc_mean']))}",
        f"  {_BULL} Replies median   {mdcode(str(round(s['replies_median'])))}",
    ])


# ── Outcomes ───────────────────────────────────────────────────────────────

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
    data   = query_outcomes_data()
    by_out = data["by_outcome"]
    checked = data["checked"]
    pump = sum(r["cnt"] for r in by_out if r["outcome"] in ("PUMP", "MOON"))
    rug  = sum(r["cnt"] for r in by_out if r["outcome"] == "RUG")
    pump_rate = pump / checked * 100 if checked else 0
    rug_rate  = rug  / checked * 100 if checked else 0

    _OUT_EMOJI = {"PUMP": "📈", "MOON": "🚀", "RUG": "💀", "FLAT": "➖"}

    lines = [
        f"📤  {mdbold('Outcome Tracker')}",
        f"`{_DIV}`",
        f"  {_BULL} Signals scored   {mdcode(data['total_signals'])}",
        f"  {_BULL} Labeled          {mdcode(checked)}  `{_bar(checked / max(MIN_TRAIN_SAMPLES,1))}`",
        f"  {_BULL} Pending          {mdcode(data['pending'])}",
        "",
        f"◈  {mdbold(ML_LABEL_WINDOW + ' breakdown')}",
    ]
    if not by_out:
        lines.append(mditalic("  No labeled outcomes yet"))
    else:
        total = sum(r["cnt"] for r in by_out)
        for r in by_out:
            outcome  = r.get("outcome") or "?"
            cnt      = safe_int(r.get("cnt"))
            avg_pct  = r.get("avg_pct")
            avg_str  = fmt_pct(avg_pct, 1, signed=True) if avg_pct is not None else "—"
            ratio    = cnt / total if total else 0
            emoji    = _OUT_EMOJI.get(outcome, "◆")
            lines.append(
                f"  {emoji} {mdcode(f'{outcome:<6}')} "
                f"`{_bar(ratio, 8)}` {mdcode(cnt)}x  avg {mdcode(avg_str)}"
            )
        lines += [
            "",
            f"  Pump `{pump_rate:.1f}%`   Rug `{rug_rate:.1f}%`",
        ]

    prog_pct = min(100, round(checked / MIN_TRAIN_SAMPLES * 100)) if MIN_TRAIN_SAMPLES else 100
    lines += [
        "",
        mditalic(f"ML window: {ML_LABEL_WINDOW}  ·  unlocks at {MIN_TRAIN_SAMPLES} samples  ·  {prog_pct}% there"),
    ]
    return "\n".join(lines)


# ── Model ──────────────────────────────────────────────────────────────────

def text_model(engine: ScoringEngine) -> str:
    from .config import ML_AVAILABLE
    if not ML_AVAILABLE:
        return (
            f"⚠️  {mdbold('ML Unavailable')}\n"
            f"`{_DIV_SM}`\n"
            f"Run: {mdcode('pip install scikit-learn joblib numpy')}"
        )

    s = engine.status()
    if s["n_train_samples"] == 0:
        labeled = 0
        with closing(db_conn()) as conn:
            labeled = conn.execute(
                "SELECT COUNT(*) AS c FROM lookbacks WHERE window_label=? AND checked=1",
                (ML_LABEL_WINDOW,)
            ).fetchone()["c"]
        prog = min(1.0, labeled / max(MIN_TRAIN_SAMPLES, 1))
        return (
            f"🤖  {mdbold('Model — Collecting Data')}\n"
            f"`{_DIV}`\n"
            f"  Progress  `{_bar(prog)}` {mdcode(f'{labeled}/{MIN_TRAIN_SAMPLES}')}\n\n"
            f"{mditalic('Retrains automatically once threshold is reached')}"
        )

    ts = s["trained_at"]
    dt_plain = (
        datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%d %b %Y %H:%M UTC")
        if ts else "?"
    )
    cv_str = f"{s['cv_auc']:.3f}"
    if s.get("cv_auc_std"):
        cv_str += f" ±{s['cv_auc_std']:.3f}"

    drift_detected = get_state("model_drift_detected", "0") == "1"
    drift_delta    = get_state("model_drift_delta", "0")

    lines = [
        f"🤖  {mdbold('ML Model — ACTIVE')}",
        f"`{_DIV}`",
        f"  {_BULL} Version    {mdcode(s.get('version') or '?')}",
        f"  {_BULL} Mode       {mdcode(s['mode'])}",
        f"  {_BULL} Trained    {mdcode(dt_plain)}",
        f"  {_BULL} Samples    {mdcode(s['n_train_samples'])}",
        f"  {_BULL} CV ROC-AUC {mdcode(cv_str)}",
        f"  {_BULL} Pump rate  {mdcode(fmt_prob(s['pump_rate']))}",
        f"  {_BULL} ML weight  {mdcode(str(round(s['ml_weight']*100)) + '%')}",
        f"`{_DIV_SM}`",
        f"  BUY ≥ {mdcode(fmt_prob(s['buy_threshold']))}   "
        f"WATCH ≥ {mdcode(fmt_prob(s['watch_threshold']))}",
    ]
    if drift_detected:
        lines += ["", f"⚠️  {mdbold('Drift detected')}  ΔAUC {mdcode(drift_delta)}"]
    return "\n".join(lines)


# ── Signal snapshot ────────────────────────────────────────────────────────

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
        return mditalic("No signals recorded yet\\.")

    lines = [f"🗒  {mdbold(f'Last {len(rows)} Signals')}", f"`{_DIV}`", ""]
    for s in rows:
        age = now_ts() - safe_int(s["created_at"])
        if age < 3600:    age_str = f"{age // 60}m"
        elif age < 86400: age_str = f"{age // 3600}h"
        else:             age_str = f"{age // 86400}d"

        rec   = s["recommendation"] or ""
        emoji = REC_EMOJI.get(rec, "")
        score = safe_int(s["score"])
        prob  = safe_float(s["probability"])
        mc    = safe_float(s["market_cap_at_signal"])
        name  = esc(s["name"] or "?")
        sym   = esc(s["symbol"] or "?")
        rf    = s["red_flags"]

        lines.append(
            f"{score_emoji(score)} {mdbold(name)} `${sym}`  "
            f"{emoji} `{score}/10`  `{fmt_prob(prob)}`\n"
            f"   MC {mdcode(fmt_usd(mc))}  {mditalic(age_str + ' ago')}"
        )
        if rf:
            lines.append(f"   🚩 {mditalic(esc(rf))}")
        lines.append("")
    return "\n".join(lines)


# ── Real trading status ────────────────────────────────────────────────────

def text_trading_status() -> str:
    s = real_stats()
    trades = get_open_real_trades()

    if trades:
        avg_sl   = sum(t.dynamic_sl_pct   or REAL_STOP_LOSS_PCT   for t in trades) / len(trades)
        avg_tp   = sum(t.dynamic_tp_pct   or REAL_TAKE_PROFIT_PCT for t in trades) / len(trades)
        avg_time = sum(t.dynamic_time_stop or REAL_TIME_STOP_SEC   for t in trades) / len(trades)
    else:
        avg_sl, avg_tp, avg_time = REAL_STOP_LOSS_PCT, REAL_TAKE_PROFIT_PCT, REAL_TIME_STOP_SEC

    wr     = s["win_rate"]
    pnl    = s["total_pnl_sol"]
    status = _status_dot(s["enabled"])
    pnl_e  = _pnl_emoji(pnl)

    return "\n".join([
        f"⚡  {mdbold('Real Trading')}  {_net_badge(s['enabled'])}",
        f"`{_DIV}`",
        f"  {status}   {_BULL} SL `{avg_sl:.1f}%`  TP `+{avg_tp:.1f}%`  T `{fmt_duration(int(avg_time))}`",
        "",
        f"  Open      {mdcode(s['open_positions'])}",
        f"  Closed    {mdcode(s['closed_positions'])}  "
        f"`{s['wins']}W` / `{s['losses']}L`",
        f"  Win rate  `{_bar(wr/100)}`  {mdcode(f'{wr:.1f}%')}",
        f"  Avg P&L   {mdcode(fmt_pct(s['avg_pnl_pct'], 2, signed=True))}",
        f"  Total P&L {pnl_e} {mdcode(f'{pnl:+.4f} SOL')}",
        f"  Max DD    " + mdcode(f"{s['max_drawdown_sol']:.4f} SOL") + "",
    ])


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
        failed_n = conn.execute(
            "SELECT COUNT(*) AS c FROM real_trades WHERE status='FAILED_EXIT'"
        ).fetchone()["c"]

    n      = s["closed_positions"]
    wr     = s["win_rate"]
    pnl    = s["total_pnl_sol"]
    status = _status_dot(s["enabled"])

    lines = [
        f"📑  {mdbold('Trading Report')}  {_net_badge(s['enabled'])}",
        f"`{_DIV}`",
        f"  {status}  Size `{REAL_POSITION_SIZE_SOL} SOL`  "
        f"SL `{REAL_STOP_LOSS_PCT}%`  TP `+{REAL_TAKE_PROFIT_PCT}%`",
        "",
        f"◈  {mdbold('Performance')}",
    ]

    if n == 0:
        lines.append(mditalic("  No closed trades yet\\."))
    else:
        lines += [
            f"  Closed    {mdcode(n)}  `{s['wins']}W` / `{s['losses']}L`",
            f"  Win rate  `{_bar(wr/100)}`  {mdcode(f'{wr:.1f}%')}",
            f"  Avg P&L   {mdcode(fmt_pct(s['avg_pnl_pct'], 2, signed=True))}",
            f"  Best      {mdcode(fmt_pct(s['best_pnl_pct'],  2, signed=True))}",
            f"  Worst     {mdcode(fmt_pct(s['worst_pnl_pct'], 2, signed=True))}",
            f"  Total     {_pnl_emoji(pnl)} {mdcode(f'{pnl:+.4f} SOL')}",
            f"  Max DD    " + mdcode(f"{s['max_drawdown_sol']:.4f} SOL") + "",
        ]
        if failed_n:
            lines.append(f"  ⚠️ Failed exits  {mdcode(failed_n)}")

    if reason_stats:
        lines += ["", f"◈  {mdbold('Exit Breakdown')}"]
        _reason_emoji = {
            "TAKE_PROFIT": "🎯", "STOP_LOSS": "🛑",
            "TRAILING_STOP": "📉", "TIME_STOP": "⏱",
            "FAILED_EXIT_RETRY": "🔁",
        }
        for r in reason_stats:
            reason = r["reason"] or "unknown"
            cnt    = safe_int(r["cnt"])
            avg    = r["avg_pct"]
            total  = r["total_sol"]
            emoji  = next((v for k, v in _reason_emoji.items() if k in reason), "◆")
            pct_s  = fmt_pct(avg, 1, signed=True) if avg is not None else "—"
            sol_s  = f"{total:+.4f} SOL" if total is not None else "—"
            short  = reason.replace("_", " ").split("_")[0] if "_" in reason else reason
            lines.append(
                f"  {emoji} {mdcode(f'{short:<16}')} "
                f"{mdcode(cnt)}x   avg {mdcode(pct_s)}   {mdcode(sol_s)}"
            )

    open_trades = get_open_real_trades()
    lines += ["", f"◈  {mdbold(f'Open Positions  ({len(open_trades)})')}"]
    if not open_trades:
        lines.append(mditalic("  None\\."))
    else:
        ts_now = now_ts()
        for t in open_trades:
            age = fmt_duration(ts_now - t.entry_time)
            lines.append(
                f"  ▸ {mdbold(esc(t.name or t.mint[:8]))}\n"
                f"    Entry {mdcode(fmt_usd(t.entry_mc, 0))}  "
                f"Age {mdcode(age)}  {mdcode(f'{t.position_size_sol:.4f} SOL')}\n"
                f"    {mditalic(f'SL {t.dynamic_sl_pct:.1f}%  TP {t.dynamic_tp_pct:.1f}%  T {fmt_duration(t.dynamic_time_stop)}')}"
            )

    lines += ["", f"◈  {mdbold('Last 15 Trades')}"]
    if not recent:
        lines.append(mditalic("  No closed trades yet\\."))
    else:
        for r in recent:
            mint     = r["mint"] or ""
            name_raw = esc(r["name"] or r["symbol"] or (mint[:6] if mint else "?"))
            pnl_pct  = safe_float(r["pnl_pct"])
            pnl_sol  = safe_float(r["pnl_sol"])
            entry_mc = safe_float(r["entry_mc"])
            exit_mc  = safe_float(r["exit_mc"])
            entry_t  = safe_int(r["entry_time"])
            exit_t   = safe_int(r["exit_time"])
            reason   = esc(r["reason"] or "?")
            dur      = fmt_duration(exit_t - entry_t) if entry_t and exit_t and exit_t > entry_t else "?"
            emoji    = _pnl_emoji(pnl_pct)
            mc_str   = (f"{mdcode(fmt_usd(entry_mc))} → {mdcode(fmt_usd(exit_mc))}"
                        if entry_mc and exit_mc else "")
            lines.append(
                f"  {emoji} {mdbold(name_raw)}  "
                f"`{fmt_pct(pnl_pct, 1, signed=True)}`  "
                f"`{pnl_sol:+.4f} SOL`\n"
                f"    {mc_str}  {mdcode(dur)}  {mditalic(reason)}"
            )

    return "\n".join(lines)


# ── Stats dashboard ────────────────────────────────────────────────────────

def query_time_to_pump_data() -> list[dict]:
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
    pumps   = out_map.get("PUMP", 0) + out_map.get("MOON", 0)
    rugs    = out_map.get("RUG", 0)
    total   = sum(out_map.values())
    pump_rt = pumps / total * 100 if total else 0

    eng    = engine.status()
    cv_str = f"{eng['cv_auc']:.3f}" + (f" ±{eng['cv_auc_std']:.3f}" if eng["cv_auc_std"] else "")
    wr     = s["win_rate"]
    pnl    = s["total_pnl_sol"]

    ttp = query_time_to_pump_data()

    lines = [
        f"📊  {mdbold('Dashboard')}  {_net_badge(s['enabled'])}",
        f"`{_DIV}`",
        "",
        f"⚡  {mdbold('Trading')}",
        f"  {_status_dot(s['enabled'])}",
        f"  Open      {mdcode(s['open_positions'])}",
        f"  Closed    {mdcode(s['closed_positions'])}  "
        f"`{s['wins']}W` / `{s['losses']}L`",
        f"  Win rate  `{_bar(wr/100)}`  {mdcode(f'{wr:.1f}%')}",
        f"  Avg P&L   {mdcode(fmt_pct(s['avg_pnl_pct'], 2, signed=True))}",
        f"  Total P&L {_pnl_emoji(pnl)} {mdcode(f'{pnl:+.4f} SOL')}",
        "",
        f"📡  {mdbold('Last 24h')}",
        f"  Signals   {mdcode(sig_24h)}",
        f"  Pump rate `{_bar(pump_rt/100)}`  {mdcode(f'{pump_rt:.1f}%')}  "
        f"`{pumps}/{total}`",
        f"  Rugs      {mdcode(rugs)}",
        "",
        f"🤖  {mdbold('Model')}",
        f"  Mode      {mdcode(eng['mode'])}",
        f"  CV AUC    {mdcode(cv_str)}",
        f"  Samples   {mdcode(eng['n_train_samples'])}",
        "",
        f"🛡  {mdbold('Defense')}",
        f"  Blacklist {mdcode(bl_count)} creators",
    ]

    if ttp:
        lines += ["", f"⏱  {mdbold('Time to peak  last 7d')}"]
        for r in ttp:
            lines.append(
                f"  Score `{r['bucket']}`  median {mdcode(str(round(r['median_min'])) + 'm')}  "
                f"n={mdcode(r['n'])}"
            )

    return "\n".join(lines)


# ── Wallet ─────────────────────────────────────────────────────────────────

async def text_wallet() -> str:
    from .real_trading import _load_wallet
    wallet  = _load_wallet()
    address = wallet["pubkey"] if wallet else "not loaded"
    sol_bal = await get_wallet_sol_balance()
    s       = real_stats()
    pnl     = s["total_pnl_sol"]

    # Shorten address for display: first 4 + last 4
    short_addr = f"{address[:4]}…{address[-4:]}" if len(address) > 10 else address

    return "\n".join([
        f"💎  {mdbold('SOL Wallet')}  {_net_badge(s['enabled'])}",
        f"`{_DIV}`",
        f"  {_BULL} Address   {mdcode(short_addr)}",
        f"  {_BULL} Balance   {mdcode(f'{sol_bal:.4f} SOL')}",
        f"  {_BULL} Positions {mdcode(s['open_positions'])} open",
        "",
        f"◈  {mdbold('P&L Summary')}",
        f"  Closed    {mdcode(s['closed_positions'])}  "
        f"`{s['wins']}W` / `{s['losses']}L`",
        f"  Win rate  `{_bar(s['win_rate']/100)}`  {mdcode('{:.1f}%'.format(s['win_rate']))}",
        f"  Total P&L {_pnl_emoji(pnl)} {mdcode(f'{pnl:+.4f} SOL')}",
        f"  Max DD    " + mdcode(f"{s['max_drawdown_sol']:.4f} SOL") + "",
        "",
        mditalic(
            f"Size {REAL_POSITION_SIZE_SOL} SOL/trade  ·  "
            f"Fee {REAL_FEE_PCT}%  ·  Slippage {REAL_SLIPPAGE_PCT}%"
        ),
    ])


# ── Top performers ─────────────────────────────────────────────────────────

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
        return mditalic(f"No PUMP/MOON outcomes in last {days}d\\.")
    lines = [
        f"🏆  {mdbold(f'Top Performers  last {days}d')}",
        f"`{_DIV}`",
        "",
    ]
    for i, r in enumerate(rows, 1):
        name    = esc(r["name"] or r["symbol"] or (r["mint"] or "?")[:8])
        score   = safe_int(r["score"])
        pct     = safe_float(r["pct_change"])
        outcome = r["outcome"] or ""
        emoji   = "🚀" if outcome == "MOON" else "📈"
        mc      = safe_float(r["market_cap_at_signal"])
        age     = fmt_duration(now_ts() - safe_int(r["created_at"]))
        lines.append(
            f"  {i}\\. {emoji} {mdbold(name)}  {mdcode(f'{score}/10')}  "
            f"`{fmt_pct(pct, 1, signed=True)}`\n"
            f"     from {mdcode(fmt_usd(mc))}  {mditalic(age + ' ago')}"
        )
        if r["mint"]:
            lines.append(f"     🔗 [pump\\.fun]({PUMP_FRONT}/{r['mint']})")
        lines.append("")
    return "\n".join(lines)


# ── Health ─────────────────────────────────────────────────────────────────

def text_health(state: BotState, engine: ScoringEngine) -> str:
    import time
    ts_now  = now_ts()
    uptime  = ts_now - safe_int(get_state("bot_started_at", str(ts_now)))
    dead_sec = int(time.time() - state.last_coin_ts)
    live    = dead_sec < STREAM_DEAD_ALERT_SEC

    with closing(db_conn()) as conn:
        total_sig  = conn.execute("SELECT COUNT(*) AS c FROM signals").fetchone()["c"]
        sig_1hr    = conn.execute(
            "SELECT COUNT(*) AS c FROM signals WHERE created_at >= ?",
            (ts_now - 3600,)).fetchone()["c"]
        dead_total = conn.execute(
            "SELECT COUNT(*) AS c FROM dead_letters").fetchone()["c"]
        dead_pend  = conn.execute(
            "SELECT COUNT(*) AS c FROM dead_letters WHERE retry_count < ? OR retry_count IS NULL",
            (DEAD_LETTER_MAX_RETRIES,)).fetchone()["c"]
        open_trades = conn.execute(
            "SELECT COUNT(*) AS c FROM real_trades WHERE status='OPEN'").fetchone()["c"]
        failed_exit = conn.execute(
            "SELECT COUNT(*) AS c FROM real_trades WHERE status='FAILED_EXIT'").fetchone()["c"]
        pending_lb  = conn.execute(
            "SELECT COUNT(*) AS c FROM lookbacks WHERE checked=0").fetchone()["c"]

    drift       = get_state("model_drift_detected", "0") == "1"
    stream_icon = f"{_DOT} `LIVE`  last coin {mdcode(fmt_duration(dead_sec) + ' ago')}" if live \
                  else f"🔴 `DEAD`  no data for {mdcode(fmt_duration(dead_sec))}"
    model_icon  = f"⚠️ `DRIFT`" if drift else f"{_DOT} {mdcode(engine.mode_label)}"

    lines = [
        f"❤️  {mdbold('Bot Health')}",
        f"`{_DIV}`",
        f"  Uptime    {mdcode(fmt_duration(uptime))}",
        "",
        f"  Stream    {stream_icon}",
        f"  Model     {model_icon}",
        "",
        f"◈  {mdbold('Activity')}",
        f"  Signals   {mdcode(total_sig)} total  {mdcode(sig_1hr)} last hr",
        f"  Lookbacks {mdcode(pending_lb)} pending",
        f"  Dead msgs {mdcode(dead_total)} total  {mdcode(dead_pend)} retry",
        f"  Open pos  {mdcode(open_trades)}",
    ]
    if failed_exit:
        lines.append(f"  ⚠️ Failed exits {mdcode(failed_exit)}")

    lines += [
        "",
        f"◈  {mdbold('Paths')}",
        f"  DB   {mdcode(DB_PATH)}",
        f"  Bak  {mdcode(DB_BACKUP_PATH)}",
        f"  Log  {mdcode(LOG_PATH)}",
    ]
    return "\n".join(lines)


# ── Help ───────────────────────────────────────────────────────────────────

def text_help() -> str:
    def cmd(c, desc):
        return f"  {mdcode(c)}  {mditalic(desc)}"

    return "\n".join([
        f"🤖  {mdbold('Pump Monitor')}  `v2.0`",
        f"`{_DIV}`",
        "",
        f"◈  {mdbold('Alerts')}",
        cmd("/menu",             "open control menu"),
        cmd("/monitor_on",       "enable alerts"),
        cmd("/monitor_off",      "disable alerts"),
        cmd("/set_threshold N",  "set score threshold 1-10"),
        "",
        f"◈  {mdbold('Signals & ML')}",
        cmd("/snapshot",         "recent scored signals"),
        cmd("/score <mint>",     "manually score any coin"),
        cmd("/outcomes",         "outcome tracker"),
        cmd("/model",            "ML model status"),
        cmd("/train",            "retrain model now"),
        cmd("/scoring_mode",     "scoring engine config"),
        cmd("/features",         "feature importances"),
        cmd("/keywords",         "learned keyword model"),
        cmd("/market",           "market context 24hr"),
        cmd("/backtest",         "signal performance by score"),
        "",
        f"◈  {mdbold('Real Trading')}",
        cmd("/real_on",          "enable SOL trading"),
        cmd("/real_off",         "disable SOL trading"),
        cmd("/real_status",      "trading summary"),
        cmd("/real_report",      "full P&L report"),
        cmd("/real_balance",     "SOL wallet balance"),
        cmd("/last",             "most recent trade"),
        "",
        f"◈  {mdbold('Watchlist')}",
        cmd("/watch <mint>",     "add to watchlist"),
        cmd("/unwatch <mint>",   "remove from watchlist"),
        cmd("/watchlist",        "show watchlist"),
        "",
        f"◈  {mdbold('Stats & Tools')}",
        cmd("/stats",            "full dashboard"),
        cmd("/wallet",           "SOL wallet & P&L"),
        cmd("/top [days]",       "top performers"),
        cmd("/health",           "bot health check"),
        cmd("/blacklist",        "manage creator blacklist"),
        "",
        mditalic("Tap Menu or type any command\\."),
    ])


# ── Real status ────────────────────────────────────────────────────────────

def text_real_status(engine) -> str:
    from .real_trading import (
        real_stats, SOLANA_NETWORK, REAL_POSITION_SIZE_SOL,
        REAL_STOP_LOSS_PCT, REAL_TAKE_PROFIT_PCT, REAL_TIME_STOP_SEC,
        REAL_MIN_SCORE, REAL_MIN_PROB,
    )
    s  = real_stats()
    wr = s["win_rate"]

    return "\n".join([
        f"⚡  {mdbold('Real Trading Status')}",
        f"`{_DIV}`",
        f"  {_status_dot(s['enabled'])}  {_net_badge(s['enabled'])}",
        "",
        f"◈  {mdbold('Positions')}",
        f"  Open      {mdcode(s['open_positions'])}",
        f"  Closed    {mdcode(s['closed_positions'])}",
        f"  Win rate  `{_bar(wr/100)}`  {mdcode(f'{wr:.1f}%')}",
        f"  Avg P&L   {mdcode(fmt_pct(s['avg_pnl_pct'], 1, signed=True))}",
        "",
        f"◈  {mdbold('Config')}",
        f"  Size      {mdcode(f'{REAL_POSITION_SIZE_SOL} SOL')} per trade",
        f"  SL / TP   {mdcode(f'{REAL_STOP_LOSS_PCT}%')} / {mdcode(f'+{REAL_TAKE_PROFIT_PCT}%')}",
        f"  Time stop {mdcode(fmt_duration(REAL_TIME_STOP_SEC))}",
        f"  Min score {mdcode(REAL_MIN_SCORE)}   Min prob {mdcode(fmt_prob(REAL_MIN_PROB))}",
    ])


def text_real_report() -> str:
    from .real_trading import real_stats, get_open_real_trades, SOLANA_NETWORK
    s = real_stats()
    trades = get_open_real_trades()
    ts_now = now_ts()

    lines = [
        f"📑  {mdbold('Trading Report')}  {_net_badge(s['enabled'])}",
        f"`{_DIV}`",
        f"  {_status_dot(s['enabled'])}",
        "",
        f"  Open {mdcode(s['open_positions'])}   "
        f"Closed {mdcode(s['closed_positions'])}",
        f"  Avg P&L {mdcode(fmt_pct(s['avg_pnl_pct'], 1, signed=True))}",
        "",
        f"◈  {mdbold(f'Open Positions  ({len(trades)})')}",
    ]
    if not trades:
        lines.append(mditalic("  None\\."))
    else:
        for t in trades:
            age = fmt_duration(ts_now - t.entry_time)
            lines.append(
                f"  ▸ {mdbold(esc(t.name or t.mint[:8]))}  "
                f"entry {mdcode(fmt_usd(t.entry_mc, 0))}  "
                f"age {mdcode(age)}"
            )
    return "\n".join(lines)


# ── Last trade ─────────────────────────────────────────────────────────────

def text_last_trade() -> str:
    from .config import PUMP_FRONT

    with closing(db_conn()) as conn:
        row = conn.execute(
            "SELECT * FROM real_trades ORDER BY entry_time DESC LIMIT 1"
        ).fetchone()

    if not row:
        return mditalic("No trades yet\\.")

    name   = esc(row["name"] or "Unknown")
    symbol = esc(row["symbol"] or "???")
    mint   = row["mint"]
    status = row["status"]
    entry_mc = float(row["entry_mc"])
    pnl_pct  = float(row["pnl_pct"] or 0)
    pnl_sol  = float(row["pnl_sol"] or 0)
    reason   = esc(row["reason"] or "")
    size_sol = float(row["position_size_sol"] or 0)
    tx_sig   = row["tx_signature"] or ""
    short_tx = f"{tx_sig[:8]}…" if tx_sig else "?"

    if status == "OPEN":
        age = fmt_duration(now_ts() - int(row["entry_time"]))
        return "\n".join(filter(None, [
            f"⚡  {mdbold('Latest Trade — OPEN')}",
            f"`{_DIV}`",
            f"  {mdbold(name)}  `${symbol}`",
            "",
            f"  {_BULL} Entry MC  {mdcode(fmt_usd(entry_mc, 0))}",
            f"  {_BULL} Size      {mdcode(f'{size_sol} SOL')}",
            f"  {_BULL} Age       {mdcode(age)}",
            f"  {_BULL} TX        {mdcode(short_tx)}",
            f"  🔗 [pump\\.fun]({PUMP_FRONT}/{mint})" if mint else "",
        ]))
    else:
        arrow   = "📈" if pnl_pct >= 0 else "📉"
        age     = fmt_duration(now_ts() - int(row["exit_time"]))
        ex_sig  = row["exit_tx_signature"] or ""
        short_ex = f"{ex_sig[:8]}…" if ex_sig else "?"
        dur     = fmt_duration(
            int(row["exit_time"]) - int(row["entry_time"])
        ) if row["exit_time"] and row["entry_time"] else "?"

        return "\n".join(filter(None, [
            f"{arrow}  {mdbold('Latest Trade — CLOSED')}",
            f"`{_DIV}`",
            f"  {mdbold(name)}  `${symbol}`",
            "",
            f"  {_BULL} P&L       {_pnl_emoji(pnl_pct)} `{fmt_pct(pnl_pct, 2, signed=True)}`  "
            f"`{pnl_sol:+.4f} SOL`",
            f"  {_BULL} Reason    {mdcode(reason)}",
            f"  {_BULL} Duration  {mdcode(dur)}",
            f"  {_BULL} Closed    {mdcode(age + ' ago')}",
            f"  {_BULL} Entry TX  {mdcode(short_tx)}",
            f"  {_BULL} Exit TX   {mdcode(short_ex)}",
            f"  🔗 [pump\\.fun]({PUMP_FRONT}/{mint})" if mint else "",
        ]))
