"""Slash-command handlers."""
import asyncio
import logging
from collections import defaultdict
from contextlib import closing
from typing import Optional

import aiohttp
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest, TelegramError
from telegram.ext import ContextTypes

from .alerts import build_keyboard, build_message
from .config import (
    DEFAULT_THRESHOLD, HTTP_TIMEOUT_SEC, MIN_TRAIN_SAMPLES, ML_AVAILABLE,
    ML_LABEL_WINDOW, PUMP_THRESHOLD_PCT,
    RUG_THRESHOLD_PCT,
)
from .db import db_conn, db_write, set_state, upsert_chat
from .enrichment import enrich_with_rpc, fetch_mc_momentum_from_db
from .keyboards import MENU_HEADER, main_menu_keyboard
from .lookback import train_executor
from .scoring import ScoringEngine
from .state import BotState, blacklist_cache
from .ui_text import (
    format_top_performers, query_top_performers, text_features,
    text_health, text_help, text_keywords, text_market, text_model,
    text_monitor_status, text_outcomes,
    text_scoring_mode, text_snapshot, text_stats,
    text_trading_report, text_trading_status, text_wallet,
)
from .utils import (
    fmt_duration, fmt_pct, fmt_prob, fmt_usd,
    mdbold, mdcode, mditalic, now_ts, safe_float, safe_int, strip_md2,
)
from .real_trading import (
    real_engine, real_stats, swap_sol_for_token, get_wallet_sol_balance,
    get_open_real_trades, SOLANA_NETWORK, REAL_POSITION_SIZE_SOL, SOLANA_WALLET_PATH,
    REAL_STOP_LOSS_PCT, REAL_TAKE_PROFIT_PCT, REAL_TIME_STOP_SEC,
)

log = logging.getLogger(__name__)

PM = "MarkdownV2"


def _state(ctx: ContextTypes.DEFAULT_TYPE) -> BotState:
    return ctx.bot_data["state"]


async def _reply(update: Update, text: str,
                 kb: Optional[InlineKeyboardMarkup] = None) -> None:
    try:
        await update.message.reply_text(text, parse_mode=PM, reply_markup=kb)
    except BadRequest as e:
        if "parse" in str(e).lower():
            log.warning("MD2 parse failed on reply: %s", e)
            try:
                await update.message.reply_text(strip_md2(text), reply_markup=kb)
            except TelegramError as e2:
                log.error("Plain reply failed: %s", e2)
        else:
            log.debug("reply BadRequest: %s", e)


async def do_train(engine: ScoringEngine) -> str:
    """Shared training handler — used by both /train and the Train button."""
    if not ML_AVAILABLE:
        return f"⚠️ {mdbold('ML not available')} — install scikit-learn, joblib, numpy"
    try:
        loop    = asyncio.get_running_loop()
        trained = await loop.run_in_executor(train_executor, engine.train)
        if trained:
            s = engine.status()
            return (
                f"✅ {mdbold('Model trained!')}\n"
                f"CV AUC: {mdcode(str(round(s['cv_auc'],3)))} "
                f"±{mdcode(str(round(s['cv_auc_std'],3)))} \\| "
                f"ML weight: {mdcode(str(round(s['ml_weight']*100))+'%')}\n"
                f"BUY ≥ {mdcode(fmt_prob(s['buy_threshold']))} \\| "
                f"WATCH ≥ {mdcode(fmt_prob(s['watch_threshold']))}"
            )
        with closing(db_conn()) as conn:
            n = conn.execute(
                "SELECT COUNT(*) AS c FROM lookbacks "
                "WHERE window_label=? AND checked=1",
                (ML_LABEL_WINDOW,),
            ).fetchone()["c"]
        return f"⚠️ Need {mdcode(MIN_TRAIN_SAMPLES)} samples, have {mdcode(n)}"
    except Exception as e:
        return f"❌ {mdcode(str(e))}"


# ---------- Basic ----------


from .config import ALLOWED_CHAT_IDS

async def _check_allowed(update: Update) -> bool:
    if not ALLOWED_CHAT_IDS:
        return True
    chat_id = update.effective_chat.id
    if chat_id not in ALLOWED_CHAT_IDS:
        if update.message:
            await update.message.reply_text("⛔ Unauthorized.")
        return False
    return True

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    await _reply(
        update,
        f"🤖 {mdbold('Pump.fun Dynamic Monitor v1.1')}\n\n"
        f"Tap {mdbold('Menu')} below to control the bot without typing any commands\\.",
        kb=InlineKeyboardMarkup([[
            InlineKeyboardButton("📋 Open Menu", callback_data="menu"),
        ]]),
    )


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    await _reply(update, text_help())


async def cmd_menu(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    await _reply(update, MENU_HEADER, kb=main_menu_keyboard())


# ---------- Monitor ----------

async def cmd_monitor_on(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    state = _state(ctx)
    cid   = update.effective_chat.id
    th    = state.alerts.get(cid, DEFAULT_THRESHOLD)
    state.alerts[cid] = th
    upsert_chat(cid, alerts_enabled=1, threshold=th)
    await _reply(update, f"🟢 {mdbold('Alerts ON')} — threshold {mdcode(f'{th}/10')}")


async def cmd_monitor_off(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    state = _state(ctx)
    state.alerts.pop(update.effective_chat.id, None)
    upsert_chat(update.effective_chat.id, alerts_enabled=0)
    await _reply(update, f"🔴 {mdbold('Alerts OFF')}")


async def cmd_monitor_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    await _reply(update, text_monitor_status(update.effective_chat.id, _state(ctx)))


async def cmd_set_threshold(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    state = _state(ctx)
    cid   = update.effective_chat.id
    if not ctx.args or not ctx.args[0].lstrip("-").isdigit():
        await _reply(update, f"Usage: {mdcode('/set_threshold <1-10>')}")
        return
    val = int(ctx.args[0])
    if not 1 <= val <= 10:
        await _reply(update, "Must be 1–10\\.")
        return
    state.alerts[cid] = val
    upsert_chat(cid, threshold=val, alerts_enabled=1)
    await _reply(update, f"✅ Threshold → {mdcode(f'{val}/10')}")


# ---------- Info ----------

async def cmd_scoring_mode(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    await _reply(update, text_scoring_mode(ctx.bot_data["engine"]))


async def cmd_features(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    await _reply(update, text_features(ctx.bot_data["engine"]))


async def cmd_keywords(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    await _reply(update, text_keywords(ctx.bot_data["engine"]))


async def cmd_market(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    await _reply(update, text_market(ctx.bot_data["market_ctx"]))


async def cmd_outcomes(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    await _reply(update, text_outcomes())


async def cmd_model(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    await _reply(update, text_model(ctx.bot_data["engine"]))


async def cmd_train(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    engine: ScoringEngine = ctx.bot_data["engine"]
    await _reply(update, "🏋 Training\\.\\.\\.")
    await _reply(update, await do_train(engine))


async def cmd_snapshot(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    await _reply(update, text_snapshot())


async def cmd_last(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show the most recent real SOL trade with its current P&L."""
    if not await _check_allowed(update): return
    from .db import db_conn
    from contextlib import closing as _closing
    from .config import PUMP_FRONT

    with _closing(db_conn()) as conn:
        row = conn.execute(
            "SELECT * FROM real_trades ORDER BY entry_time DESC LIMIT 1"
        ).fetchone()

    if not row:
        await _reply(update, "No real trades yet\\. Use /real\\_on to enable trading\\.")
        return

    name     = row["name"] or "Unknown"
    symbol   = row["symbol"] or "???"
    mint     = row["mint"]
    status   = row["status"]
    entry_mc = float(row["entry_mc"])
    entry_sol = float(row["entry_sol"])

    if status == "OPEN":
        lines = [
            f"⚡ {mdbold('LATEST REAL TRADE — OPEN')}",
            f"{mdbold(name)} \\({mdcode('$' + symbol)}\\)",
            "",
            f"💰 Entry MC: {mdcode(fmt_usd(entry_mc, 0))}",
            f"🔷 Size: {mdcode(f'{entry_sol:.4f} SOL')}",
            f"🕐 Opened: {mdcode(fmt_duration(now_ts() - int(row['entry_time'])))} ago",
        ]
    else:
        pnl_pct  = float(row["pnl_pct"] or 0)
        pnl_sol  = float(row["pnl_sol"] or 0)
        reason   = row["reason"] or ""
        arrow    = "📈" if pnl_pct >= 0 else "📉"
        # exit_time is NULL for FAILED_EXIT / ABANDONED trades
        exit_time = row["exit_time"]
        closed_ago = (
            mdcode(fmt_duration(now_ts() - int(exit_time))) + " ago"
            if exit_time else mdcode(status)
        )
        lines = [
            f"{arrow} {mdbold('LATEST REAL TRADE')} — {mdcode(status)}",
            f"{mdbold(name)} \\({mdcode('$' + symbol)}\\)",
            "",
            f"{arrow} P&L: {mdcode(f'{pnl_pct:+.1f}%')}  "
            f"\\({mdcode(f'{pnl_sol:+.4f} SOL')}\\)",
            f"📌 Reason: {mdcode(reason)}",
            f"🕐 Closed: {closed_ago}",
        ]

    if mint:
        lines.append(f"🪙 {mdcode(mint)}")
        lines.append(f"🔗 [Pump\\.fun]({PUMP_FRONT}/{mint})")

    await _reply(update, "\n".join(lines))


# ---------- Diagnostics ----------

async def cmd_health(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    await _reply(update, text_health(_state(ctx), ctx.bot_data["engine"]))


async def cmd_stats(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    await _reply(update, text_stats(_state(ctx), ctx.bot_data["engine"]))


# ---------- Score / backtest / watchlist ----------

async def cmd_score(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    if not ctx.args or len(ctx.args[0]) < 32:
        await _reply(update, f"Usage: {mdcode('/score <mint_address>')}")
        return
    mint = ctx.args[0].strip()
    await _reply(update, f"🔍 Fetching {mdcode(mint[:8])}\\.\\.\\.")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://frontend-api-v3.pump.fun/coins/{mint}",
                timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT_SEC),
            ) as resp:
                if resp.status != 200:
                    await _reply(update, f"❌ pump\\.fun returned HTTP {mdcode(resp.status)}")
                    return
                data = await resp.json(content_type=None)

            if not isinstance(data, dict) or not data.get("mint"):
                await _reply(update, "❌ Coin not found")
                return

            engine: ScoringEngine = ctx.bot_data["engine"]

            coin = {**data, "mint": mint}
            # Enrich with on-chain data so on-chain features are populated,
            # matching the live pipeline behaviour.
            coin = await enrich_with_rpc(coin, session)

        loop = asyncio.get_running_loop()
        coin["_mc_momentum_pct"] = await loop.run_in_executor(
            None, fetch_mc_momentum_from_db, mint,
        )
        result = engine.score(coin)
        text = build_message(coin, result)
        kb = build_keyboard(coin)
        await _reply(
            update,
            text + "\n\n" + mditalic("\\(manual scoring — not saved\\)"),
            kb=kb,
        )
    except Exception as e:
        await _reply(update, f"❌ {mdcode(str(e))}")


async def cmd_backtest(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    await _reply(update, "📊 Running backtest\\.\\.\\.")
    try:
        with closing(db_conn()) as conn:
            rows = conn.execute("""
                SELECT s.score, lb.outcome, lb.pct_change
                FROM signals s
                JOIN lookbacks lb ON lb.signal_id = s.id
                WHERE lb.window_label = ?
                  AND lb.checked = 1
                  AND lb.outcome IS NOT NULL
                ORDER BY s.score
            """, (ML_LABEL_WINDOW,)).fetchall()

        if not rows:
            await _reply(update, mditalic("No labeled data yet — check back later"))
            return

        buckets: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            score = safe_int(r["score"])
            pct   = safe_float(r["pct_change"])
            if   score >= 9: bucket = "9-10 🔥"
            elif score >= 7: bucket = "7-8 ⭐"
            elif score >= 5: bucket = "5-6 👍"
            else:            bucket = "1-4 🤔"
            buckets[bucket].append(pct)

        lines = [
            f"🎯 {mdbold('Backtest Results')}",
            f"{mditalic(f'Label window: {ML_LABEL_WINDOW} | {len(rows)} total labeled signals')}",
            "",
        ]

        for bucket in ["9-10 🔥", "7-8 ⭐", "5-6 👍", "1-4 🤔"]:
            vals = buckets.get(bucket, [])
            if not vals:
                continue
            n = len(vals)
            pumps = sum(1 for v in vals if v >= PUMP_THRESHOLD_PCT)
            rugs  = sum(1 for v in vals if v <= RUG_THRESHOLD_PCT)
            avg_pct = sum(vals) / n
            lines += [
                f"{mdbold(bucket)}",
                f"Signals {mdcode(n)} \\| "
                f"Pump rate {mdcode(f'{pumps/n*100:.1f}%')} \\| "
                f"Rug rate {mdcode(f'{rugs/n*100:.1f}%')}",
                f"Avg outcome {mdcode(fmt_pct(avg_pct, 1, signed=True))}",
                "",
            ]
        await _reply(update, "\n".join(lines))
    except Exception as e:
        await _reply(update, f"❌ {mdcode(str(e))}")


async def cmd_watch(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    if not ctx.args or len(ctx.args[0]) < 32:
        await _reply(update, f"Usage: {mdcode('/watch <mint_address>')}")
        return
    mint = ctx.args[0].strip()
    chat_id = update.effective_chat.id

    with closing(db_conn()) as conn:
        sig = conn.execute(
            "SELECT name, symbol FROM signals WHERE mint=? "
            "ORDER BY created_at DESC LIMIT 1",
            (mint,),
        ).fetchone()
    name   = sig["name"]   if sig else ""
    symbol = sig["symbol"] if sig else ""

    try:
        def _w():
            with closing(db_conn()) as conn, conn:
                conn.execute(
                    "INSERT OR IGNORE INTO watchlist"
                    "(chat_id,mint,name,symbol,added_at) VALUES(?,?,?,?,?)",
                    (chat_id, mint, name, symbol, now_ts()))
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, db_write, _w)
        await _reply(update, f"👁 Watching {mdbold(name or mint[:8])}")
    except Exception as e:
        await _reply(update, f"❌ {mdcode(str(e))}")


async def cmd_unwatch(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    if not ctx.args:
        await _reply(update, f"Usage: {mdcode('/unwatch <mint_address>')}")
        return
    mint = ctx.args[0].strip()
    chat_id = update.effective_chat.id

    def _w():
        with closing(db_conn()) as conn, conn:
            conn.execute(
                "DELETE FROM watchlist WHERE chat_id=? AND mint=?",
                (chat_id, mint))
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, db_write, _w)
    await _reply(update, f"✅ Removed {mdcode(mint[:8])} from watchlist")


async def cmd_watchlist(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    chat_id = update.effective_chat.id
    with closing(db_conn()) as conn:
        rows = conn.execute(
            "SELECT mint, name, symbol, added_at FROM watchlist "
            "WHERE chat_id=? ORDER BY added_at DESC",
            (chat_id,),
        ).fetchall()
    if not rows:
        await _reply(update, mditalic("Your watchlist is empty. Use /watch <mint>"))
        return

    from .config import PUMP_FRONT
    lines = [f"👁 {mdbold('Your Watchlist')} \\({len(rows)} coins\\)", ""]
    for r in rows:
        name = r["name"] or r["symbol"] or (r["mint"] or "?")[:8]
        mint = r["mint"] or ""
        age = fmt_duration(now_ts() - safe_int(r["added_at"]))
        lines.append(
            f"• {mdbold(name)} {mdcode(mint[:8])} \\| added {mdcode(age)} ago"
        )
        if mint:
            lines.append(f"   🔗 [Pump\\.fun]({PUMP_FRONT}/{mint})")
    await _reply(update, "\n".join(lines))


# ---------- Blacklist / top ----------

async def cmd_blacklist(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    args = ctx.args or []
    if not args:
        with closing(db_conn()) as conn:
            rows = conn.execute(
                "SELECT creator, reason, auto_added, added_at FROM creator_blacklist "
                "ORDER BY added_at DESC LIMIT 30"
            ).fetchall()
        if not rows:
            await _reply(update, mditalic("No blacklisted creators."))
            return
        lines = [f"🚫 {mdbold('Creator Blacklist')} \\({len(rows)}\\)", ""]
        for r in rows:
            tag = "🤖" if r["auto_added"] else "👤"
            age = fmt_duration(now_ts() - safe_int(r["added_at"]))
            lines.append(
                f"{tag} {mdcode(r['creator'][:12])} — "
                f"{mditalic(r['reason'] or 'manual')} — {mdcode(age)} ago"
            )
        lines.append("")
        lines.append(mditalic(
            "Usage: /blacklist add <wallet> [reason] | /blacklist remove <wallet>"))
        await _reply(update, "\n".join(lines))
        return

    action = args[0].lower()
    if action == "add" and len(args) >= 2:
        creator = args[1]
        reason  = " ".join(args[2:]) or "manual"
        def _w():
            with closing(db_conn()) as conn, conn:
                conn.execute(
                    "INSERT OR REPLACE INTO creator_blacklist"
                    "(creator,reason,added_at,auto_added) VALUES(?,?,?,0)",
                    (creator, reason, now_ts()))
        db_write(_w)
        blacklist_cache.invalidate()
        await _reply(update, f"🚫 Blacklisted {mdcode(creator[:12])}")
    elif action == "remove" and len(args) >= 2:
        creator = args[1]
        def _w():
            with closing(db_conn()) as conn, conn:
                conn.execute(
                    "DELETE FROM creator_blacklist WHERE creator=?",
                    (creator,))
        db_write(_w)
        blacklist_cache.invalidate()
        await _reply(update, f"✅ Removed {mdcode(creator[:12])} from blacklist")
    else:
        await _reply(update, mditalic(
            "Usage: /blacklist [add|remove] <wallet> [reason]"))


async def cmd_top(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    days = 7
    if ctx.args and ctx.args[0].isdigit():
        days = max(1, min(30, int(ctx.args[0])))
    rows = query_top_performers(days=days)
    await _reply(update, format_top_performers(rows, days))


# ---------- Real Trading ----------

async def cmd_real_on(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    if not real_engine.enabled:
        await real_engine.set_enabled(True)
        set_state("real_trading_enabled", "1")
        await _reply(update,
            f"✅ {mdbold('Real trading ON')} — network: {mdcode(SOLANA_NETWORK)}")
    else:
        await _reply(update, "Real trading is already ON.")


async def cmd_real_off(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    await real_engine.set_enabled(False)
    set_state("real_trading_enabled", "0")
    await _reply(update, f"❌ {mdbold('Real trading OFF')}")


async def cmd_real_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    await _reply(update, text_trading_status())


async def cmd_real_balance(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    bal = await get_wallet_sol_balance()
    lines = [
        f"💰 {mdbold('Wallet Balance')}",
        f"Network: {mdcode(SOLANA_NETWORK)}",
        f"SOL Balance: {mdcode(f'{bal:.4f} SOL')}  \\(USD price not fetched\\)",
        "",
        mditalic(f"Wallet file: {SOLANA_WALLET_PATH}"),
    ]
    await _reply(update, "\n".join(lines))


async def cmd_real_report(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    await _reply(update, text_trading_report())


def text_score_stats() -> str:
    """Score-tier accuracy across ALL signals ever recorded — shows real model quality.

    Uses the 4hr lookback window (same window the ML model trains on) to determine
    whether each signal did well (MOON/PUMP/UP = ≥10% gain) or didn't (STALE/DOWN/RUG).
    Also shows signals still pending an outcome.
    """
    with closing(db_conn()) as conn:
        rows = conn.execute("""
            SELECT
                s.score,
                COUNT(DISTINCT s.id)                                               AS total,
                SUM(CASE WHEN lb.outcome IN ('MOON','PUMP','UP')  THEN 1 ELSE 0 END) AS did_well,
                SUM(CASE WHEN lb.outcome IN ('STALE','DOWN','RUG') THEN 1 ELSE 0 END) AS did_not,
                SUM(CASE WHEN lb.outcome = 'MOON'  THEN 1 ELSE 0 END)              AS moon,
                SUM(CASE WHEN lb.outcome = 'PUMP'  THEN 1 ELSE 0 END)              AS pump,
                SUM(CASE WHEN lb.outcome = 'UP'    THEN 1 ELSE 0 END)              AS up,
                SUM(CASE WHEN lb.outcome = 'STALE' THEN 1 ELSE 0 END)              AS stale,
                SUM(CASE WHEN lb.outcome = 'DOWN'  THEN 1 ELSE 0 END)              AS down,
                SUM(CASE WHEN lb.outcome = 'RUG'   THEN 1 ELSE 0 END)              AS rug,
                AVG(CASE WHEN lb.checked = 1 THEN lb.pct_change END)               AS avg_pct
            FROM signals s
            LEFT JOIN lookbacks lb
                   ON lb.signal_id = s.id
                  AND lb.window_label = '4hr'
                  AND lb.checked      = 1
            WHERE s.score IS NOT NULL
            GROUP BY s.score
            ORDER BY s.score DESC
        """).fetchall()

    if not rows:
        return "📊 No signals recorded yet\."

    lines = [
        f"📊 {mdbold('Model Signal Accuracy')}",
        mditalic("All signals ever recorded, 4hr outcome window"),
        "",
    ]
    for r in rows:
        score    = int(r["score"])
        total    = int(r["total"])
        did_well = int(r["did_well"] or 0)
        did_not  = int(r["did_not"]  or 0)
        moon     = int(r["moon"]  or 0)
        pump     = int(r["pump"]  or 0)
        up       = int(r["up"]    or 0)
        stale    = int(r["stale"] or 0)
        down     = int(r["down"]  or 0)
        rug      = int(r["rug"]   or 0)
        pending  = total - did_well - did_not
        avg_pct  = float(r["avg_pct"] or 0)
        success  = did_well / (did_well + did_not) * 100 if (did_well + did_not) > 0 else 0

        lines += [
            f"{mdbold(f'{score}/10')} — {mdcode(f'{total:,} signals')}  "
            f"\\({mdcode(f'{success:.0f}%')} success rate\\)",
            f"  ✅ Did well:  {mdbold(str(did_well))}  "
            f"— 🌙 {moon}  🚀 {pump}  📈 {up}",
            f"  ❌ Did not:   {mdbold(str(did_not))}  "
            f"— ➡️ {stale}  📉 {down}  💀 {rug}",
            f"  ⏳ Pending:  {pending}  "
            f"  Avg 4hr: {mdcode(f'{avg_pct:+.1f}%' if (did_well+did_not) else 'n/a')}",
            "",
        ]
    return "\n".join(lines)


async def cmd_score_stats(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show win-rate and P&L broken down by entry score tier."""
    if not await _check_allowed(update): return
    await _reply(update, text_score_stats())


async def cmd_trade_size(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Set trade size: /trade_size 0.25"""
    if not await _check_allowed(update): return
    from .db import get_state, set_state
    args = ctx.args
    if not args:
        current = float(get_state("real_position_size_sol") or REAL_POSITION_SIZE_SOL)
        await _reply(update,
            f"💱 {mdbold('Trade Size')}\n"
            f"Current: {mdcode(f'{current} SOL')} per trade\n\n"
            f"Usage: {mdcode('/trade_size 0.25')}\n"
            f"Min: {mdcode('0.01')} | Max: {mdcode('10.0')}"
        )
        return
    try:
        val = float(args[0])
    except ValueError:
        await _reply(update, "❌ Invalid number\\. Example: /trade_size 0\\.25")
        return
    if not 0.01 <= val <= 10.0:
        await _reply(update, "❌ Must be between 0\\.01 and 10\\.0 SOL")
        return
    set_state("real_position_size_sol", str(val))
    await _reply(update, f"✅ Trade size set to {mdcode(f'{val} SOL')} per trade")


async def cmd_import_wallet(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Import wallet: /import_wallet <base58_key or JSON array>"""
    if not await _check_allowed(update): return
    import json as _json

    # Delete the message immediately for security
    try:
        await update.message.delete()
    except Exception:
        pass

    args = ctx.args
    if not args:
        await _reply(update,
            f"📥 {mdbold('Import Wallet')}\n\n"
            f"Usage: {mdcode('/import_wallet <private_key>')}\n"
            f"Accepts base58 or 64\\-byte JSON array\\.\n\n"
            "⚠️ " + mditalic('Use in private chat only\\.')
        )
        return

    raw = " ".join(args).strip()
    secret_bytes = None

    # Try JSON array
    try:
        arr = _json.loads(raw)
        if isinstance(arr, list) and len(arr) == 64:
            secret_bytes = bytes(arr)
    except Exception:
        pass

    # Try base58
    if secret_bytes is None:
        try:
            import base58 as _b58
            decoded = _b58.b58decode(raw)
            if len(decoded) == 64:
                secret_bytes = decoded
        except Exception:
            pass

    if secret_bytes is None:
        await _reply(update, "❌ Invalid key\\. Must be 64\\-byte JSON array or base58 string\\.")
        return

    try:
        from solders.keypair import Keypair
        kp = Keypair.from_bytes(secret_bytes)
        pubkey = str(kp.pubkey())
    except Exception as e:
        await _reply(update, f"❌ Failed to load keypair: {mdcode(str(e))}")
        return

    # Back up existing wallet and save new one
    import os, shutil
    wallet_path = SOLANA_WALLET_PATH
    if os.path.exists(wallet_path):
        shutil.copy(wallet_path, wallet_path + ".bak")

    with open(wallet_path, "w") as f:
        _json.dump(list(secret_bytes), f)

    await _reply(update,
        f"✅ {mdbold('Wallet imported')}\n\n"
        f"Address: {mdcode(pubkey)}\n"
        f"Network: {mdcode(SOLANA_NETWORK.upper())}\n\n"
        + mditalic('Old wallet backed up to wallet\\.json\\.bak')
    )


# ---------------------------------------------------------------------------
# Plain-text message handler — no slash needed
# ---------------------------------------------------------------------------

def _looks_like_private_key(text: str) -> bool:
    """True if text looks like a base58 private key (87-88 chars) or 64-int JSON array."""
    t = text.strip()
    if t.startswith("["):
        try:
            import json as _j
            arr = _j.loads(t)
            return isinstance(arr, list) and len(arr) == 64
        except Exception:
            return False
    import re
    return bool(re.match(r"^[1-9A-HJ-NP-Za-km-z]{80,90}$", t))


def _looks_like_mint(text: str) -> bool:
    """True if text looks like a Solana mint address (base58, 32-44 chars, no spaces)."""
    import re
    return bool(re.match(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$", text.strip()))


async def _import_wallet_from_text(update: Update, raw: str) -> None:
    """Parse raw key text and save wallet — shared by auto-detect and pending flows."""
    import json as _j, os, shutil
    secret_bytes = None

    try:
        arr = _j.loads(raw)
        if isinstance(arr, list) and len(arr) == 64:
            secret_bytes = bytes(arr)
    except Exception:
        pass

    if secret_bytes is None:
        try:
            import base58 as _b58
            decoded = _b58.b58decode(raw)
            if len(decoded) == 64:
                secret_bytes = decoded
        except Exception:
            pass

    if secret_bytes is None:
        await _reply(update, "❌ Couldn't parse that key\. Send a base58 string or 64\-byte JSON array\.")
        return

    try:
        from solders.keypair import Keypair
        kp     = Keypair.from_bytes(secret_bytes)
        pubkey = str(kp.pubkey())
    except Exception as e:
        await _reply(update, f"❌ Invalid keypair: {mdcode(str(e))}")
        return

    wallet_path = SOLANA_WALLET_PATH
    if os.path.exists(wallet_path):
        shutil.copy(wallet_path, wallet_path + ".bak")
    with open(wallet_path, "w") as f:
        _j.dump(list(secret_bytes), f)

    await _reply(update,
        f"✅ {mdbold('Wallet imported')}\n\n"
        f"Address: {mdcode(pubkey)}\n"
        f"Network: {mdcode(SOLANA_NETWORK.upper())}\n\n"
        + mditalic("Old wallet backed up to wallet\\.json\\.bak")
    )


async def handle_plain_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Routes plain-text messages so users never need to type slash commands for input.

    Handles:
    • Pending action set by a button press (import_wallet, trade_size, watch, score_mint)
    • Auto-detection of private keys  → wallet import (private chat only)
    • Auto-detection of mint addresses → quick score / watch / blacklist menu
    """
    if not update.message or not update.message.text:
        return
    if not await _check_allowed(update):
        return

    text      = update.message.text.strip()
    is_private = update.effective_chat.type == "private"
    pending    = ctx.user_data.pop("pending", None)

    # ── Pending: wallet import ──────────────────────────────────────────────
    if pending == "import_wallet" or (pending is None and is_private and _looks_like_private_key(text)):
        if not is_private:
            await _reply(update, "⚠️ Wallet import only works in a private chat for security\.")
            return
        try:
            await update.message.delete()
        except Exception:
            pass
        await _import_wallet_from_text(update, text)
        return

    # ── Pending: custom trade size ──────────────────────────────────────────
    if pending == "trade_size":
        from .db import set_state as _ss
        try:
            val = float(text.replace(",", "."))
        except ValueError:
            await _reply(update, f"❌ That doesn't look like a number\\. Example: {mdcode('0.05')}")
            return
        if not 0.01 <= val <= 10.0:
            await _reply(update, f"❌ Must be between {mdcode('0.01')} and {mdcode('10.0')} SOL\\.")
            return
        _ss("real_position_size_sol", str(val))
        await _reply(update, f"✅ Trade size set to {mdcode(f'{val} SOL')} per trade")
        return

    # ── Pending: watch a mint ───────────────────────────────────────────────
    if pending == "watch":
        if not _looks_like_mint(text):
            await _reply(update, "❌ That doesn't look like a valid mint address\.")
            return
        ctx.args = [text]
        await cmd_watch(update, ctx)
        return

    # ── Pending: score a mint ───────────────────────────────────────────────
    if pending == "score_mint":
        if not _looks_like_mint(text):
            await _reply(update, "❌ That doesn't look like a valid mint address\.")
            return
        ctx.args = [text]
        await cmd_score(update, ctx)
        return

    # ── Auto-detect mint address → quick action buttons ─────────────────────
    if _looks_like_mint(text):
        mint = text
        kb = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("🔍 Score",    callback_data=f"qs_score_{mint}"),
                InlineKeyboardButton("👁 Watch",    callback_data=f"qs_watch_{mint}"),
                InlineKeyboardButton("🚫 Blacklist", callback_data=f"qs_bl_{mint}"),
            ]
        ])
        await _reply(update,
            f"📋 Mint: {mdcode(mint[:8])} — what would you like to do?",
            kb=kb,
        )
        return
