"""Inline button callbacks."""
import logging

from telegram import Update
from telegram.error import BadRequest, TelegramError
from telegram.ext import ContextTypes

from .config import (
    ALLOWED_CHAT_IDS, DEFAULT_THRESHOLD,
    ML_LABEL_WINDOW, PUMP_THRESHOLD_PCT, RUG_THRESHOLD_PCT,
)
from .db import set_state, upsert_chat, get_state
from .keyboards import (
    MENU_HEADER, back_keyboard, main_menu_keyboard, threshold_keyboard,
    more_keyboard, trade_size_keyboard, wallet_keyboard,
)
from .market import MarketContext
from .scoring import ScoringEngine
from .state import BotState
from .ui_text import (
    format_top_performers, query_top_performers, text_features,
    text_keywords, text_market, text_model, text_monitor_status,
    text_outcomes, text_scoring_mode, text_snapshot, text_stats, text_wallet,
    text_health, text_help, text_real_status, text_real_report,
    text_last_trade,
)
from .utils import mdbold, mdcode, mditalic, strip_md2
from .commands import do_train
from .real_trading import real_engine, SOLANA_NETWORK, REAL_POSITION_SIZE_SOL

log = logging.getLogger(__name__)
PM = "MarkdownV2"


async def _do_backtest() -> str:
    """Run backtest and return formatted text."""
    from .config import PUMP_THRESHOLD_PCT, RUG_THRESHOLD_PCT
    from .db import db_conn
    from collections import defaultdict
    from contextlib import closing
    from .utils import fmt_pct, fmt_usd, mdbold, mdcode, safe_float, safe_int

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
        return mditalic("No labeled data yet — check back later")

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
            f"Signals {mdcode(n)} | "
            f"Pump rate {mdcode(f'{pumps/n*100:.1f}%')} | "
            f"Rug rate {mdcode(f'{rugs/n*100:.1f}%')}",
            f"Avg outcome {mdcode(fmt_pct(avg_pct, 1, signed=True))}",
            "",
        ]
    return "\n".join(lines)


async def handle_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    try:
        await query.answer()
    except Exception:
        pass

    cid = update.effective_chat.id

    # Enforce the same allow-list as commands — callbacks bypass CommandHandler
    if ALLOWED_CHAT_IDS and cid not in ALLOWED_CHAT_IDS:
        try:
            await query.answer("Unauthorized", show_alert=True)
        except Exception:
            pass
        return

    data = query.data
    state:  BotState       = ctx.bot_data["state"]
    engine: ScoringEngine  = ctx.bot_data["engine"]
    mctx:   MarketContext  = ctx.bot_data["market_ctx"]

    async def show(text: str, kb=None):
        target_kb = kb or back_keyboard()
        try:
            await query.edit_message_text(text, parse_mode=PM, reply_markup=target_kb)
        except BadRequest as e:
            err = str(e).lower()
            if "not modified" in err:
                return
            if "parse" in err:
                log.warning("MD2 parse failed in callback %s: %s", data, e)
                try:
                    await query.edit_message_text(
                        strip_md2(text), reply_markup=target_kb)
                except TelegramError as e2:
                    log.error("Plain callback fallback failed: %s", e2)
                return
            log.debug("edit_message_text: %s", e)
        except TelegramError as e:
            log.debug("edit_message_text telegram: %s", e)

    try:
        if data == "menu":
            await show(MENU_HEADER, kb=main_menu_keyboard())

        elif data == "close_menu":
            try:
                await query.message.delete()
            except Exception:
                pass

        elif data == "monitor_on":
            th = state.alerts.get(cid, DEFAULT_THRESHOLD)
            state.alerts[cid] = th
            upsert_chat(cid, alerts_enabled=1, threshold=th)
            await show(f"🟢 {mdbold('Alerts ON')} — threshold {mdcode(f'{th}/10')}")

        elif data == "monitor_off":
            state.alerts.pop(cid, None)
            upsert_chat(cid, alerts_enabled=0)
            await show(f"🔴 {mdbold('Alerts OFF')}")

        elif data == "monitor_status":
            await show(text_monitor_status(cid, state))

        elif data == "threshold_menu":
            current = state.alerts.get(cid, DEFAULT_THRESHOLD)
            await show(
                f"🎚 {mdbold('Set Alert Threshold')}\n"
                f"Current: {mdcode(f'{current}/10')}\n"
                f"Choose new minimum score:",
                kb=threshold_keyboard(),
            )

        elif data.startswith("set_threshold_"):
            try:
                val = int(data.split("_")[-1])
            except ValueError:
                await show("❌ Invalid threshold")
                return
            if not 1 <= val <= 10:
                await show("❌ Must be 1–10")
                return
            state.alerts[cid] = val
            upsert_chat(cid, threshold=val, alerts_enabled=1)
            await show(
                f"✅ Threshold set to {mdcode(f'{val}/10')}\n"
                f"Alerts are {mdbold('ON')}\\."
            )

        elif data == "scoring_mode": await show(text_scoring_mode(engine))
        elif data == "features":     await show(text_features(engine))
        elif data == "keywords":     await show(text_keywords(engine))
        elif data == "market":       await show(text_market(mctx))
        elif data == "outcomes":     await show(text_outcomes())
        elif data == "model":        await show(text_model(engine))
        elif data == "snapshot":     await show(text_snapshot())
        elif data == "stats":        await show(text_stats(state, engine))
        elif data == "wallet":       await show(await text_wallet())

        elif data == "top":
            rows = query_top_performers(days=7)
            await show(format_top_performers(rows, 7))

        elif data == "train":
            try:
                await query.edit_message_text("🏋 Training\\.\\.\\.", parse_mode=PM)
            except Exception:
                pass
            await show(await do_train(engine))

        # ---------- More menu ----------
        elif data == "more":
            await show(
                f"⚙️ {mdbold('More Options')}",
                kb=more_keyboard(),
            )

        elif data == "health":      await show(text_health(state, engine))
        elif data == "help":        await show(text_help())
        elif data == "backtest":
            await show("📊 Running backtest\\.\\.\\.")
            await show(await _do_backtest())
        elif data == "last":        await show(text_last_trade())
        elif data == "real_on":
            if not real_engine.enabled:
                await real_engine.set_enabled(True)
                set_state("real_trading_enabled", "1")
            await show(f"✅ {mdbold('Real trading ON')} — network: {mdcode(SOLANA_NETWORK)}")
        elif data == "real_off":
            await real_engine.set_enabled(False)
            set_state("real_trading_enabled", "0")
            await show(f"❌ {mdbold('Real trading OFF')}")
        elif data == "real_status": await show(text_real_status(engine))
        elif data == "real_report": await show(text_real_report())
        elif data == "trade_size_menu":
            current = float(get_state("real_position_size_sol") or REAL_POSITION_SIZE_SOL)
            await show(
                f"💱 {mdbold('Trade Size')}\n"
                f"Current: {mdcode(f'{current} SOL')} per trade\n"
                f"Choose a preset or use {mdcode('/trade_size 0.3')} for custom:",
                kb=trade_size_keyboard(current),
            )

        elif data.startswith("set_size_"):
            try:
                val = float(data.split("set_size_")[1])
            except ValueError:
                await show("❌ Invalid size")
                return
            if not 0.01 <= val <= 10.0:
                await show("❌ Must be between 0.01 and 10.0 SOL")
                return
            set_state("real_position_size_sol", str(val))
            await show(
                f"✅ Trade size set to {mdcode(f'{val} SOL')} per trade",
                kb=trade_size_keyboard(val),
            )

        elif data == "size_custom_hint":
            await show(
                f"✏️ {mdbold('Custom Trade Size')}\n\n"
                f"Use the command: {mdcode('/trade_size 0.3')}\n"
                f"Min: {mdcode('0.01 SOL')} | Max: {mdcode('10.0 SOL')}",
            )

        elif data == "wallet_menu":
            await show(
                f"👛 {mdbold('Wallet Management')}\n\n"
                f"Network: {mdcode(SOLANA_NETWORK.upper())}",
                kb=wallet_keyboard(),
            )

        elif data == "wallet_address":
            from .real_trading import _load_wallet
            wallet = _load_wallet()
            address = wallet["pubkey"] if wallet else "not loaded"
            await show(
                f"📋 {mdbold('Wallet Address')}\n\n"
                f"{mdcode(address)}\n\n"
                f"Network: {mdcode(SOLANA_NETWORK.upper())}",
                kb=wallet_keyboard(),
            )

        elif data == "wallet_export_key":
            if update.effective_chat.type != "private":
                await show(
                    f"⚠️ {mdbold('Private chats only')}\n\n"
                    f"Use {mdcode('/export_wallet')} in a private chat with the bot for security\\.",
                    kb=wallet_keyboard(),
                )
            else:
                from .real_trading import _load_wallet
                import json as _json
                import base58 as _base58
                wallet = _load_wallet()
                if not wallet or wallet.get("simulated"):
                    await show("❌ No wallet loaded\\.", kb=wallet_keyboard())
                else:
                    b58_key  = _base58.b58encode(wallet["secret"]).decode()
                    key_json = _json.dumps(list(wallet["secret"]))
                    _header  = mdbold('Private Key \u2014 keep secret!')
                    _warning = mditalic('Delete this message after saving. Anyone with this key controls your funds.')
                    await show(
                        f"🔑 {_header}\n\n"
                        f"Phantom / Solflare \\(Base58\\):\n"
                        f"{mdcode(b58_key)}\n\n"
                        f"Raw JSON array \\(wallet\\.json\\):\n"
                        f"{mdcode(key_json)}\n\n"
                        f"⚠️ {_warning}",
                        kb=wallet_keyboard(),
                    )

        elif data == "wallet_import_hint":
            _warning = mditalic('Use in private chat only. Bot deletes your message immediately. Stop trading first.')
            await show(
                f"📥 {mdbold('Import Wallet')}\n\n"
                f"Send the command:\n"
                f"{mdcode('/import_wallet <private_key>')}\n\n"
                f"Accepts:\n"
                f"• Base58 encoded private key\n"
                f"• 64\\-byte JSON array\n\n"
                f"⚠️ {_warning}",
            )

        elif data == "back":
            await show(MENU_HEADER, kb=main_menu_keyboard())

        else:
            try:
                await query.answer("Unknown action", show_alert=True)
            except Exception:
                pass

    except Exception as e:
        log.error("handle_callback error for %s: %s", data, e, exc_info=True)

