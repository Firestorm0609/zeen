"""Real trading engine: devnet/mainnet SOL <> token swaps.

Architecture:
  - Network-agnostic: SOLANA_NETWORK env var switches RPC + wallet context
  - Swap execution: Jupiter API v6 + solders signing
  - DB table `real_trades` mirrors `paper_trades`
  - Network switching: set SOLANA_NETWORK="devnet" or "mainnet" in .env
  - Safety gates mirror paper trading with stricter defaults
"""
import asyncio
import base64
import json
import logging
import time
import urllib.request
from contextlib import closing
from dataclasses import dataclass, field
from typing import Optional

import aiohttp
import base58

from .config import (
    DEVNET_RPC_URL, MAINNET_RPC_URL, PUMP_FRONT,
    REAL_CONFIDENCE_GATE_STD, REAL_DAILY_LOSS_LIMIT_PCT,
    REAL_TRADING_ENABLED, REAL_MAX_CONCURRENT, REAL_MAX_POSITION_PCT,
    REAL_FEE_PCT, REAL_LOSS_STREAK_PAUSE, REAL_MINT_COOLDOWN_SEC,
    REAL_MIN_SCORE, REAL_MIN_PROB,
    REAL_POSITION_SIZE_SOL, REAL_SLIPPAGE_PCT,
    REAL_STOP_LOSS_PCT, REAL_TAKE_PROFIT_PCT,
    REAL_TIME_STOP_SEC, SOLANA_NETWORK, SOLANA_WALLET_PATH,
)
from .db import db_conn, db_write
from .enrichment import fetch_coin_mc
from .state import BotState, is_creator_blacklisted
from .trading import compute_dynamic_exit_params
from .utils import now_ts, safe_float, safe_int

log = logging.getLogger(__name__)

SOL_MINT = "So11111111111111111111111111111111111111112"


# ---------- Wallet ----------

def _load_wallet() -> Optional[dict]:
    """Load Solana keypair (64-byte JSON array or dict with 'secret' + 'pubkey')."""
    try:
        with open(SOLANA_WALLET_PATH, "r") as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) == 64:
            secret = bytes(data)
            pub = base58.b58encode(secret[32:]).decode()
            return {"secret": secret, "pubkey": pub}
        if isinstance(data, dict) and "secret" in data and "pubkey" in data:
            return data
        log.error("Invalid wallet format in %s", SOLANA_WALLET_PATH)
        return None
    except FileNotFoundError:
        log.warning("Wallet not found: %s — real swaps will be simulated", SOLANA_WALLET_PATH)
        return {"pubkey": "SIMULATED", "simulated": True}
    except Exception as e:
        log.error("Failed to load wallet %s: %s", SOLANA_WALLET_PATH, e)
        return None


def _get_rpc_url() -> str:
    return DEVNET_RPC_URL if SOLANA_NETWORK == "devnet" else MAINNET_RPC_URL


async def get_wallet_sol_balance() -> float:
    """Return SOL balance of the configured wallet."""
    wallet = _load_wallet()
    if not wallet or wallet.get("simulated"):
        return 0.0
    pubkey = wallet["pubkey"]
    url = _get_rpc_url()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json={"jsonrpc": "2.0", "id": 1,
                      "method": "getBalance", "params": [pubkey]},
                timeout=aiohttp.ClientTimeout(total=15),
                headers={"Content-Type": "application/json"},
            ) as resp:
                if resp.status != 200:
                    return 0.0
                data = await resp.json()
                if "error" in data:
                    return 0.0
                lamports = (data.get("result") or {}).get("value", 0)
                return lamports / 1_000_000_000
    except Exception as e:
        log.debug("getBalance: %s", e)
        return 0.0



# ---------- Jupiter swap ----------

async def _jupiter_quote(
    session: aiohttp.ClientSession,
    input_mint: str, output_mint: str, amount: int, slippage_bps: int,
) -> Optional[dict]:
    url = (
        f"https://api.jup.ag/swap/v1/quote"
        f"?inputMint={input_mint}&outputMint={output_mint}"
        f"&amount={amount}&slippageBps={slippage_bps}&onlyDirectRoutes=false"
    )
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                body = await resp.text()
                log.warning("Jupiter quote HTTP %s | body: %s", resp.status, body[:300])
                return None
            return await resp.json()
    except Exception as e:
        log.warning("Jupiter quote: %s", e)
        return None


async def _jupiter_swap(
    session: aiohttp.ClientSession, quote: dict, wallet_pubkey: str,
) -> Optional[dict]:
    url = "https://api.jup.ag/swap/v1/swap"
    payload = {
        "quoteResponse": quote,
        "userPublicKey": wallet_pubkey,
        "wrapUnwrapSOL": True,
    }
    try:
        async with session.post(url, json=payload,
                                timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                log.warning("Jupiter swap HTTP %s", resp.status)
                return None
            return await resp.json()
    except Exception as e:
        log.warning("Jupiter swap: %s", e)
        return None


def _sign_and_submit(base64_tx: str, secret_key: bytes) -> tuple[bool, str]:
    """Sign a base64-encoded Solana versioned transaction and submit via RPC."""
    try:
        import solders.transaction as _tx
        import solders.keypair as _kp

        tx_bytes = base64.b64decode(base64_tx)
        tx = _tx.VersionedTransaction.from_bytes(tx_bytes)
        kp = _kp.Keypair.from_bytes(secret_key)
        signed_tx = _tx.VersionedTransaction(tx.message, [kp])
        signed_bytes = bytes(signed_tx)
        encoded = base64.b64encode(signed_bytes).decode()

        url = _get_rpc_url()
        payload = {
            "jsonrpc": "2.0", "id": 1,
            "method": "sendTransaction",
            "params": [encoded, {"encoding": "base64", "skipPreflight": True, "preflightCommitment": "processed", "maxRetries": 3}],
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            if "error" in result:
                return False, str(result["error"])
            return True, result.get("result", "")
    except ImportError:
        log.error("solders not installed — cannot sign transactions")
        log.error("Install: pip install solders")
        return False, "solders not installed"
    except Exception as e:
        log.error("sign_and_submit: %s", e)
        return False, str(e)


async def execute_swap(
    session: aiohttp.ClientSession,
    input_mint: str, output_mint: str,
    amount_lamports: int, wallet: dict,
) -> tuple[bool, str, float]:
    """Execute a swap via Jupiter.

    Returns (success, tx_signature_or_error, output_amount).
    On devnet without a real wallet, runs in simulation mode.
    """
    if wallet.get("simulated"):
        log.warning("SIMULATED swap: %s -> %s amount=%s (no wallet)",
                    input_mint[:8], output_mint[:8], amount_lamports)
        return True, "simulated", amount_lamports * 0.9

    slippage_bps = int(REAL_SLIPPAGE_PCT * 100)
    quote = await _jupiter_quote(
        session, input_mint, output_mint, amount_lamports, slippage_bps)
    if not quote:
        return False, "no quote", 0.0

    swap_data = await _jupiter_swap(session, quote, wallet["pubkey"])
    if not swap_data:
        return False, "swap tx build failed", 0.0

    tx_b64 = swap_data.get("swapTransaction")
    if not tx_b64:
        log.error("Jupiter response missing swapTransaction")
        return False, "no swapTransaction", 0.0

    ok, sig_or_err = _sign_and_submit(tx_b64, wallet["secret"])
    if not ok:
        log.error("Swap signing/submission failed: %s", sig_or_err)
        return False, sig_or_err, 0.0

    out_amount = float(quote.get("outAmount", 0))
    log.info("Swap submitted | tx: %s | outAmount: %s", sig_or_err, out_amount)
    return True, sig_or_err, out_amount


async def swap_sol_for_token(
    session: aiohttp.ClientSession, mint: str, sol_lamports: int, wallet: dict,
) -> tuple[bool, str, float]:
    return await execute_swap(session, SOL_MINT, mint, sol_lamports, wallet)


async def swap_token_for_sol(
    session: aiohttp.ClientSession, mint: str, token_lamports: int, wallet: dict,
) -> tuple[bool, str, float]:
    return await execute_swap(session, mint, SOL_MINT, token_lamports, wallet)


# ---------- RealTrade dataclass ----------

@dataclass
class RealTrade:
    id: int
    mint: str
    name: str
    symbol: str
    entry_time: int
    entry_sol: float
    entry_mc: float
    position_size_sol: float
    token_amount: float = 0.0
    entry_score: int = 0
    entry_prob: float = 0.0
    highest_mc: float = 0.0
    trailing_stop_price: float = 0.0
    dynamic_sl_pct: float = 0.0
    dynamic_tp_pct: float = 0.0
    dynamic_time_stop: int = 0


# ---------- DB ----------

def init_real_trades_db() -> None:
    with closing(db_conn()) as conn, conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS real_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mint TEXT NOT NULL,
                name TEXT, symbol TEXT,
                entry_time INTEGER NOT NULL,
                entry_sol REAL NOT NULL,
                entry_mc REAL NOT NULL,
                position_size_sol REAL NOT NULL,
                token_amount REAL DEFAULT 0,
                entry_score INTEGER, entry_prob REAL,
                highest_mc REAL,
                trailing_stop_price REAL,
                dynamic_sl_pct REAL, dynamic_tp_pct REAL,
                dynamic_time_stop INTEGER,
                exit_time INTEGER, exit_mc REAL,
                pnl_sol REAL, pnl_pct REAL,
                reason TEXT, status TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_real_trades_status
                ON real_trades(status);
            CREATE INDEX IF NOT EXISTS idx_real_trades_mint
                ON real_trades(mint);
        """)


def get_open_real_trades() -> list[RealTrade]:
    with closing(db_conn()) as conn:
        rows = conn.execute(
            "SELECT * FROM real_trades WHERE status='OPEN' ORDER BY entry_time"
        ).fetchall()
    return [
        RealTrade(
            id=int(r["id"]), mint=r["mint"], name=r["name"] or "",
            symbol=r["symbol"] or "", entry_time=int(r["entry_time"]),
            entry_sol=float(r["entry_sol"]),
            entry_mc=float(r["entry_mc"]),
            position_size_sol=float(r["position_size_sol"]),
            token_amount=float(r["token_amount"] or 0),
            entry_score=safe_int(r["entry_score"]),
            entry_prob=safe_float(r["entry_prob"]),
            highest_mc=float(r["highest_mc"] or r["entry_mc"]),
            trailing_stop_price=float(r["trailing_stop_price"] or r["entry_mc"]),
            dynamic_sl_pct=safe_float(r["dynamic_sl_pct"]),
            dynamic_tp_pct=safe_float(r["dynamic_tp_pct"]),
            dynamic_time_stop=safe_int(r["dynamic_time_stop"]),
        )
        for r in rows
    ]


# ---------- RealTradingEngine ----------

class RealTradingEngine:
    def __init__(self):
        self._enabled = False
        self._lock = asyncio.Lock()

    async def set_enabled(self, val: bool) -> None:
        async with self._lock:
            self._enabled = val

    @property
    def enabled(self) -> bool:
        return self._enabled

    def check_entry(self, state: BotState, coin: dict, result: dict) -> tuple[bool, str]:
        if not self._enabled or not REAL_TRADING_ENABLED:
            return False, "real trading disabled"

        score = safe_int(result.get("score", 0))
        if score < REAL_MIN_SCORE:
            return False, f"score {score} < min {REAL_MIN_SCORE}"

        prob = safe_float(result.get("probability", 0))
        if prob < REAL_MIN_PROB:
            return False, f"prob {prob:.2f} < min {REAL_MIN_PROB}"

        cv_std = safe_float(result.get("ml_cv_auc_std", 0.0))
        if cv_std > REAL_CONFIDENCE_GATE_STD:
            return False, f"low confidence (std={cv_std:.3f})"

        mint = coin.get("mint", "")
        if not mint:
            return False, "no mint"

        mc = safe_float(coin.get("usd_market_cap"))
        if mc <= 0:
            return False, "invalid mc"

        creator = (coin.get("creator") or coin.get("user")
                   or coin.get("traderPublicKey") or "")
        if creator and is_creator_blacklisted(creator):
            return False, "creator blacklisted"

        with closing(db_conn()) as conn:
            n = conn.execute(
                "SELECT COUNT(*) AS c FROM real_trades WHERE status='OPEN'"
            ).fetchone()["c"]
        if n >= REAL_MAX_CONCURRENT:
            return False, f"max concurrent ({REAL_MAX_CONCURRENT})"

        with closing(db_conn()) as conn:
            last = conn.execute(
                "SELECT COALESCE(MAX(entry_time),0) AS t FROM real_trades WHERE mint=?",
                (mint,),
            ).fetchone()["t"]
        if last and (now_ts() - last) < REAL_MINT_COOLDOWN_SEC:
            return False, "mint cooldown"

        # --- SOL balance check (on-chain) ---
        from .db import get_state as _get_state
        _size_str = _get_state("real_position_size_sol", "")
        size_sol = float(_size_str) if _size_str else REAL_POSITION_SIZE_SOL
        # Note: async check done in open_trade before swap; here we do a quick DB-free gate
        # using the last known on-chain balance stored at last open (non-blocking best effort)

        # --- Loss streak guard ---
        with closing(db_conn()) as conn:
            streak_rows = conn.execute(
                "SELECT pnl_pct FROM real_trades WHERE status='CLOSED' "
                "ORDER BY exit_time DESC LIMIT 5"
            ).fetchall()
        streak = 0
        for r in streak_rows:
            if safe_float(r["pnl_pct"]) < 0:
                streak += 1
            else:
                break
        if streak >= REAL_LOSS_STREAK_PAUSE:
            return False, f"loss streak ({streak} consecutive losses)"

        # --- Daily loss limit (based on realised pnl_sol vs starting position size) ---
        cutoff = now_ts() - 86400
        with closing(db_conn()) as conn:
            daily_row = conn.execute(
                "SELECT COALESCE(SUM(pnl_sol), 0) AS s FROM real_trades "
                "WHERE status='CLOSED' AND exit_time >= ?",
                (cutoff,),
            ).fetchone()
        daily_sol = safe_float(daily_row["s"]) if daily_row else 0.0
        if daily_sol < 0:
            # Express loss as % of one position size as a conservative proxy
            daily_loss_pct = abs(daily_sol) / size_sol * 100 if size_sol > 0 else 0
            if daily_loss_pct >= REAL_DAILY_LOSS_LIMIT_PCT:
                return False, f"daily loss limit hit ({daily_loss_pct:.1f}%)"

        return True, "ok"

    async def open_trade(
        self, coin: dict, result: dict,
        market_ctx=None, bot=None, state=None,
    ) -> "Optional[RealTrade]":
        mint = coin.get("mint", "")
        name = coin.get("name", "")
        symbol = coin.get("symbol", "")
        mc = safe_float(coin.get("usd_market_cap"))

        sl, tp, time_sec = compute_dynamic_exit_params(coin, result, market_ctx)
        sl = max(REAL_STOP_LOSS_PCT * 0.5, min(REAL_STOP_LOSS_PCT * 1.5, sl))
        tp = max(REAL_TAKE_PROFIT_PCT * 0.5, min(REAL_TAKE_PROFIT_PCT * 2.5, tp))
        time_sec = max(300, min(4 * 3600, time_sec))

        from .db import get_state
        _size_str = get_state("real_position_size_sol", "")
        size_sol = float(_size_str) if _size_str else REAL_POSITION_SIZE_SOL
        wallet = _load_wallet()
        if not wallet:
            log.error("Cannot open real trade: wallet load failed")
            return None

        sol_lamports = int(size_sol * 1_000_000_000)

        # Check on-chain balance before attempting swap
        on_chain_bal = await get_wallet_sol_balance()
        if on_chain_bal < size_sol:
            log.error("open_trade: insufficient on-chain SOL (have %.4f, need %.4f)",
                      on_chain_bal, size_sol)
            return None

        async with aiohttp.ClientSession() as session:
            ok, msg, token_amount = await swap_sol_for_token(
                session, mint, sol_lamports, wallet)
        if not ok:
            log.error("Real swap failed: %s", msg)
            return None

        log.info(
            "REAL OPEN | %s | %.4f SOL -> tokens | SL=%.1f%% TP=%.1f%% time=%ds | on-chain=%.4f SOL",
            name or mint[:8], size_sol, sl, tp, time_sec, on_chain_bal - size_sol,
        )

        ts = now_ts()
        def _w():
            with closing(db_conn()) as conn:
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    conn.execute("""
                        INSERT INTO real_trades
                            (mint, name, symbol, entry_time, entry_sol, entry_mc,
                             position_size_sol, token_amount,
                             entry_score, entry_prob,
                             highest_mc, trailing_stop_price,
                             dynamic_sl_pct, dynamic_tp_pct, dynamic_time_stop,
                             status)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'OPEN')
                    """, (mint, name, symbol, ts, size_sol, mc,
                           size_sol, token_amount,
                           safe_int(result.get("score", 0)),
                           safe_float(result.get("probability", 0)),
                           mc, mc, sl, tp, time_sec))
                    trade_id = conn.execute(
                        "SELECT last_insert_rowid() AS id"
                    ).fetchone()["id"]
                    conn.execute("COMMIT")
                    return trade_id
                except Exception as e:
                    try:
                        conn.execute("ROLLBACK")
                    except Exception:
                        pass
                    log.error("open_trade tx failed: %s", e)
                    return None
        trade_id = db_write(_w)
        if not trade_id:
            return None

        trade = RealTrade(
            id=trade_id, mint=mint, name=name, symbol=symbol,
            entry_time=ts, entry_sol=size_sol, entry_mc=mc,
            position_size_sol=size_sol,
            token_amount=token_amount,   # raw lamports as returned by Jupiter outAmount
            entry_score=safe_int(result.get("score", 0)),
            entry_prob=safe_float(result.get("probability", 0)),
            highest_mc=mc, trailing_stop_price=mc,
            dynamic_sl_pct=sl, dynamic_tp_pct=tp,
            dynamic_time_stop=time_sec,
        )

        if bot and state:
            try:
                from telegram.helpers import escape_markdown
                text = (
                    f"⚡ *REAL TRADE OPENED* — {SOLANA_NETWORK.upper()}\n"
                    f"🪙 {escape_markdown(name or mint[:8], version=2)}\n"
                    f"Size: `{size_sol} SOL`\n"
                    f"SL: `{sl:.1f}%` \\| TP: `{tp:.1f}%` \\| Time: `{time_sec}s`"
                )
                for cid in list(state.alerts.keys()):
                    try:
                        await bot.send_message(
                            chat_id=cid, text=text, parse_mode="MarkdownV2")
                    except Exception:
                        pass
            except Exception as e:
                log.debug("real trade open notify: %s", e)

        return trade


# ---------- Global ----------

real_engine = RealTradingEngine()


async def maybe_open_real_trade(
    state: "BotState", coin: dict, result: dict,
    market_ctx=None, bot=None,
) -> None:
    ok, reason = real_engine.check_entry(state, coin, result)
    if ok:
        await real_engine.open_trade(coin, result, market_ctx, bot, state)
    else:
        log.debug("REAL SKIP | %s | %s", (coin.get("mint") or "?")[:8], reason)


async def real_monitor_loop(bot=None) -> None:
    """Monitor open real trades: check prices + execute exits."""
    while True:
        try:
            if not real_engine.enabled:
                await asyncio.sleep(60)
                continue

            trades = get_open_real_trades()
            if not trades:
                await asyncio.sleep(60)
                continue

            async with aiohttp.ClientSession() as session:
                for t in trades:
                    mc = await fetch_coin_mc(session, t.mint)
                    if mc is None:
                        continue

                    # Trailing stop update
                    if mc > t.highest_mc:
                        t.highest_mc = mc
                        sl_pct = t.dynamic_sl_pct or REAL_STOP_LOSS_PCT
                        t.trailing_stop_price = mc * (1.0 - sl_pct / 100.0)
                        def _w(tid=t.id, hmc=t.highest_mc, tsp=t.trailing_stop_price):
                            with closing(db_conn()) as conn, conn:
                                conn.execute(
                                    "UPDATE real_trades SET highest_mc=?, trailing_stop_price=? "
                                    "WHERE id=? AND status='OPEN'",
                                    (hmc, tsp, tid))
                        db_write(_w)

                    # Exit checks
                    sl_pct = t.dynamic_sl_pct or REAL_STOP_LOSS_PCT
                    tp_pct = t.dynamic_tp_pct or REAL_TAKE_PROFIT_PCT
                    time_stop = t.dynamic_time_stop or REAL_TIME_STOP_SEC
                    age = now_ts() - t.entry_time

                    should_close = False
                    reason = ""
                    if mc <= t.entry_mc * (1.0 - sl_pct / 100.0):
                        should_close = True
                        reason = f"STOP_LOSS_{sl_pct:.1f}%"
                    elif mc >= t.entry_mc * (1.0 + tp_pct / 100.0):
                        should_close = True
                        reason = f"TAKE_PROFIT_{tp_pct:.1f}%"
                    elif t.highest_mc > t.entry_mc and t.trailing_stop_price > 0:
                        if mc <= t.trailing_stop_price:
                            trail_pnl = ((t.trailing_stop_price - t.entry_mc)
                                        / t.entry_mc) * 100.0
                            should_close = True
                            reason = f"TRAILING_STOP_{trail_pnl:+.1f}%"
                    elif age >= time_stop:
                        should_close = True
                        reason = f"TIME_STOP_{time_stop}s"

                    if should_close:
                        # Execute sell: token -> SOL
                        wallet = _load_wallet()
                        sol_received_actual = t.position_size_sol  # fallback: assume break-even
                        if wallet and not wallet.get("simulated") and t.token_amount > 0:
                            # token_amount is already stored as raw lamports (Jupiter outAmount)
                            raw_amount = int(t.token_amount)
                            ok, msg, sol_received = await swap_token_for_sol(
                                session, t.mint, raw_amount, wallet)
                            if ok:
                                sol_received_actual = sol_received / 1_000_000_000
                                exit_mc = sol_received_actual / (t.position_size_sol or 1) * t.entry_mc
                            else:
                                exit_mc = mc
                        else:
                            exit_mc = mc
                            # simulated: estimate proceeds from MC-based pnl
                            pnl_est = ((exit_mc - t.entry_mc) / t.entry_mc
                                       if t.entry_mc > 0 else 0.0)
                            sol_received_actual = t.position_size_sol * (1 + pnl_est)

                        pnl_pct = ((exit_mc - t.entry_mc) / t.entry_mc * 100
                                   if t.entry_mc > 0 else 0)
                        pnl_sol = t.position_size_sol * (pnl_pct / 100.0)

                        def _close(tid=t.id, tmc=exit_mc, tp=pnl_pct, tsol=pnl_sol,
                                   r=reason):
                            with closing(db_conn()) as conn:
                                try:
                                    conn.execute("BEGIN IMMEDIATE")
                                    row = conn.execute(
                                        "SELECT status FROM real_trades WHERE id=?", (tid,)
                                    ).fetchone()
                                    if not row or row["status"] != "OPEN":
                                        conn.execute("ROLLBACK")
                                        return False
                                    conn.execute("""
                                        UPDATE real_trades
                                        SET exit_time=?, exit_mc=?, pnl_sol=?, pnl_pct=?,
                                            reason=?, status='CLOSED'
                                        WHERE id=? AND status='OPEN'
                                    """, (now_ts(), tmc, tsol, tp, r, tid))
                                    conn.execute("COMMIT")
                                    return True
                                except Exception as e:
                                    try:
                                        conn.execute("ROLLBACK")
                                    except Exception:
                                        pass
                                    log.error("_close tx failed: %s", e)
                                    return False
                        db_write(_close)

                        log.info("REAL CLOSE | %s | %s | %+.1f%% (%+.3f SOL)",
                                 t.name or t.mint[:8], reason, pnl_pct, pnl_sol)

                        if bot:
                            try:
                                from .state import BotState  # avoid circular at module level
                                from telegram.helpers import escape_markdown
                                emoji = "🟢" if pnl_pct > 0 else "🔴"
                                text = (
                                    f"{emoji} *TRADE CLOSED* — {SOLANA_NETWORK.upper()}\n"
                                    f"🪙 {escape_markdown(t.name or t.mint[:8], version=2)}\n"
                                    f"Reason: `{escape_markdown(reason, version=2)}`\n"
                                    f"P&L: `{pnl_pct:+.1f}%` \\(`{pnl_sol:+.4f} SOL`\\)"
                                )
                                # Pull alert chat IDs from app bot_data if available
                                app = getattr(bot, "_application", None)
                                alert_chats: list[int] = []
                                if app and hasattr(app, "bot_data"):
                                    st = app.bot_data.get("state")
                                    if st:
                                        alert_chats = list(st.alerts.keys())
                                for cid in alert_chats:
                                    try:
                                        await bot.send_message(
                                            chat_id=cid, text=text,
                                            parse_mode="MarkdownV2")
                                    except Exception:
                                        pass
                            except Exception as e:
                                log.debug("real close notify: %s", e)

        except Exception as e:
            log.error("real_monitor_loop: %s", e)
        await asyncio.sleep(60)


def real_stats() -> dict:
    with closing(db_conn()) as conn:
        closed = conn.execute(
            "SELECT pnl_pct, pnl_sol FROM real_trades WHERE status='CLOSED' "
            "ORDER BY exit_time DESC LIMIT 1000"
        ).fetchall()
        open_n = conn.execute(
            "SELECT COUNT(*) AS c FROM real_trades WHERE status='OPEN'"
        ).fetchone()["c"]

    n = len(closed)
    wins   = sum(1 for r in closed if safe_float(r["pnl_pct"]) > 0)
    losses = n - wins
    total_sol = sum(safe_float(r["pnl_sol"]) for r in closed)
    avg_pct   = sum(safe_float(r["pnl_pct"]) for r in closed) / n if n else 0.0
    best      = max((safe_float(r["pnl_pct"]) for r in closed), default=0.0)
    worst     = min((safe_float(r["pnl_pct"]) for r in closed), default=0.0)

    # Max drawdown (cumulative SOL equity curve)
    equity = peak = max_dd = 0.0
    for r in reversed(closed):
        equity += safe_float(r["pnl_sol"])
        peak    = max(peak, equity)
        max_dd  = max(max_dd, peak - equity)

    return {
        "enabled":          real_engine.enabled,
        "network":          SOLANA_NETWORK,
        "open_positions":   int(open_n),
        "closed_positions": n,
        "wins":             wins,
        "losses":           losses,
        "win_rate":         wins / n * 100 if n else 0.0,
        "total_pnl_sol":    total_sol,
        "avg_pnl_pct":      avg_pct,
        "best_pnl_pct":     best,
        "worst_pnl_pct":    worst,
        "max_drawdown_sol": max_dd,
    }
