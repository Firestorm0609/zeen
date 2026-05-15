"""Real trading engine: devnet/mainnet SOL <> token swaps.

Architecture:
  - Network-agnostic: SOLANA_NETWORK env var switches RPC + wallet context
  - Swap execution: Jupiter API v6 + solders signing
  - Network switching: set SOLANA_NETWORK="devnet" or "mainnet" in .env
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
    REAL_PRIORITY_FEE_LAMPORTS, REAL_MONITOR_INTERVAL_SEC,
    REAL_DAILY_SPEND_CAP_SOL, MAINNET_CONFIRMED, REAL_MAX_EXIT_RETRIES,
    REAL_TX_CONFIRM_TIMEOUT, REAL_TX_CONFIRM_INTERVAL,
    REAL_MIN_SCORE, REAL_MIN_PROB, REAL_MIN_MCAP, REAL_MAX_MCAP, REAL_MIN_TP_PROFIT_PCT,
    REAL_POSITION_SIZE_SOL,
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

# Rate limiter: prevents simultaneous Jupiter API calls causing 429s
_jupiter_lock = asyncio.Lock()
_JUPITER_CALL_DELAY = 0.25  # seconds between Jupiter requests


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
            # Ensure secret is bytes regardless of storage format
            secret = data["secret"]
            if isinstance(secret, list):
                secret = bytes(secret)
            elif isinstance(secret, str):
                import base58 as _b58
                secret = _b58.b58decode(secret)
            return {"secret": secret, "pubkey": data["pubkey"]}
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


async def _get_sol_balance_async(pubkey: str) -> float:
    """Async SOL balance check — does not block the event loop."""
    url = _get_rpc_url()
    payload = {"jsonrpc": "2.0", "id": 1, "method": "getBalance", "params": [pubkey]}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload,
                timeout=aiohttp.ClientTimeout(total=5),
                headers={"Content-Type": "application/json"},
            ) as resp:
                if resp.status != 200:
                    return 0.0
                result = await resp.json()
                if "error" in result:
                    return 0.0
                return (result.get("result") or {}).get("value", 0) / 1_000_000_000
    except Exception as e:
        log.debug("_get_sol_balance_async: %s", e)
        return 0.0


async def fetch_token_decimals(mint: str) -> int:
    """Fetch SPL token decimal places from RPC. Defaults to 6 on failure."""
    url = _get_rpc_url()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json={"jsonrpc": "2.0", "id": 1, "method": "getAccountInfo",
                      "params": [mint, {"encoding": "jsonParsed"}]},
                timeout=aiohttp.ClientTimeout(total=10),
                headers={"Content-Type": "application/json"},
            ) as resp:
                if resp.status != 200:
                    return 6
                data = await resp.json()
                decimals = (
                    data.get("result", {})
                        .get("value", {})
                        .get("data", {})
                        .get("parsed", {})
                        .get("info", {})
                        .get("decimals", 6)
                )
                return int(decimals)
    except Exception as e:
        log.debug("fetch_token_decimals %s: %s", mint[:8], e)
        return 6


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

async def _get_token_balance_async(
    session: aiohttp.ClientSession, pubkey: str, mint: str
) -> int:
    """Fetch actual on-chain SPL token balance in raw units. Returns 0 on failure."""
    url = _get_rpc_url()
    try:
        async with session.post(
            url,
            json={"jsonrpc": "2.0", "id": 1,
                  "method": "getTokenAccountsByOwner",
                  "params": [pubkey, {"mint": mint},
                             {"encoding": "jsonParsed"}]},
            timeout=aiohttp.ClientTimeout(total=5),
            headers={"Content-Type": "application/json"},
        ) as resp:
            if resp.status != 200:
                return 0
            data = await resp.json()
            accounts = (data.get("result") or {}).get("value", [])
            if not accounts:
                return 0
            amount = (
                accounts[0]
                .get("account", {})
                .get("data", {})
                .get("parsed", {})
                .get("info", {})
                .get("tokenAmount", {})
                .get("amount", "0")
            )
            return int(amount)
    except Exception as e:
        log.debug("_get_token_balance_async %s: %s", mint[:8], e)
        return 0


async def _jupiter_quote(
    session: aiohttp.ClientSession,
    input_mint: str, output_mint: str, amount: int,
    slippage_bps: int = 2000,
) -> Optional[dict]:
    # autoSlippage is intentionally omitted here: the swap body uses
    # dynamicSlippage + maxAutoSlippageBps, which overrides the quote's
    # slippage anyway.  Specifying both causes them to fight each other
    # and can produce a tighter tolerance than intended, triggering
    # on-chain error 6001 (SlippageToleranceExceeded).
    url = (
        f"https://api.jup.ag/swap/v1/quote"
        f"?inputMint={input_mint}&outputMint={output_mint}"
        f"&amount={amount}&slippageBps={slippage_bps}&onlyDirectRoutes=false"
    )
    async with _jupiter_lock:
        await asyncio.sleep(_JUPITER_CALL_DELAY)
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
        "dynamicSlippage": True,
        # Give dynamic slippage enough headroom for thin-liquidity pump.fun tokens.
        # Without this ceiling Jupiter may compute a tolerance below what micro-cap
        # price movement requires, causing on-chain error 6001.
        "maxAutoSlippageBps": 2000,   # 20 % ceiling
        "prioritizationFeeLamports": REAL_PRIORITY_FEE_LAMPORTS,
    }
    async with _jupiter_lock:
        await asyncio.sleep(_JUPITER_CALL_DELAY)
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


def _sign_tx(base64_tx: str, secret_key: bytes) -> tuple[bool, str]:
    """Sign a base64-encoded Solana versioned transaction. Returns (ok, signed_b64_or_error)."""
    try:
        import solders.transaction as _tx
        import solders.keypair as _kp
        tx_bytes = base64.b64decode(base64_tx)
        tx = _tx.VersionedTransaction.from_bytes(tx_bytes)
        kp = _kp.Keypair.from_bytes(secret_key)
        signed_tx = _tx.VersionedTransaction(tx.message, [kp])
        return True, base64.b64encode(bytes(signed_tx)).decode()
    except ImportError:
        log.error("solders not installed — pip install solders")
        return False, "solders not installed"
    except Exception as e:
        log.error("_sign_tx: %s", e)
        return False, str(e)


async def _submit_and_confirm(
    session: aiohttp.ClientSession,
    signed_b64: str,
) -> tuple[bool, str]:
    """Submit a signed transaction and poll until confirmed or timeout.

    Returns (confirmed, signature_or_error).
    """
    url = _get_rpc_url()
    send_payload = {
        "jsonrpc": "2.0", "id": 1,
        "method": "sendTransaction",
        "params": [signed_b64, {
            "encoding": "base64",
            "skipPreflight": True,
            "preflightCommitment": "processed",
            "maxRetries": 3,
        }],
    }
    try:
        async with session.post(
            url, json=send_payload,
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"Content-Type": "application/json"},
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                return False, f"sendTransaction HTTP {resp.status}: {body[:200]}"
            send_result = await resp.json()

        if "error" in send_result:
            return False, str(send_result["error"])
        sig = send_result.get("result", "")
        if not sig:
            return False, "no signature in sendTransaction response"

        log.info("TX submitted: %s — polling for confirmation", sig)

        # ── Poll getSignatureStatuses ────────────────────────────────
        deadline = time.monotonic() + REAL_TX_CONFIRM_TIMEOUT
        confirm_payload = {
            "jsonrpc": "2.0", "id": 1,
            "method": "getSignatureStatuses",
            "params": [[sig], {"searchTransactionHistory": True}],
        }
        while time.monotonic() < deadline:
            await asyncio.sleep(REAL_TX_CONFIRM_INTERVAL)
            try:
                async with session.post(
                    url, json=confirm_payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                    headers={"Content-Type": "application/json"},
                ) as cresp:
                    if cresp.status != 200:
                        continue
                    cdata = await cresp.json()

                statuses = (
                    (cdata.get("result") or {})
                    .get("value") or [None]
                )
                status = statuses[0] if statuses else None
                if status is None:
                    log.debug("TX %s: not yet seen", sig[:16])
                    continue

                if status.get("err"):
                    err = status["err"]
                    log.error("TX %s FAILED on-chain: %s", sig[:16], err)
                    return False, f"on-chain error: {err}"

                conf = status.get("confirmationStatus", "")
                log.debug("TX %s: %s", sig[:16], conf)
                if conf in ("confirmed", "finalized"):
                    log.info("TX %s confirmed (%s)", sig[:16], conf)
                    return True, sig

            except Exception as poll_err:
                log.debug("TX poll error: %s", poll_err)

        log.error("TX %s: confirmation timeout after %ds", sig[:16], REAL_TX_CONFIRM_TIMEOUT)
        return False, f"confirmation timeout ({REAL_TX_CONFIRM_TIMEOUT}s): {sig}"

    except Exception as e:
        log.error("_submit_and_confirm: %s", e)
        return False, str(e)


async def execute_swap(
    session: aiohttp.ClientSession,
    input_mint: str, output_mint: str,
    amount_lamports: int, wallet: dict,
) -> tuple[bool, str, float]:
    """Execute a swap via Jupiter.

    Returns (success, tx_signature_or_error, output_amount).
    On devnet without a real wallet, runs in simulation mode.

    Automatically retries once with higher slippage on on-chain error 6001
    (SlippageToleranceExceeded), which is common on low-liquidity pump.fun tokens.
    """
    if wallet.get("simulated"):
        log.warning("SIMULATED swap: %s -> %s amount=%s (no wallet)",
                    input_mint[:8], output_mint[:8], amount_lamports)
        return True, "simulated", amount_lamports * 0.9

    # Attempt 1: normal slippage (2000 bps = 20 %)
    # Attempt 2: elevated slippage (3000 bps = 30 %) on 6001 retry
    for attempt, slippage_bps in enumerate([2000, 3000], start=1):
        if attempt > 1:
            log.warning("Retrying swap with elevated slippage (%d bps) after 6001", slippage_bps)

        quote = await _jupiter_quote(
            session, input_mint, output_mint, amount_lamports,
            slippage_bps=slippage_bps)
        if not quote:
            return False, "no quote", 0.0

        swap_data = await _jupiter_swap(session, quote, wallet["pubkey"])
        if not swap_data:
            return False, "swap tx build failed", 0.0

        tx_b64 = swap_data.get("swapTransaction")
        if not tx_b64:
            log.error("Jupiter response missing swapTransaction")
            return False, "no swapTransaction", 0.0

        ok, signed_b64 = _sign_tx(tx_b64, wallet["secret"])
        if not ok:
            log.error("TX signing failed: %s", signed_b64)
            return False, signed_b64, 0.0

        confirmed, sig_or_err = await _submit_and_confirm(session, signed_b64)
        if confirmed:
            out_amount = float(quote.get("outAmount", 0))
            log.info("Swap confirmed | tx: %s | outAmount: %s", sig_or_err, out_amount)
            return True, sig_or_err, out_amount

        # On 6001 (SlippageToleranceExceeded) retry with wider slippage
        if attempt == 1 and "6001" in str(sig_or_err):
            log.warning("Swap hit SlippageToleranceExceeded (6001) — will retry")
            continue

        log.error("TX not confirmed: %s", sig_or_err)
        return False, sig_or_err, 0.0

    # Should not reach here, but be safe
    return False, "swap failed after slippage retry", 0.0


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
    token_decimals: int = 6
    tx_signature: str = ""
    entry_score: int = 0
    entry_prob: float = 0.0
    highest_mc: float = 0.0
    trailing_stop_price: float = 0.0
    dynamic_sl_pct: float = 0.0
    dynamic_tp_pct: float = 0.0
    dynamic_time_stop: int = 0
    failed_exit_attempts: int = 0


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
                token_decimals INTEGER DEFAULT 6,
                entry_score INTEGER, entry_prob REAL,
                highest_mc REAL,
                trailing_stop_price REAL,
                dynamic_sl_pct REAL, dynamic_tp_pct REAL,
                dynamic_time_stop INTEGER,
                tx_signature TEXT,
                exit_tx_signature TEXT,
                exit_time INTEGER, exit_mc REAL,
                failed_exit_attempts INTEGER DEFAULT 0,
                exit_error TEXT,
                pnl_sol REAL, pnl_pct REAL,
                reason TEXT, status TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_real_trades_status
                ON real_trades(status);
            CREATE INDEX IF NOT EXISTS idx_real_trades_mint
                ON real_trades(mint);
        """)
        # Migrate existing DBs — safe to run repeatedly
        for col, definition in [
            ("token_decimals", "INTEGER DEFAULT 6"),
            ("failed_exit_attempts", "INTEGER DEFAULT 0"),
            ("exit_error", "TEXT"),
            ("tx_signature", "TEXT"),
            ("exit_tx_signature", "TEXT"),
        ]:
            try:
                conn.execute(f"ALTER TABLE real_trades ADD COLUMN {col} {definition}")
            except Exception:
                pass  # column already exists


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
            token_decimals=int(r["token_decimals"] or 6),
            tx_signature=r["tx_signature"] or "" if "tx_signature" in r.keys() else "",
            entry_score=safe_int(r["entry_score"]),
            entry_prob=safe_float(r["entry_prob"]),
            highest_mc=float(r["highest_mc"] or r["entry_mc"]),
            trailing_stop_price=float(r["trailing_stop_price"] or r["entry_mc"]),
            dynamic_sl_pct=safe_float(r["dynamic_sl_pct"]),
            dynamic_tp_pct=safe_float(r["dynamic_tp_pct"]),
            dynamic_time_stop=safe_int(r["dynamic_time_stop"]),
            failed_exit_attempts=int(r["failed_exit_attempts"] or 0),
        )
        for r in rows
    ]


def get_failed_exit_trades() -> list[RealTrade]:
    """Return FAILED_EXIT trades that still have retries remaining."""
    with closing(db_conn()) as conn:
        rows = conn.execute(
            "SELECT * FROM real_trades "
            "WHERE status='FAILED_EXIT' "
            "AND COALESCE(failed_exit_attempts, 0) < ? "
            "ORDER BY entry_time",
            (REAL_MAX_EXIT_RETRIES,),
        ).fetchall()
    return [
        RealTrade(
            id=int(r["id"]), mint=r["mint"], name=r["name"] or "",
            symbol=r["symbol"] or "", entry_time=int(r["entry_time"]),
            entry_sol=float(r["entry_sol"]),
            entry_mc=float(r["entry_mc"]),
            position_size_sol=float(r["position_size_sol"]),
            token_amount=float(r["token_amount"] or 0),
            token_decimals=int(r["token_decimals"] or 6),
            tx_signature=r["tx_signature"] or "" if "tx_signature" in r.keys() else "",
            entry_score=safe_int(r["entry_score"]),
            entry_prob=safe_float(r["entry_prob"]),
            highest_mc=float(r["highest_mc"] or r["entry_mc"]),
            trailing_stop_price=float(r["trailing_stop_price"] or r["entry_mc"]),
            dynamic_sl_pct=safe_float(r["dynamic_sl_pct"]),
            dynamic_tp_pct=safe_float(r["dynamic_tp_pct"]),
            dynamic_time_stop=safe_int(r["dynamic_time_stop"]),
            failed_exit_attempts=int(r["failed_exit_attempts"] or 0),
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

    async def check_entry(self, state: BotState, coin: dict, result: dict) -> tuple[bool, str]:
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

        if REAL_MIN_MCAP > 0 and mc < REAL_MIN_MCAP:
            return False, f"mcap ${mc:,.0f} below min ${REAL_MIN_MCAP:,.0f}"
        if REAL_MAX_MCAP > 0 and mc > REAL_MAX_MCAP:
            return False, f"mcap ${mc:,.0f} above max ${REAL_MAX_MCAP:,.0f}"

        creator = (coin.get("creator") or coin.get("user")
                   or coin.get("traderPublicKey") or "")
        if creator and is_creator_blacklisted(creator):
            return False, "creator blacklisted"

        from .db import get_state as _get_state
        _size_str = _get_state("real_position_size_sol", "")
        size_sol = float(_size_str) if _size_str else REAL_POSITION_SIZE_SOL
        cutoff = now_ts() - 86400

        # --- Single batched DB check ---
        with closing(db_conn()) as conn:
            n = conn.execute(
                "SELECT COUNT(*) AS c FROM real_trades WHERE status='OPEN'"
            ).fetchone()["c"]
            last = conn.execute(
                "SELECT COALESCE(MAX(entry_time),0) AS t FROM real_trades WHERE mint=?",
                (mint,),
            ).fetchone()["t"]
            spend_row = conn.execute(
                "SELECT COALESCE(SUM(position_size_sol), 0) AS s FROM real_trades "
                "WHERE entry_time >= ?", (cutoff,),
            ).fetchone()
            streak_rows = conn.execute(
                "SELECT pnl_pct FROM real_trades WHERE status IN ('CLOSED','ABANDONED') "
                "ORDER BY exit_time DESC LIMIT 5"
            ).fetchall()
            daily_row = conn.execute(
                "SELECT COALESCE(SUM(pnl_sol), 0) AS s FROM real_trades "
                "WHERE status IN ('CLOSED','ABANDONED') AND exit_time >= ?", (cutoff,),
            ).fetchone()

        if n >= REAL_MAX_CONCURRENT:
            return False, f"max concurrent ({REAL_MAX_CONCURRENT})"

        if last and (now_ts() - last) < REAL_MINT_COOLDOWN_SEC:
            return False, "mint cooldown"

        daily_spent = float(spend_row["s"]) if spend_row else 0.0
        if daily_spent + size_sol > REAL_DAILY_SPEND_CAP_SOL:
            return False, f"daily spend cap hit ({daily_spent:.3f}/{REAL_DAILY_SPEND_CAP_SOL} SOL)"

        # --- REAL_MAX_POSITION_PCT: disabled (position size managed via REAL_POSITION_SIZE_SOL) ---

        streak = 0
        for r in streak_rows:
            if safe_float(r["pnl_pct"]) < 0:
                streak += 1
            else:
                break
        if streak >= REAL_LOSS_STREAK_PAUSE:
            return False, f"loss streak ({streak} consecutive losses)"

        daily_sol = safe_float(daily_row["s"]) if daily_row else 0.0
        if daily_sol < 0:
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
        tp = max(REAL_TAKE_PROFIT_PCT * 0.75, min(REAL_TAKE_PROFIT_PCT * 2.5, tp))  # floor: 60% at base 80
        time_sec = max(300, min(4 * 3600, time_sec))

        from .db import get_state
        _size_str = get_state("real_position_size_sol", "")
        size_sol = float(_size_str) if _size_str else REAL_POSITION_SIZE_SOL
        # Mainnet safety guard — requires MAINNET_CONFIRMED=true in .env
        if SOLANA_NETWORK == "mainnet" and not MAINNET_CONFIRMED:
            log.error("Mainnet guard: set MAINNET_CONFIRMED=true in .env to enable live trading")
            return None

        wallet = _load_wallet()
        if not wallet:
            log.error("Cannot open real trade: wallet load failed")
            return None

        sol_lamports = int(size_sol * 1_000_000_000)

        # Check on-chain balance before attempting swap (skip for simulated/devnet)
        if not wallet.get("simulated"):
            on_chain_bal = await get_wallet_sol_balance()
            if on_chain_bal < size_sol:
                log.error("open_trade: insufficient on-chain SOL (have %.4f, need %.4f)",
                          on_chain_bal, size_sol)
                return None
        else:
            on_chain_bal = size_sol  # devnet simulated: balance is irrelevant

        async with aiohttp.ClientSession() as session:
            ok, tx_sig, token_amount = await swap_sol_for_token(
                session, mint, sol_lamports, wallet)
            token_decimals = await fetch_token_decimals(mint) if ok else 6
        msg = tx_sig  # preserve for error reporting
        if ok:
            # Fetch actual on-chain MC at swap confirmation time for accurate entry_mc
            async with aiohttp.ClientSession() as _mc_session:
                confirmed_mc = await fetch_coin_mc(_mc_session, mint)
            if confirmed_mc and confirmed_mc > 0:
                log.info("Entry MC updated: signal=$%,.0f confirmed=$%,.0f | %s",
                         mc, confirmed_mc, mint[:8])
                mc = confirmed_mc
        if not ok:
            log.error("Real swap failed: %s", msg)
            if bot and state:
                try:
                    from telegram.helpers import escape_markdown
                    _etxt = (
                        f"⚠️ *ENTRY SWAP FAILED*\n"
                        f"🪙 {escape_markdown(name or mint[:8], version=2)}\n"
                        f"Error: `{escape_markdown(str(msg)[:100], version=2)}`"
                    )
                    for _cid in list(state.alerts.keys()):
                        try:
                            await bot.send_message(chat_id=_cid, text=_etxt, parse_mode="MarkdownV2")
                        except Exception:
                            pass
                except Exception as _ne:
                    log.debug("entry swap fail notify: %s", _ne)
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
                             position_size_sol, token_amount, token_decimals,
                             tx_signature, entry_score, entry_prob,
                             highest_mc, trailing_stop_price,
                             dynamic_sl_pct, dynamic_tp_pct, dynamic_time_stop,
                             status)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'OPEN')
                    """, (mint, name, symbol, ts, size_sol, mc,
                           size_sol, token_amount, token_decimals,
                           tx_sig,
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
            token_amount=token_amount,
            token_decimals=token_decimals,
            tx_signature=tx_sig,
            entry_score=safe_int(result.get("score", 0)),
            entry_prob=safe_float(result.get("probability", 0)),
            highest_mc=mc, trailing_stop_price=mc,
            dynamic_sl_pct=sl, dynamic_tp_pct=tp,
            dynamic_time_stop=time_sec,
        )

        if bot and state:
            try:
                from telegram.helpers import escape_markdown
                score  = safe_float(result.get("score", 0))
                prob   = safe_float(result.get("probability", 0))
                text = (
                    f"⚡ *REAL TRADE OPENED* — {SOLANA_NETWORK.upper()}\n"
                    f"🪙 {escape_markdown(name or mint[:8], version=2)}\n"
                    f"💰 Entry MC: `${mc:,.0f}`\n"
                    f"⭐ Score: `{int(score)}/10` \\| ML Prob: `{prob:.0%}`\n"
                    f"📦 Size: `{size_sol} SOL`\n"
                    f"🛑 SL: `{sl:.1f}%` \\| 🎯 TP: `{tp:.1f}%` \\| ⏱ Time: `{time_sec}s`"
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
    ok, reason = await real_engine.check_entry(state, coin, result)
    if ok:
        await real_engine.open_trade(coin, result, market_ctx, bot, state)
    else:
        log.info("REAL SKIP | %s | %s", (coin.get("mint") or "?")[:8], reason)


async def real_monitor_loop(bot=None, state=None) -> None:
    """Monitor open real trades: check prices + execute exits."""
    _notify_state = state
    while True:
        try:
            if not real_engine.enabled:
                await asyncio.sleep(REAL_MONITOR_INTERVAL_SEC)
                continue

            trades = get_open_real_trades()
            if not trades:
                await asyncio.sleep(REAL_MONITOR_INTERVAL_SEC)
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
                    elif t.highest_mc > t.entry_mc and t.trailing_stop_price > 0 and mc <= t.trailing_stop_price:
                        trail_pnl = ((t.trailing_stop_price - t.entry_mc)
                                    / t.entry_mc) * 100.0
                        should_close = True
                        reason = f"TRAILING_STOP_{trail_pnl:+.1f}%"
                    elif mc >= t.entry_mc * (1.0 + tp_pct / 100.0):
                        should_close = True
                        reason = f"TAKE_PROFIT_{tp_pct:.1f}%"
                    elif age >= time_stop:
                        should_close = True
                        reason = f"TIME_STOP_{time_stop}s"

                    if should_close:
                        # Execute sell: token -> SOL
                        wallet = _load_wallet()
                        sol_received_actual = t.position_size_sol  # fallback: assume break-even
                        # Initialise so the _close closure can always reference them safely,
                        # even when the simulated/zero-token branch is taken and
                        # swap_token_for_sol is never called (prevents NameError).
                        ok = False
                        exit_sig = ""
                        if wallet and not wallet.get("simulated") and t.token_amount > 0:
                            # Fetch actual on-chain balance to sell everything, avoiding dust
                            raw_amount = await _get_token_balance_async(
                                session, wallet["pubkey"], t.mint)
                            if raw_amount == 0:
                                raw_amount = int(round(t.token_amount))
                                log.debug("on-chain balance fetch failed, using stored amount")

                            # For TP exits: check quote first — if price has already dumped
                            # below minimum profit threshold, skip this cycle and wait
                            if "TAKE_PROFIT" in reason and raw_amount > 0:
                                _pre_quote = await _jupiter_quote(
                                    session, t.mint, SOL_MINT, raw_amount)
                                if _pre_quote:
                                    _expected_sol = int(_pre_quote.get("outAmount", 0)) / 1_000_000_000
                                    _quote_pnl = (_expected_sol - t.position_size_sol) / (t.position_size_sol or 1) * 100
                                    if _quote_pnl < REAL_MIN_TP_PROFIT_PCT:
                                        log.info(
                                            "TP quote check: %.1f%% profit < min %.1f%% — price dumped, skipping cycle | %s",
                                            _quote_pnl, REAL_MIN_TP_PROFIT_PCT, t.mint[:8])
                                        should_close = False
                                        continue

                            ok, exit_sig, sol_received = await swap_token_for_sol(
                                session, t.mint, raw_amount, wallet)
                            if ok:
                                sol_received_actual = sol_received / 1_000_000_000
                                # Fetch actual on-chain MC at exit confirmation time
                                confirmed_exit_mc = await fetch_coin_mc(session, t.mint)
                                exit_mc = confirmed_exit_mc if confirmed_exit_mc and confirmed_exit_mc > 0                                     else sol_received_actual / (t.position_size_sol or 1) * t.entry_mc
                            else:
                                log.error("Exit swap FAILED for %s: %s — marking FAILED_EXIT", t.mint[:8], exit_sig)
                                def _fail_exit(tid=t.id, emsg=str(exit_sig)):
                                    with closing(db_conn()) as conn, conn:
                                        conn.execute(
                                            "UPDATE real_trades SET "
                                            "failed_exit_attempts=COALESCE(failed_exit_attempts,0)+1,"
                                            "exit_error=?, status='FAILED_EXIT' "
                                            "WHERE id=? AND status='OPEN'",
                                            (emsg[:500], tid))
                                db_write(_fail_exit)
                                if bot:
                                    try:
                                        from telegram.helpers import escape_markdown as _esc
                                        _ftxt = (
                                            f"🚨 *EXIT SWAP FAILED*\n"
                                            f"🪙 {_esc(t.name or t.mint[:8], version=2)}\n"
                                            f"Trigger: `{_esc(reason, version=2)}`\n"
                                            f"Error: `{_esc(str(exit_sig)[:100], version=2)}`\n"
                                            f"Status: `FAILED_EXIT` \u2014 check manually\\!"
                                        )
                                        _app = getattr(bot, "_application", None)
                                        _fchats = []
                                        if _app and hasattr(_app, "bot_data"):
                                            _st = _app.bot_data.get("state")
                                            if _st:
                                                _fchats = list(_st.alerts.keys())
                                        for _cid in _fchats:
                                            try:
                                                await bot.send_message(chat_id=_cid, text=_ftxt, parse_mode="MarkdownV2")
                                            except Exception:
                                                pass
                                    except Exception as _fe:
                                        log.debug("failed exit notify: %s", _fe)
                                exit_mc = mc  # best available price for PnL calc
                                should_close = False  # prevent _close() from running
                        else:
                            exit_mc = mc
                            # simulated: estimate proceeds from MC-based pnl
                            pnl_est = ((exit_mc - t.entry_mc) / t.entry_mc
                                       if t.entry_mc > 0 else 0.0)
                            sol_received_actual = t.position_size_sol * (1 + pnl_est)

                        pnl_pct = ((exit_mc - t.entry_mc) / t.entry_mc * 100
                                   if t.entry_mc > 0 else 0)
                        # For real swaps use actual SOL received; simulated uses MC estimate
                        if wallet and not wallet.get("simulated"):
                            pnl_sol = sol_received_actual - t.position_size_sol
                        else:
                            pnl_sol = t.position_size_sol * (pnl_pct / 100.0)

                        def _close(tid=t.id, tmc=exit_mc, tp=pnl_pct, tsol=pnl_sol,
                                   r=reason, esig=exit_sig if ok else ""):
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
                                            reason=?, exit_tx_signature=?, status='CLOSED'
                                        WHERE id=? AND status='OPEN'
                                    """, (now_ts(), tmc, tsol, tp, r, esig, tid))
                                    conn.execute("COMMIT")
                                    return True
                                except Exception as e:
                                    try:
                                        conn.execute("ROLLBACK")
                                    except Exception:
                                        pass
                                    log.error("_close tx failed: %s", e)
                                    return False
                        if not should_close:  # exit swap failed — already marked FAILED_EXIT
                            continue
                        db_write(_close)

                        log.info("REAL CLOSE | %s | %s | %+.1f%% (%+.3f SOL)",
                                 t.name or t.mint[:8], reason, pnl_pct, pnl_sol)

                        if bot and _notify_state:
                            try:
                                from telegram.helpers import escape_markdown
                                emoji = "🟢" if pnl_pct > 0 else "🔴"
                                pnl_icon = "📈" if pnl_pct > 0 else "📉"
                                text = (
                                    f"{emoji} *TRADE CLOSED* — {SOLANA_NETWORK.upper()}\n"
                                    f"🪙 {escape_markdown(t.name or t.mint[:8], version=2)}\n"
                                    f"💰 Entry MC: `${t.entry_mc:,.0f}`\n"
                                    f"📊 Exit MC: `${exit_mc:,.0f}`\n"
                                    f"🏔 Peak MC: `${t.highest_mc:,.0f}`\n"
                                    f"🏁 Reason: `{escape_markdown(reason, version=2)}`\n"
                                    f"{pnl_icon} P&L: `{pnl_pct:+.1f}%` \\(`{pnl_sol:+.4f} SOL`\\)"
                                )
                                for cid in list(_notify_state.alerts.keys()):
                                    try:
                                        await bot.send_message(
                                            chat_id=cid, text=text,
                                            parse_mode="MarkdownV2")
                                    except Exception:
                                        pass
                            except Exception as e:
                                log.debug("real close notify: %s", e)

            # ── FAILED_EXIT retry block ──────────────────────────────────────
            failed = get_failed_exit_trades()
            if failed:
                wallet = _load_wallet()
                async with aiohttp.ClientSession() as session:
                    for t in failed:
                        mc = await fetch_coin_mc(session, t.mint)
                        exit_mc = mc if mc else t.entry_mc
                        ok, msg, sol_received = (False, "no wallet", 0.0)
                        if wallet and not wallet.get("simulated") and t.token_amount > 0:
                            # Fetch actual on-chain balance to avoid dust on retry
                            raw_amount = await _get_token_balance_async(
                                session, wallet["pubkey"], t.mint)
                            if raw_amount == 0:
                                raw_amount = int(round(t.token_amount))
                                log.debug("retry: on-chain balance fetch failed, using stored amount")
                            ok, msg, sol_received = await swap_token_for_sol(
                                session, t.mint, raw_amount, wallet)

                        if ok:
                            sol_actual = sol_received / 1_000_000_000
                            pnl_sol = sol_actual - t.position_size_sol
                            # Use SOL-based pnl_pct when MC unavailable
                            pnl_pct = (pnl_sol / t.position_size_sol * 100
                                       if t.position_size_sol else 0.0)
                            if mc is not None and t.entry_mc > 0:
                                pnl_pct = (exit_mc - t.entry_mc) / t.entry_mc * 100
                            def _retry_close(tid=t.id, emc=exit_mc,
                                             ps=pnl_sol, pp=pnl_pct):
                                with closing(db_conn()) as conn, conn:
                                    conn.execute("""
                                        UPDATE real_trades
                                        SET exit_time=?, exit_mc=?, pnl_sol=?,
                                            pnl_pct=?, reason='FAILED_EXIT_RETRY',
                                            status='CLOSED'
                                        WHERE id=? AND status='FAILED_EXIT'
                                    """, (now_ts(), emc, ps, pp, tid))
                            db_write(_retry_close)
                            log.info("FAILED_EXIT retried OK | %s | %+.3f SOL",
                                     t.name or t.mint[:8], pnl_sol)
                            if bot:
                                try:
                                    from telegram.helpers import escape_markdown as _esc
                                    _rtxt = (
                                        f"✅ *FAILED EXIT RECOVERED*\n"
                                        f"🪙 {_esc(t.name or t.mint[:8], version=2)}\n"
                                        f"P&L: `{pnl_pct:+.1f}%` \\(`{pnl_sol:+.4f} SOL`\\)"
                                    )
                                    _app = getattr(bot, "_application", None)
                                    _rchats = []
                                    if _app and hasattr(_app, "bot_data"):
                                        _st = _app.bot_data.get("state")
                                        if _st:
                                            _rchats = list(_st.alerts.keys())
                                    for _cid in _rchats:
                                        try:
                                            await bot.send_message(
                                                chat_id=_cid, text=_rtxt,
                                                parse_mode="MarkdownV2")
                                        except Exception:
                                            pass
                                except Exception as _re:
                                    log.debug("retry_close notify: %s", _re)
                        else:
                            # Increment attempt counter
                            def _inc_fail(tid=t.id, emsg=str(msg),
                                          attempts=t.failed_exit_attempts):
                                with closing(db_conn()) as conn, conn:
                                    conn.execute(
                                        "UPDATE real_trades SET "
                                        "failed_exit_attempts=?, exit_error=? "
                                        "WHERE id=?",
                                        (attempts + 1, emsg[:500], tid))
                            db_write(_inc_fail)
                            log.warning("FAILED_EXIT retry %d/%d failed | %s: %s",
                                        t.failed_exit_attempts + 1,
                                        REAL_MAX_EXIT_RETRIES,
                                        t.mint[:8], msg)
                            # Final attempt exhausted — alert and give up
                            if t.failed_exit_attempts + 1 >= REAL_MAX_EXIT_RETRIES:
                                log.error("FAILED_EXIT max retries hit for %s — manual action required",
                                          t.mint[:8])
                                # Mark ABANDONED so the trade leaves FAILED_EXIT and is
                                # counted as a loss in real_stats() instead of disappearing.
                                def _abandon(tid=t.id, emsg=str(msg)):
                                    with closing(db_conn()) as conn, conn:
                                        conn.execute(
                                            "UPDATE real_trades "
                                            "SET status='ABANDONED', exit_error=? "
                                            "WHERE id=? AND status='FAILED_EXIT'",
                                            (emsg[:500], tid))
                                db_write(_abandon)
                                if bot:
                                    try:
                                        from telegram.helpers import escape_markdown as _esc
                                        _etxt = (
                                            f"🆘 *EXIT RETRIES EXHAUSTED*\n"
                                            f"🪙 {_esc(t.name or t.mint[:8], version=2)}\n"
                                            f"After {REAL_MAX_EXIT_RETRIES} attempts\\."
                                            f" Manual sell required\\!"
                                        )
                                        _app = getattr(bot, "_application", None)
                                        _echats = []
                                        if _app and hasattr(_app, "bot_data"):
                                            _st = _app.bot_data.get("state")
                                            if _st:
                                                _echats = list(_st.alerts.keys())
                                        for _cid in _echats:
                                            try:
                                                await bot.send_message(
                                                    chat_id=_cid, text=_etxt,
                                                    parse_mode="MarkdownV2")
                                            except Exception:
                                                pass
                                    except Exception as _ee:
                                        log.debug("exhausted notify: %s", _ee)

        except Exception as e:
            log.error("real_monitor_loop: %s", e)
        await asyncio.sleep(REAL_MONITOR_INTERVAL_SEC)


def real_stats() -> dict:
    with closing(db_conn()) as conn:
        closed = conn.execute(
            "SELECT pnl_pct, pnl_sol FROM real_trades WHERE status IN ('CLOSED','ABANDONED') "
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
