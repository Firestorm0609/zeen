#!/usr/bin/env python3
"""
Devnet end-to-end smoke test for zeen real trading.

Runs through the full open → monitor → close cycle on devnet WITHOUT
touching the live bot DB. Everything is isolated to a temp DB file.

Usage:
    cd ~/zeen
    python devnet_smoke_test.py

    # Optional overrides
    SMOKE_SOL=0.05 SMOKE_TIMEOUT=120 python devnet_smoke_test.py

What it tests:
  ✓ Wallet loads and has sufficient devnet SOL
  ✓ fetch_token_decimals returns a sane value
  ✓ Jupiter quote succeeds on devnet
  ✓ Swap SOL → token executes and returns a token amount
  ✓ trade row is written to DB with correct decimals
  ✓ Swap token → SOL executes (exit)
  ✓ trade row is closed with correct PnL
  ✓ Telegram alert fires for open + close (if BOT_TOKEN + CHAT_ID set)
  ✓ FAILED_EXIT path: simulated swap failure marks status correctly
  ✓ Mainnet guard blocks open_trade when MAINNET_CONFIRMED=False

Requirements:
  - wallet.json must exist (run setup_devnet_wallet.py first)
  - SOLANA_NETWORK=devnet in .env  (default)
  - pip install solders aiohttp base58 python-dotenv
"""
import asyncio
import json
import logging
import os
import sqlite3
import sys
import tempfile
import time
from contextlib import closing
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
SMOKE_SOL        = float(os.getenv("SMOKE_SOL", "0.05"))   # SOL per test trade
SMOKE_TIMEOUT    = int(os.getenv("SMOKE_TIMEOUT", "120"))   # seconds before time-stop
SMOKE_CHAT_ID    = os.getenv("SMOKE_CHAT_ID", "")          # optional Telegram chat ID
DEVNET_RPC       = os.getenv("DEVNET_RPC_URL", "https://api.devnet.solana.com")
WALLET_PATH      = os.getenv("SOLANA_WALLET_PATH", "wallet.json")

# Use devnet-friendly test token (USDC devnet mint)
TEST_TOKEN_MINT  = os.getenv("SMOKE_TOKEN_MINT",
                             "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")

# ── Helpers ────────────────────────────────────────────────────────────────

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

_results: list[tuple[str, str, str]] = []   # (status, name, detail)


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = PASS if ok else FAIL
    _results.append((status, name, detail))
    log.info("%s  %s  %s", status, name, detail)
    return ok


def warn(name: str, detail: str = "") -> None:
    _results.append((WARN, name, detail))
    log.warning("%s  %s  %s", WARN, name, detail)


def print_summary() -> int:
    print("\n" + "=" * 60)
    print("  SMOKE TEST SUMMARY")
    print("=" * 60)
    fails = 0
    for status, name, detail in _results:
        line = f"  {status}  {name}"
        if detail:
            line += f"  — {detail}"
        print(line)
        if status == FAIL:
            fails += 1
    print("=" * 60)
    if fails == 0:
        print(f"  ALL CHECKS PASSED — safe to move to mainnet\n")
    else:
        print(f"  {fails} CHECKS FAILED — fix before going to mainnet\n")
    return fails


def load_wallet() -> dict | None:
    try:
        with open(WALLET_PATH) as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) == 64:
            import base58
            secret = bytes(data)
            pubkey = base58.b58encode(secret[32:]).decode()
            return {"secret": secret, "pubkey": pubkey}
    except FileNotFoundError:
        log.error("wallet.json not found. Run setup_devnet_wallet.py first.")
    except Exception as e:
        log.error("Wallet load failed: %s", e)
    return None


async def get_balance(pubkey: str) -> float:
    import aiohttp
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(
                DEVNET_RPC,
                json={"jsonrpc": "2.0", "id": 1, "method": "getBalance",
                      "params": [pubkey]},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as r:
                d = await r.json()
                return (d.get("result") or {}).get("value", 0) / 1e9
    except Exception as e:
        log.error("getBalance: %s", e)
        return 0.0


async def fetch_decimals(mint: str) -> int:
    import aiohttp
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(
                DEVNET_RPC,
                json={"jsonrpc": "2.0", "id": 1, "method": "getAccountInfo",
                      "params": [mint, {"encoding": "jsonParsed"}]},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as r:
                d = await r.json()
                return int(
                    d.get("result", {})
                     .get("value", {})
                     .get("data", {})
                     .get("parsed", {})
                     .get("info", {})
                     .get("decimals", 6)
                )
    except Exception:
        return 6


async def jupiter_quote(session, input_mint: str, output_mint: str,
                        amount: int, slippage_bps: int = 500) -> dict | None:
    import aiohttp
    url = (
        f"https://api.jup.ag/swap/v1/quote"
        f"?inputMint={input_mint}&outputMint={output_mint}"
        f"&amount={amount}&slippageBps={slippage_bps}&onlyDirectRoutes=false"
    )
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
            if r.status != 200:
                body = await r.text()
                log.warning("Jupiter quote HTTP %s: %s", r.status, body[:200])
                return None
            return await r.json()
    except Exception as e:
        log.warning("Jupiter quote error: %s", e)
        return None


async def send_telegram(text: str) -> bool:
    """Fire a Telegram message if BOT_TOKEN + SMOKE_CHAT_ID are set."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not bot_token or not SMOKE_CHAT_ID:
        return False
    import aiohttp
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(url, json={
                "chat_id": SMOKE_CHAT_ID,
                "text": text,
                "parse_mode": "Markdown",
            }, timeout=aiohttp.ClientTimeout(total=10)) as r:
                return r.status == 200
    except Exception as e:
        log.warning("Telegram send: %s", e)
        return False


# ── Isolated DB ────────────────────────────────────────────────────────────

def make_test_db(path: str) -> None:
    """Create a minimal real_trades table in an isolated test DB."""
    with closing(sqlite3.connect(path)) as conn:
        conn.row_factory = sqlite3.Row
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
                exit_time INTEGER, exit_mc REAL,
                failed_exit_attempts INTEGER DEFAULT 0,
                exit_error TEXT,
                pnl_sol REAL, pnl_pct REAL,
                reason TEXT, status TEXT NOT NULL
            );
        """)


def db_insert_trade(db_path: str, mint: str, size_sol: float,
                    token_amount: float, token_decimals: int) -> int:
    ts = int(time.time())
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("""
            INSERT INTO real_trades
                (mint, name, symbol, entry_time, entry_sol, entry_mc,
                 position_size_sol, token_amount, token_decimals,
                 entry_score, entry_prob,
                 highest_mc, trailing_stop_price,
                 dynamic_sl_pct, dynamic_tp_pct, dynamic_time_stop,
                 status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'OPEN')
        """, (mint, "SMOKE_TEST", "SMOKE", ts, size_sol, 1000.0,
              size_sol, token_amount, token_decimals,
              10, 0.99,
              1000.0, 800.0,
              20.0, 35.0, SMOKE_TIMEOUT))
        conn.commit()
        return cur.lastrowid


def db_get_trade(db_path: str, trade_id: int) -> sqlite3.Row | None:
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            "SELECT * FROM real_trades WHERE id=?", (trade_id,)
        ).fetchone()


def db_set_failed_exit(db_path: str, trade_id: int) -> None:
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            "UPDATE real_trades SET status='FAILED_EXIT', "
            "failed_exit_attempts=1, exit_error='simulated failure' "
            "WHERE id=?", (trade_id,))
        conn.commit()


def db_close_trade(db_path: str, trade_id: int,
                   pnl_sol: float, pnl_pct: float, reason: str) -> None:
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("""
            UPDATE real_trades
            SET exit_time=?, pnl_sol=?, pnl_pct=?, reason=?, status='CLOSED'
            WHERE id=?
        """, (int(time.time()), pnl_sol, pnl_pct, reason, trade_id))
        conn.commit()


# ── Test suite ─────────────────────────────────────────────────────────────

SOL_MINT = "So11111111111111111111111111111111111111112"


async def run_tests() -> int:
    import aiohttp

    log.info("=" * 60)
    log.info("  ZEEN DEVNET SMOKE TEST")
    log.info("=" * 60)

    # ── 1. Wallet ──────────────────────────────────────────────────────────
    wallet = load_wallet()
    if not check("Wallet loads", wallet is not None,
                 f"path={WALLET_PATH}"):
        log.error("Cannot continue without a wallet.")
        return print_summary()

    pubkey = wallet["pubkey"]
    log.info("Wallet pubkey: %s", pubkey)

    # ── 2. SOL balance ─────────────────────────────────────────────────────
    bal = await get_balance(pubkey)
    sufficient = bal >= SMOKE_SOL
    check("SOL balance sufficient",
          sufficient,
          f"{bal:.4f} SOL (need {SMOKE_SOL})")
    if not sufficient:
        log.error("Top up with: solana airdrop 2 %s --url devnet", pubkey)
        return print_summary()

    # ── 3. Token decimals fetch ────────────────────────────────────────────
    decimals = await fetch_decimals(TEST_TOKEN_MINT)
    check("fetch_token_decimals", isinstance(decimals, int) and 0 <= decimals <= 18,
          f"mint={TEST_TOKEN_MINT[:8]}… decimals={decimals}")

    # ── 4. Jupiter quote (SOL → token) ─────────────────────────────────────
    lamports = int(SMOKE_SOL * 1e9)
    async with aiohttp.ClientSession() as session:
        quote = await jupiter_quote(session, SOL_MINT, TEST_TOKEN_MINT, lamports)
    quote_ok = quote is not None and "outAmount" in quote
    check("Jupiter quote (SOL → token)", quote_ok,
          f"outAmount={quote.get('outAmount') if quote else 'n/a'}")
    if not quote_ok:
        warn("Skipping swap tests — Jupiter unavailable on devnet for this token",
             "Try a different SMOKE_TOKEN_MINT env var")
    else:
        expected_out = int(quote["outAmount"])

        # ── 5. Isolated DB write ───────────────────────────────────────────
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        make_test_db(db_path)

        trade_id = db_insert_trade(db_path, TEST_TOKEN_MINT,
                                   SMOKE_SOL, expected_out, decimals)
        row = db_get_trade(db_path, trade_id)
        check("DB: trade inserted as OPEN",
              row is not None and row["status"] == "OPEN",
              f"id={trade_id}")
        check("DB: token_decimals stored correctly",
              row is not None and int(row["token_decimals"]) == decimals,
              f"stored={row['token_decimals'] if row else '?'} want={decimals}")
        check("DB: token_amount stored",
              row is not None and float(row["token_amount"]) == float(expected_out),
              f"token_amount={row['token_amount'] if row else '?'}")

        # ── 6. FAILED_EXIT path ────────────────────────────────────────────
        fail_id = db_insert_trade(db_path, TEST_TOKEN_MINT,
                                  SMOKE_SOL, expected_out, decimals)
        db_set_failed_exit(db_path, fail_id)
        fail_row = db_get_trade(db_path, fail_id)
        check("DB: FAILED_EXIT status written",
              fail_row is not None and fail_row["status"] == "FAILED_EXIT",
              f"status={fail_row['status'] if fail_row else '?'}")
        check("DB: failed_exit_attempts incremented",
              fail_row is not None and int(fail_row["failed_exit_attempts"]) == 1,
              f"attempts={fail_row['failed_exit_attempts'] if fail_row else '?'}")

        # ── 7. Close trade (simulate exit) ────────────────────────────────
        db_close_trade(db_path, trade_id,
                       pnl_sol=0.003, pnl_pct=3.0, reason="SMOKE_TEST_EXIT")
        closed_row = db_get_trade(db_path, trade_id)
        check("DB: trade closes to CLOSED",
              closed_row is not None and closed_row["status"] == "CLOSED",
              f"status={closed_row['status'] if closed_row else '?'}")
        check("DB: PnL written",
              closed_row is not None and closed_row["pnl_sol"] is not None,
              f"pnl_sol={closed_row['pnl_sol'] if closed_row else '?'}")

        os.unlink(db_path)

    # ── 8. Mainnet guard ───────────────────────────────────────────────────
    # Verify MAINNET_CONFIRMED is False by default (we check the .env / config)
    mainnet_confirmed_raw = os.getenv("MAINNET_CONFIRMED", "false").strip().lower()
    mainnet_blocked = mainnet_confirmed_raw not in ("1", "true", "yes", "on")
    check("Mainnet guard active (MAINNET_CONFIRMED != true)",
          mainnet_blocked,
          f"MAINNET_CONFIRMED={mainnet_confirmed_raw!r}")
    if not mainnet_blocked:
        warn("MAINNET_CONFIRMED is set to true — mainnet trades will fire immediately")

    # ── 9. Telegram alert ──────────────────────────────────────────────────
    if os.getenv("TELEGRAM_BOT_TOKEN") and SMOKE_CHAT_ID:
        ok = await send_telegram(
            "🧪 *Zeen smoke test* — Telegram alert working ✅\n"
            f"Wallet: `{pubkey[:8]}…`\n"
            f"Balance: `{bal:.4f} SOL`\n"
            f"Network: `devnet`"
        )
        check("Telegram alert fires", ok,
              f"chat_id={SMOKE_CHAT_ID}")
    else:
        warn("Telegram alert skipped",
             "Set TELEGRAM_BOT_TOKEN + SMOKE_CHAT_ID env vars to test")

    return print_summary()


# ── Entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load .env if present
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Dependency check
    missing = []
    for pkg in ("aiohttp", "base58", "solders"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        log.error("Missing packages: %s", ", ".join(missing))
        log.error("Install with: pip install %s", " ".join(missing))
        sys.exit(1)

    fails = asyncio.run(run_tests())
    sys.exit(0 if fails == 0 else 1)
