"""Main per-coin processing pipeline."""
import asyncio
import logging
import time as _time

from telegram import Bot

from .alerts import send_alert
from .config import (
    MAX_CONCURRENT_PROCESS, MAX_MARKET_CAP, MIN_MARKET_CAP,
    MIN_CREATOR_WALLET_AGE_DAYS, MAX_VOLUME_MC_RATIO,
)
from .market import MarketContext
from .scoring import ScoringEngine
from .state import BotState
from .storage import save_signal, save_snapshot
from .lookback import schedule_lookbacks
from .trading import record_creator_token
from .real_trading import maybe_open_real_trade
from .utils import safe_float, safe_int

log = logging.getLogger(__name__)

_semaphore: asyncio.Semaphore | None = None


def init_semaphore() -> None:
    """Call once from inside the running event loop (e.g. at bot startup)."""
    global _semaphore
    _semaphore = asyncio.Semaphore(MAX_CONCURRENT_PROCESS)


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(MAX_CONCURRENT_PROCESS)
    return _semaphore


def hard_filter(coin: dict) -> tuple[bool, str]:
    name   = (coin.get("name")        or "").strip()
    symbol = (coin.get("symbol")      or "").strip()
    desc   = (coin.get("description") or "").strip()
    mc     = safe_float(coin.get("usd_market_cap"))

    if len(name)   < 2:     return False, "name too short"
    if len(symbol) < 2:     return False, "symbol too short"
    if not desc:            return False, "no description"
    if mc < MIN_MARKET_CAP: return False, "below min MC"
    if mc > MAX_MARKET_CAP: return False, "above max MC"
    if not any([coin.get("twitter"), coin.get("telegram"), coin.get("website")]):
        return False, "no socials"

    # Suggestion 8: bundled launch — multiple wallets transacted in the same slot
    bundle_score = coin.get("_rpc_bundle_score")
    if bundle_score is not None and bundle_score >= 1.0:
        return False, "bundled launch detected"

    # Suggestion 10: volume/MC ratio — if explicit 5m volume already exceeds
    # MAX_VOLUME_MC_RATIO of market cap the token has been heavily churned.
    # We only use volume_5m here — never the generic "volume" key, which may
    # be 24h volume and would produce a far too aggressive filter.
    volume_5m = safe_float(coin.get("volume_5m") or 0)
    if mc > 0 and volume_5m > 0 and volume_5m / mc > MAX_VOLUME_MC_RATIO:
        return False, f"vol/MC too high ({volume_5m/mc:.0%})"

    # Suggestion 11: creator wallet age — brand-new wallets are a strong rug signal
    wallet_age = coin.get("_rpc_creator_wallet_age_days")
    if wallet_age is not None and wallet_age < MIN_CREATOR_WALLET_AGE_DAYS:
        return False, f"creator wallet too young ({wallet_age:.1f}d)"

    return True, "ok"


async def process_coin(coin: dict, bot: Bot, engine: ScoringEngine,
                       market_ctx: MarketContext, state: BotState) -> None:
    async with _get_semaphore():
        loop = asyncio.get_running_loop()
        try:
            mint = coin.get("mint", "")
            if not mint:
                return

            if await state.seen_recently(mint):
                return

            state.last_coin_ts = _time.time()
            state.stream_dead_alerted = False

            mc      = safe_float(coin.get("usd_market_cap"))
            replies = safe_int(coin.get("reply_count"))

            if mc > 0:
                market_ctx.update(mc, replies)

            await loop.run_in_executor(None, save_snapshot, coin)

            ok, reason = hard_filter(coin)
            if not ok:
                log.debug("Filtered %s: %s", mint[:8], reason)
                await state.mark_seen(mint)
                return

            result    = await loop.run_in_executor(None, engine.score, coin)
            signal_id = await loop.run_in_executor(None, save_signal, coin, result)
            await loop.run_in_executor(None, schedule_lookbacks, signal_id, mint)

            creator = (coin.get("creator") or coin.get("user")
                       or coin.get("traderPublicKey") or "")
            if creator and mint:
                await loop.run_in_executor(None, record_creator_token, creator, mint)

            await maybe_open_real_trade(state, coin, result, market_ctx, bot)
            await send_alert(bot, coin, result, state)

            await state.mark_seen(mint)
        except Exception as e:
            log.error("process_coin failed for %s: %s",
                      (coin.get("mint", "?") or "?")[:8], e)
            mint = coin.get("mint")
            if mint:
                await state.mark_seen(mint)
