from typing import Optional
#!/usr/bin/env python3
"""Devnet wallet setup helper for zeen real trading.

Usage:
    python setup_devnet_wallet.py

What it does:
  1. Generates a new Solana keypair (or reuses existing wallet.json)
  2. Saves it to wallet.json (configurable via SOLANA_WALLET_PATH)
  3. Requests a SOL airdrop on devnet (2 SOL)
  4. Creates/updates .env with real trading settings
  5. Prints the public key and confirms balance

Requirements:
  pip install solders base58
  solana CLI (optional, for airdrop via CLI fallback)
"""
import json
import logging
import os
import subprocess
import sys
import time

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

WALLET_PATH = os.getenv("SOLANA_WALLET_PATH", "wallet.json")
ENV_PATH = ".env"
AIRDROP_AMOUNT = 2  # SOL


def generate_keypair() -> dict:
    from solders.keypair import Keypair
    kp = Keypair()
    secret = list(bytes(kp))
    pubkey = str(kp.pubkey())
    return {"secret": secret, "pubkey": pubkey}


def save_wallet(data: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(data["secret"], f)
    log.info("Wallet saved to %s", path)
    log.info("Public key: %s", data["pubkey"])


def load_existing_wallet(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            secret = json.load(f)
        if isinstance(secret, list) and len(secret) == 64:
            from solders.keypair import Keypair
            kp = Keypair.from_bytes(bytes(secret))
            return {"secret": secret, "pubkey": str(kp.pubkey())}
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning("Existing wallet unreadable: %s", e)
    return None


def airdrop_via_rpc(pubkey: str, amount_sol: int) -> bool:
    """Try airdrop via Solana RPC (no CLI needed)."""
    import base64
    import urllib.request

    url = "https://api.devnet.solana.com"
    txid = None

    for attempt in range(3):
        payload = {
            "jsonrpc": "2.0", "id": 1,
            "method": "requestAirdrop",
            "params": [pubkey, amount_sol * 1_000_000_000,
                       {"commitment": "confirmed"}],
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
            if "error" in result:
                log.warning("Airdrop RPC error (attempt %d): %s",
                             attempt + 1, result["error"])
                time.sleep(2)
                continue
            txid = result.get("result", {}).get("signature")
            if txid:
                log.info("Airdrop RPC success! tx: %s", txid)
                return True
        except Exception as e:
            log.warning("Airdrop RPC failed (attempt %d): %s",
                         attempt + 1, e)
            time.sleep(2)

    return False


def airdrop_via_cli(pubkey: str, amount_sol: int) -> bool:
    """Fallback: airdrop via solana CLI."""
    try:
        result = subprocess.run(
            ["solana", "airdrop", str(amount_sol), pubkey, "--url", "devnet"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            log.info("Airdrop CLI success: %s", result.stdout.strip())
            return True
        log.warning("Airdrop CLI failed: %s", result.stderr.strip())
    except FileNotFoundError:
        log.warning("solana CLI not found")
    except Exception as e:
        log.warning("Airdrop CLI error: %s", e)
    return False


def check_balance(pubkey: str) -> float:
    import urllib.request
    url = "https://api.devnet.solana.com"
    payload = {
        "jsonrpc": "2.0", "id": 1,
        "method": "getBalance",
        "params": [pubkey],
    }
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
        if "error" in result:
            return 0.0
        lamports = (result.get("result") or {}).get("value", 0)
        return lamports / 1_000_000_000
    except Exception:
        return 0.0


def update_env(pubkey: str) -> None:
    """Create or update .env with real trading settings."""
    lines = []
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, "r") as f:
            lines = f.readlines()

    keys_to_set = {
        "REAL_TRADING_ENABLED": "True",
        "SOLANA_NETWORK": "devnet",
        "SOLANA_WALLET_PATH": WALLET_PATH,
        "REAL_POSITION_SIZE_SOL": "0.1",
        "REAL_MIN_SCORE": "8",
        "REAL_MIN_PROB": "0.75",
    }

    updated = set()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        matched = False
        for key in keys_to_set:
            if stripped.startswith(key + "=") or stripped.startswith(key + " "):
                new_lines.append(f"{key}={keys_to_set[key]}\n")
                updated.add(key)
                matched = True
                break
        if not matched:
            new_lines.append(line)

    for key in keys_to_set:
        if key not in updated:
            new_lines.append(f"{key}={keys_to_set[key]}\n")

    with open(ENV_PATH, "w") as f:
        f.writelines(new_lines)

    log.info(".env updated: %s", ENV_PATH)
    for k, v in keys_to_set.items():
        log.info("  %s=%s", k, v)


def main():
    log.info("=== Zeen Devnet Wallet Setup ===")

    # Check dependencies
    try:
        import solders  # noqa: F401
        import base58  # noqa: F401
    except ImportError:
        log.error("Missing dependencies. Install with:")
        log.error("  pip install solders base58")
        sys.exit(1)

    # Generate or load wallet
    wallet = load_existing_wallet(WALLET_PATH)
    if wallet:
        log.info("Using existing wallet: %s", wallet["pubkey"])
    else:
        log.info("Generating new keypair...")
        wallet = generate_keypair()
        save_wallet(wallet, WALLET_PATH)

    pubkey = wallet["pubkey"]
    log.info("Public key: %s", pubkey)

    # Check current balance
    bal = check_balance(pubkey)
    log.info("Current balance: %.4f SOL", bal)

    if bal >= AIRDROP_AMOUNT:
        log.info("Balance sufficient, skipping airdrop.")
    else:
        log.info("Requesting %d SOL airdrop on devnet...", AIRDROP_AMOUNT)
        ok = airdrop_via_rpc(pubkey, AIRDROP_AMOUNT)
        if not ok:
            log.info("RPC airdrop failed, trying CLI...")
            ok = airdrop_via_cli(pubkey, AIRDROP_AMOUNT)
        if ok:
            time.sleep(3)
            new_bal = check_balance(pubkey)
            log.info("New balance: %.4f SOL", new_bal)
        else:
            log.warning("Airdrop failed. You can manually run:")
            log.warning("  solana airdrop %d %s --url devnet", AIRDROP_AMOUNT, pubkey)

    # Update .env
    update_env(pubkey)

    log.info("")
    log.info("=== Setup Complete ===")
    log.info("Wallet: %s", WALLET_PATH)
    log.info("Public key: %s", pubkey)
    log.info("")
    log.info("Next steps:")
    log.info("  1. Verify .env has REAL_TRADING_ENABLED=True")
    log.info("  2. pip install solders")
    log.info("  3. python -m zeen  (or run your bot)")
    log.info("  4. In Telegram: /real_on to enable real trading")


if __name__ == "__main__":
    main()
