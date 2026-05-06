"""All inline keyboard builders."""
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from .utils import mdbold

MENU_HEADER = f"🤖 {mdbold('Pump.fun Monitor v1.1')} — tap an action:"


def main_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🟢 Alerts ON",  callback_data="monitor_on"),
            InlineKeyboardButton("🔴 Alerts OFF", callback_data="monitor_off"),
            InlineKeyboardButton("ℹ️ Status",     callback_data="monitor_status"),
        ],
        [
            InlineKeyboardButton("🎚 Threshold", callback_data="threshold_menu"),
            InlineKeyboardButton("📊 Scoring",   callback_data="scoring_mode"),
            InlineKeyboardButton("🧬 Features",  callback_data="features"),
        ],
        [
            InlineKeyboardButton("🔤 Keywords", callback_data="keywords"),
            InlineKeyboardButton("📈 Market",   callback_data="market"),
            InlineKeyboardButton("📤 Outcomes", callback_data="outcomes"),
        ],
        [
            InlineKeyboardButton("🤖 Model",    callback_data="model"),
            InlineKeyboardButton("🏋 Train",    callback_data="train"),
            InlineKeyboardButton("🗒 Snapshot", callback_data="snapshot"),
        ],
        [
            InlineKeyboardButton("📊 Stats",  callback_data="stats"),
            InlineKeyboardButton("💰 Wallet", callback_data="wallet"),
            InlineKeyboardButton("🏆 Top",    callback_data="top"),
        ],
        [
            InlineKeyboardButton("⚡ Real ON",  callback_data="real_on"),
            InlineKeyboardButton("⛔ Real OFF", callback_data="real_off"),
        ],
        [
            InlineKeyboardButton("📊 Real Status",  callback_data="real_status"),
            InlineKeyboardButton("📑 Real Report", callback_data="real_report"),
        ],
        [InlineKeyboardButton("⚙️ More…", callback_data="more")],
        [InlineKeyboardButton("✖ Close", callback_data="close_menu")],
    ])


def more_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("❤️ Health",    callback_data="health"),
            InlineKeyboardButton("❓ Help",      callback_data="help"),
        ],
        [
            InlineKeyboardButton("📊 Backtest",  callback_data="backtest"),
            InlineKeyboardButton("⚡ Last Trade", callback_data="last"),
        ],
        [
            InlineKeyboardButton("💱 Trade Size",   callback_data="trade_size_menu"),
            InlineKeyboardButton("👛 Wallet Mgmt",  callback_data="wallet_menu"),
        ],
        [InlineKeyboardButton("🔙 Main Menu", callback_data="menu")],
    ])


def trade_size_keyboard(current: float = 0.1) -> InlineKeyboardMarkup:
    sizes = [0.05, 0.1, 0.25, 0.5, 1.0]
    buttons = []
    for s in sizes:
        label = f"{'✅ ' if abs(s - current) < 0.001 else ''}{s} SOL"
        buttons.append(InlineKeyboardButton(label, callback_data=f"set_size_{s}"))
    return InlineKeyboardMarkup([
        buttons[:3],
        buttons[3:],
        [InlineKeyboardButton("✏️ Custom (/trade_size)", callback_data="size_custom_hint")],
        [InlineKeyboardButton("🔙 Back", callback_data="more")],
    ])


def wallet_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📋 Show Address", callback_data="wallet_address")],
        [InlineKeyboardButton("🔑 Export Private Key", callback_data="wallet_export_key")],
        [InlineKeyboardButton("📥 Import Wallet", callback_data="wallet_import_hint")],
        [InlineKeyboardButton("🔙 Back", callback_data="more")],
    ])


def threshold_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(str(i), callback_data=f"set_threshold_{i}")
         for i in range(1, 6)],
        [InlineKeyboardButton(str(i), callback_data=f"set_threshold_{i}")
         for i in range(6, 11)],
        [InlineKeyboardButton("🔙 Back", callback_data="menu")],
    ])


def back_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(text="⬅️ Back", callback_data="back")]
    ])
