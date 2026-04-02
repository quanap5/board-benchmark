#!/usr/bin/env python3
"""Shared logging utilities with color and emoji for benchmark scripts."""

import sys

# ANSI color codes (work on Linux/Mac/Jetson terminals)
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
BG_GREEN = "\033[42m"
BG_RED = "\033[41m"
BG_BLUE = "\033[44m"


def supports_color():
    """Check if terminal supports color."""
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


NO_COLOR = not supports_color()


def c(text, color):
    """Wrap text with color. No-op if terminal doesn't support color."""
    if NO_COLOR:
        return text
    return "{}{}{}".format(color, text, RESET)


def info(msg):
    print("{} {}".format(c("ℹ️  INFO", CYAN + BOLD), msg))


def ok(msg):
    print("{} {}".format(c("✅ OK  ", GREEN + BOLD), msg))


def warn(msg):
    print("{} {}".format(c("⚠️  WARN", YELLOW + BOLD), msg))


def error(msg):
    print("{} {}".format(c("❌ ERR ", RED + BOLD), msg))


def step(msg):
    print("{} {}".format(c("🔧 STEP", MAGENTA + BOLD), msg))


def speed(msg):
    print("{} {}".format(c("🚀 PERF", GREEN + BOLD), msg))


def wait(msg):
    print("{} {}".format(c("⏳ WAIT", YELLOW), msg))


def header(title):
    """Print a decorated section header."""
    w = 58
    line = c("=" * w, CYAN)
    icon = "🏁" if "SUMMARY" in title.upper() or "COMPARISON" in title.upper() else \
           "📊" if "BENCHMARK" in title.upper() or "RESULT" in title.upper() else \
           "🖥️ " if "HARDWARE" in title.upper() or "SYSTEM" in title.upper() else \
           "⚡" if "ENGINE" in title.upper() or "CONVERT" in title.upper() else "📋"
    print("\n{}".format(line))
    print("  {} {}".format(icon, c(title, BOLD + WHITE)))
    print(line)


def metric(label, value, unit=""):
    """Print a formatted metric line."""
    val_str = "{}".format(value)
    if unit and value != "N/A":
        val_str = "{} {}".format(value, unit)
    print("  {:<20} {}".format(label, c(val_str, BOLD)))


def table_row(label, *values):
    """Print a table row with multiple columns."""
    row = "  {:<16}".format(label)
    for v in values:
        row += " {:>14}".format(c(str(v), BOLD))
    print(row)


def divider():
    print(c("  " + "-" * 54, DIM))
