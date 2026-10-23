"""
Runs once (on GitHub Actions) and sends match analysis to your Telegram.

- If MATCHES is set (typed when you run the workflow manually), it analyses
  each match: separate matches with a semicolon, e.g.
    Arsenal vs Chelsea; PL: Man City vs Liverpool
- Otherwise (the daily scheduled run) it sends win probabilities for all of
  today's fixtures in your leagues.
Needs match_analyst_bot.py in the same folder.
"""
import os
import re

import requests

import match_analyst_bot as bot

TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT = os.environ["TELEGRAM_CHAT_ID"]


def send(text):
    for i in range(0, len(text), 3900):
        requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            data={"chat_id": CHAT, "text": text[i:i + 3900]},
            timeout=30,
        )


def main():
    raw = os.getenv("MATCHES", "").strip()
    matches = [m.strip() for m in re.split(r"[;\n]", raw) if m.strip()]
    if matches:
        for m in matches:
            try:
                send(bot.handle(m))
            except Exception as e:
                send(f"{m}: error - {e}")
    else:
        try:
            send(bot.handle("/today"))
        except Exception as e:
            send(f"Daily report failed: {e}")


if __name__ == "__main__":
    main()
