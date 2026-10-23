"""
Match Analyst Bot (Telegram) - head-to-head, form, stats and win probability.

Free data source: football-data.org (free key, 10 requests/min, covers
PL, PD (La Liga), BL1, SA, FL1, DED, PPL, ELC, BSA, CL, EC, WC).

Setup:
  pip install requests
  Environment variables:
    FOOTBALL_DATA_KEY  free key from https://www.football-data.org/client/register
    TELEGRAM_TOKEN     from @BotFather
  Optional:
    LEAGUES            default PL,PD,BL1,SA,FL1
    SEASONS_BACK       default 3 (current season + 2 previous)
    ALLOWED_CHAT_ID    only answer this chat id (recommended)
  Run:  python match_analyst_bot.py

Usage in Telegram:
  Arsenal vs Chelsea
  PL: Man City vs Liverpool        (force a league)
  /today                           (win probabilities for today's fixtures)
"""
import difflib
import json
import math
import os
import re
import time
from datetime import date, datetime

import requests

FD_KEY = os.environ["FOOTBALL_DATA_KEY"]
TG_TOKEN = os.environ["TELEGRAM_TOKEN"]
LEAGUES = [x.strip().upper() for x in os.getenv("LEAGUES", "PL,PD,BL1,SA,FL1").split(",")]
SEASONS_BACK = int(os.getenv("SEASONS_BACK", "3"))
ALLOWED = os.getenv("ALLOWED_CHAT_ID")
BASE = "https://api.football-data.org/v4"
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

_last_call = [0.0]


# ---------------------------------------------------------------- data layer
def api(path, params=None, ttl=6 * 3600):
    key = path + json.dumps(params or {}, sort_keys=True)
    fn = os.path.join(CACHE_DIR, re.sub(r"\W+", "_", key)[:150] + ".json")
    if os.path.exists(fn) and time.time() - os.path.getmtime(fn) < ttl:
        with open(fn) as f:
            return json.load(f)
    for attempt in range(2):
        wait = 6.5 - (time.time() - _last_call[0])  # stay under 10 req/min
        if wait > 0:
            time.sleep(wait)
        r = requests.get(BASE + path, headers={"X-Auth-Token": FD_KEY},
                         params=params, timeout=30)
        _last_call[0] = time.time()
        if r.status_code == 429 and attempt == 0:
            time.sleep(60)
            continue
        r.raise_for_status()
        data = r.json()
        with open(fn, "w") as f:
            json.dump(data, f)
        return data


def current_season():
    t = date.today()
    return t.year if t.month >= 8 else t.year - 1


def load_season(code, year):
    ttl = 6 * 3600 if year == current_season() else 30 * 86400
    try:
        data = api(f"/competitions/{code}/matches",
                   {"season": year, "status": "FINISHED"}, ttl)
    except requests.HTTPError:
        return []  # season not available on the free tier
    out = []
    for m in data.get("matches", []):
        ft = m["score"]["fullTime"]
        if ft["home"] is None or ft["away"] is None:
            continue
        out.append({
            "d": datetime.strptime(m["utcDate"][:10], "%Y-%m-%d").date(),
            "h": m["homeTeam"]["name"], "a": m["awayTeam"]["name"],
            "hg": ft["home"], "ag": ft["away"],
            "alias": [(m["homeTeam"]["name"], m["homeTeam"]), (m["awayTeam"]["name"], m["awayTeam"])],
        })
    return out


def load_history(code, seasons=SEASONS_BACK):
    cur = current_season()
    hist = []
    for y in range(cur, cur - seasons, -1):
        hist += load_season(code, y)
    return hist


def alias_map(matches):
    amap = {}
    for m in matches:
        for name, t in m["alias"]:
            for a in (name, t.get("shortName"), t.get("tla")):
                if a:
                    amap[a.lower()] = name
    return amap


def resolve(q, amap):
    q = q.lower().strip()
    if q in amap:
        return amap[q]
    subs = {v for a, v in amap.items() if q in a}
    if len(subs) == 1:
        return subs.pop()
    close = difflib.get_close_matches(q, list(amap.keys()), n=1, cutoff=0.6)
    return amap[close[0]] if close else None


def find_league(qa, qb):
    cur = current_season()
    for code in LEAGUES:
        ms = load_season(code, cur) + load_season(code, cur - 1)
        amap = alias_map(ms)
        a, b = resolve(qa, amap), resolve(qb, amap)
        if a and b and a != b:
            return code, a, b
    return None, None, None


# ---------------------------------------------------------------- the model
def pois(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def rates(hist, team, avgH, avgA, k=5):
    hs = hc = hw = as_ = ac = aw = 0.0
    for m in hist:
        if m["h"] == team:
            hs += m["w"] * m["hg"]; hc += m["w"] * m["ag"]; hw += m["w"]
        elif m["a"] == team:
            as_ += m["w"] * m["ag"]; ac += m["w"] * m["hg"]; aw += m["w"]
    return {  # shrunk toward league average so thin data doesn't mislead
        "att_h": (hs + k * avgH) / (hw + k) / avgH,
        "def_h": (hc + k * avgA) / (hw + k) / avgA,
        "att_a": (as_ + k * avgA) / (aw + k) / avgA,
        "def_a": (ac + k * avgH) / (aw + k) / avgH,
    }


def analyze(code, home, away):
    hist = load_history(code)
    today = date.today()
    for m in hist:
        m["w"] = 0.5 ** ((today - m["d"]).days / 365)  # recent games count more
    sw = sum(m["w"] for m in hist)
    avgH = sum(m["w"] * m["hg"] for m in hist) / sw
    avgA = sum(m["w"] * m["ag"] for m in hist) / sw
    rh, ra = rates(hist, home, avgH, avgA), rates(hist, away, avgH, avgA)
    lh = avgH * rh["att_h"] * ra["def_a"]
    la = avgA * ra["att_a"] * rh["def_h"]

    pH = pD = pA = over = btts = 0.0
    scores = []
    for i in range(11):
        for j in range(11):
            p = pois(i, lh) * pois(j, la)
            scores.append((p, i, j))
            if i > j: pH += p
            elif i == j: pD += p
            else: pA += p
            if i + j > 2: over += p
            if i > 0 and j > 0: btts += p
    tot = pH + pD + pA
    pH, pD, pA, over, btts = [x / tot for x in (pH, pD, pA, over, btts)]
    scores.sort(reverse=True)

    # head-to-head from loaded seasons (either venue)
    h2h = sorted([m for m in hist if {m["h"], m["a"]} == {home, away}],
                 key=lambda m: m["d"], reverse=True)
    hw = dw = aw_ = 0
    for m in h2h:
        hg = m["hg"] if m["h"] == home else m["ag"]
        ag = m["ag"] if m["h"] == home else m["hg"]
        if hg > ag: hw += 1
        elif hg == ag: dw += 1
        else: aw_ += 1
    n = len(h2h)
    if n >= 3:  # small nudge from H2H, capped at 20%
        b = min(0.2, 0.04 * n)
        pH = (1 - b) * pH + b * hw / n
        pD = (1 - b) * pD + b * dw / n
        pA = (1 - b) * pA + b * aw_ / n

    def form(team):
        games = sorted([m for m in hist if team in (m["h"], m["a"])],
                       key=lambda m: m["d"], reverse=True)[:5]
        s, pts, gf, ga = "", 0, 0, 0
        for m in games:
            f, a = (m["hg"], m["ag"]) if m["h"] == team else (m["ag"], m["hg"])
            gf += f; ga += a
            s += "W" if f > a else "D" if f == a else "L"
            pts += 3 if f > a else 1 if f == a else 0
        return s, pts, gf, ga

    return dict(code=code, home=home, away=away, pH=pH, pD=pD, pA=pA,
                lh=lh, la=la, over=over, btts=btts, scores=scores[:3],
                h2h=h2h[:5], h2h_n=n, h2h_rec=(hw, dw, aw_),
                form_h=form(home), form_a=form(away), n_matches=len(hist))


def fmt_full(r):
    L = [f"{r['home']} vs {r['away']} ({r['code']})", "",
         "WIN PROBABILITY",
         f"{r['home']}: {r['pH']:.0%}",
         f"Draw: {r['pD']:.0%}",
         f"{r['away']}: {r['pA']:.0%}", "",
         f"Expected goals: {r['lh']:.2f} - {r['la']:.2f}",
         "Likely scores: " + ", ".join(f"{i}-{j} ({p:.0%})" for p, i, j in r["scores"]),
         f"Over 2.5 goals: {r['over']:.0%} | Both teams score: {r['btts']:.0%}", ""]
    for who, f in ((r["home"], r["form_h"]), (r["away"], r["form_a"])):
        L.append(f"Form {who}: {f[0] or 'n/a'} ({f[1]} pts, {f[2]} scored, {f[3]} conceded)")
    L.append("")
    if r["h2h_n"]:
        hw, dw, aw_ = r["h2h_rec"]
        L.append(f"HEAD TO HEAD (last {r['h2h_n']} in data): "
                 f"{r['home']} {hw}W, {dw}D, {r['away']} {aw_}W")
        for m in r["h2h"]:
            L.append(f"  {m['d']} {m['h']} {m['hg']}-{m['ag']} {m['a']}")
    else:
        L.append("HEAD TO HEAD: no meetings in the loaded seasons")
    L += ["", f"Based on {r['n_matches']} league matches, recent ones weighted more.",
          "Estimate only: injuries, lineups and motivation are not included."]
    return "\n".join(L)


def fmt_line(r):
    return (f"{r['home']} vs {r['away']}: "
            f"{r['pH']:.0%} / {r['pD']:.0%} / {r['pA']:.0%}")


# ---------------------------------------------------------------- bot logic
def handle(text):
    text = text.strip()
    if text.startswith(("/start", "/help")):
        return ("Send a match like:\nArsenal vs Chelsea\nPL: Man City vs Liverpool\n\n"
                "/today - probabilities for today's fixtures\n"
                f"Leagues: {', '.join(LEAGUES)}")
    if text.startswith("/today"):
        lines = []
        d = date.today().isoformat()
        for code in LEAGUES:
            try:
                data = api(f"/competitions/{code}/matches",
                           {"dateFrom": d, "dateTo": d}, ttl=3600)
            except requests.HTTPError:
                continue
            for m in data.get("matches", []):
                try:
                    r = analyze(code, m["homeTeam"]["name"], m["awayTeam"]["name"])
                    lines.append(fmt_line(r))
                except Exception:
                    pass
        return ("Today (home / draw / away):\n" + "\n".join(lines)) if lines \
            else "No fixtures found today in your leagues."

    forced = None
    mt = re.match(r"^([A-Za-z0-9]{2,3})\s*:\s*(.+)$", text)
    if mt and mt.group(1).upper() in LEAGUES:
        forced, text = mt.group(1).upper(), mt.group(2)
    parts = re.split(r"\s+(?:vs\.?|v|x|-)\s+", text, flags=re.I)
    if len(parts) != 2:
        return "Send it as: Team A vs Team B"
    qa, qb = parts
    if forced:
        amap = alias_map(load_season(forced, current_season()) + load_season(forced, current_season() - 1))
        a, b = resolve(qa, amap), resolve(qb, amap)
        code = forced if a and b else None
    else:
        code, a, b = find_league(qa, qb)
    if not code:
        return "Couldn't find both teams in your leagues. Check spelling or add the league code, e.g. PL: Arsenal vs Chelsea"
    return fmt_full(analyze(code, a, b))


def tg(method, **kw):
    return requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/{method}",
                         data=kw, timeout=70).json()


def reply(chat, text):
    for i in range(0, len(text), 3900):
        tg("sendMessage", chat_id=chat, text=text[i:i + 3900])


def main():
    print("Bot running...")
    offset = 0
    while True:
        try:
            res = tg("getUpdates", offset=offset, timeout=50)
            for u in res.get("result", []):
                offset = u["update_id"] + 1
                msg = u.get("message") or {}
                text, chat = msg.get("text"), msg.get("chat", {}).get("id")
                if not text or (ALLOWED and str(chat) != ALLOWED):
                    continue
                tg("sendMessage", chat_id=chat, text="Analysing...")
                try:
                    reply(chat, handle(text))
                except Exception as e:
                    reply(chat, f"Error: {e}")
        except Exception as e:
            print("Loop error:", e)
            time.sleep(5)


if __name__ == "__main__":
    main()
