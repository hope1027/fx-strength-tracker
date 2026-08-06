"""
貨幣強弱快照腳本 — 搭配 GitHub Actions 定時執行

每次執行會做這幾件事：
 1. 呼叫 Twelve Data 的 quote API，一次拿回 27 個「XXX/USD」報價
 2. 換算成每個貨幣「以美元計價的價值」(vUSD)
 3. 讀取歷史快照（data/snapshots.csv），找出：
      - 今天最早一筆快照 → 當日基準
      - 昨天同一時段最接近的一筆 → 跨日基準
 4. 用一籃子平均法（跟其餘 27 個貨幣比）算出每個貨幣的相對強弱分數
    （當日動能 / 跨日趨勢 各算一組）
 5. 找出最強 3 名、最弱 3 名，寫入 results/latest.md 與 results/history.csv
 6. 把這次快照寫回 data/snapshots.csv，供下一次執行比對基準

環境變數（用 GitHub Secrets 設定，不要寫死在程式碼裡）：
 TWELVE_DATA_API_KEY   必填，去 twelvedata.com 免費申請
 TELEGRAM_BOT_TOKEN    選填，設定後每次執行會推播摘要到 Telegram
 TELEGRAM_CHAT_ID      選填，搭配上面一起用
"""

import os
import csv
import math
import time
import datetime
import pathlib
import requests

# ---------------- 可自行調整的部分 ----------------
CURRENCIES = [
    "EUR", "JPY", "GBP", "CNY", "AUD", "CAD", "CHF", "HKD", "SGD", "KRW",
    "INR", "MXN", "BRL", "ZAR", "SEK", "NOK", "DKK", "PLN", "THB", "IDR",
    "TWD", "ILS", "TRY", "RUB", "SAR", "AED", "MYR", "NZD",
]  # 27 個非美元貨幣，加上 USD 本身 = 28 個
# 注意：HKD、DKK、SAR、AED 屬於盯住匯率(peg)貨幣，日常波動極小，
# 如果要讓強弱排名更乾淨反映市場動能，可以在這份清單裡先拿掉，
# 或是保留但在挑選最強/最弱時另外標註排除。
# ---------------------------------------------------

DATA_DIR = pathlib.Path("data")
RESULTS_DIR = pathlib.Path("results")
SNAPSHOT_CSV = DATA_DIR / "snapshots.csv"
HISTORY_CSV = RESULTS_DIR / "history.csv"

API_KEY = os.environ["TWELVE_DATA_API_KEY"]
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID")


CHUNK_SIZE = 7  # 免費方案每分鐘只有 8 次額度，抓 7 個留一點緩衝
WAIT_BETWEEN_BATCHES = 65  # 秒，故意等超過 1 分鐘再打下一批，避免踩到 429


def fetch_vusd():
    """分批呼叫 Twelve Data，取得每個貨幣兌美元的價格（= vUSD）。
    免費方案有「每分鐘 8 次」的額度限制，一次把 27 個 symbol
    塞進同一個請求會直接超過限制被擋（429），所以拆成小批，
    批次之間刻意等待超過 1 分鐘。"""
    url = "https://api.twelvedata.com/quote"
    vusd = {"USD": 1.0}

    batches = [CURRENCIES[i:i + CHUNK_SIZE] for i in range(0, len(CURRENCIES), CHUNK_SIZE)]
    for idx, batch in enumerate(batches):
        symbols = ",".join(f"{c}/USD" for c in batch)
        resp = requests.get(url, params={"symbol": symbols, "apikey": API_KEY}, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        # 只查一個 symbol 時 Twelve Data 回傳單一物件；多個 symbol 時回傳 {symbol: {...}}
        if "symbol" in data:
            data = {data["symbol"]: data}

        for c in batch:
            key = f"{c}/USD"
            item = data.get(key)
            if item and "close" in item:
                vusd[c] = float(item["close"])
            else:
                print(f"⚠️ 沒有拿到 {key} 的報價，本次先跳過")

        if idx < len(batches) - 1:
            print(f"已抓完第 {idx + 1}/{len(batches)} 批，等 {WAIT_BETWEEN_BATCHES} 秒再抓下一批…")
            time.sleep(WAIT_BETWEEN_BATCHES)

    return vusd


def load_snapshots():
    if not SNAPSHOT_CSV.exists():
        return []
    with open(SNAPSHOT_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_snapshot(ts_iso, vusd):
    DATA_DIR.mkdir(exist_ok=True)
    is_new = not SNAPSHOT_CSV.exists()
    fieldnames = ["timestamp"] + sorted(vusd.keys())
    with open(SNAPSHOT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            w.writeheader()
        row = {"timestamp": ts_iso, **{k: f"{v:.8f}" for k, v in vusd.items()}}
        w.writerow(row)


def find_baseline(snapshots, mode, now):
    """mode='today' 找今天最早一筆；mode='yesterday' 找昨天最接近同一時間的一筆"""
    today = now.date()
    yesterday = today - datetime.timedelta(days=1)
    candidates = []
    for row in snapshots:
        ts = datetime.datetime.fromisoformat(row["timestamp"])
        if mode == "today" and ts.date() == today:
            candidates.append((ts, row))
        elif mode == "yesterday" and ts.date() == yesterday:
            candidates.append((ts, row))
    if not candidates:
        return None
    if mode == "today":
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    candidates.sort(
        key=lambda x: abs(
            (x[0].hour * 60 + x[0].minute) - (now.hour * 60 + now.minute)
        )
    )
    return candidates[0][1]


def relative_change_scores(baseline_row, vusd_now):
    """跟基準快照比，算每個貨幣這段期間的一籃子相對強弱變化（單位：指數點）"""
    if not baseline_row:
        return None
    changes = {}
    for c, v_now in vusd_now.items():
        v_base = baseline_row.get(c)
        if v_base is None:
            continue
        v_base = float(v_base)
        if v_base <= 0:
            continue
        changes[c] = math.log(v_now / v_base)

    codes = list(changes.keys())
    scores = {}
    for c in codes:
        others = [changes[o] for o in codes if o != c]
        scores[c] = (changes[c] - sum(others) / len(others)) * 100 if others else 0.0
    return scores


def top_bottom(scores, n=7):
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:n], ranked[-n:][::-1]


def write_results(ts_iso, today_scores, yday_scores):
    RESULTS_DIR.mkdir(exist_ok=True)

    def fmt_block(title, scores):
        if not scores:
            return f"### {title}\n（沒有足夠的歷史資料可比較，之後幾次執行會自動補齊）\n"
        top, bottom = top_bottom(scores)
        lines = [f"### {title}", "", "**最強 7 名**", ""]
        lines += [f"- {c}：{s:+.3f}" for c, s in top]
        lines.append("")
        lines.append("**最弱 7 名**")
        lines.append("")
        lines += [f"- {c}：{s:+.3f}" for c, s in bottom]
        return "\n".join(lines) + "\n"

    content = f"# 貨幣強弱快照 — {ts_iso}\n\n"
    content += fmt_block("當日動能（跟今天最早一筆比）", today_scores) + "\n"
    content += fmt_block("跨日趨勢（跟昨天同時段比）", yday_scores)

    (RESULTS_DIR / "latest.md").write_text(content, encoding="utf-8")

    is_new = not HISTORY_CSV.exists()
    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["timestamp", "basis", "rank", "currency", "score"])
        for basis, scores in (("today", today_scores), ("yesterday", yday_scores)):
            if not scores:
                continue
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for i, (c, s) in enumerate(ranked, start=1):
                w.writerow([ts_iso, basis, i, c, f"{s:.4f}"])

    return content


def notify_telegram(text):
    if not (TG_TOKEN and TG_CHAT):
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": TG_CHAT, "text": text[:4000]}, timeout=15)


def main():
    now = datetime.datetime.now(datetime.timezone.utc)
    ts_iso = now.replace(microsecond=0).isoformat()

    vusd = fetch_vusd()
    snapshots = load_snapshots()

    today_scores = relative_change_scores(find_baseline(snapshots, "today", now), vusd)
    yday_scores = relative_change_scores(find_baseline(snapshots, "yesterday", now), vusd)

    append_snapshot(ts_iso, vusd)
    content = write_results(ts_iso, today_scores, yday_scores)
    notify_telegram(content)

    print(content)


if __name__ == "__main__":
    main()
