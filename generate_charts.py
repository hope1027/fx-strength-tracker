"""
從 data/snapshots.csv 算出貨幣強弱指數，並把 1H / 4H / 1D 三種週期
聚合成 K 線（OHLC），畫成圖片輸出。

輸出：
 charts/strength_1H.png
 charts/strength_4H.png
 charts/strength_1D.png

每張圖只畫「目前排名最強 7 名 + 最弱 7 名」共 14 個貨幣（左欄最強、
右欄最弱），避免 28 個貨幣全部畫在一起太雜亂、看不出重點。

聚合方式：因為原始資料是每小時一筆的離散快照，不是連續報價，
所以用「收盤價序列」去近似 OHLC——
  開盤 = 這段期間第一筆強弱分數
  收盤 = 這段期間最後一筆強弱分數
  最高 = 這段期間分數的最大值
  最低 = 這段期間分數的最小值
這是很常見的近似做法，影線會比真實報價保守一點，但方向、型態
的判讀完全夠用。
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SNAPSHOT_CSV = pathlib.Path("data/snapshots.csv")
CHART_DIR = pathlib.Path("charts")
SCALE = 9000
MAX_CANDLES = 30  # 每張小圖最多顯示幾根 K 棒，太多根會太擠

# (顯示用標籤, 要連續幾筆快照湊成一根 K 棒)
TIMEFRAMES = [("1H", 1), ("4H", 4), ("1D", 24)]


def load_strength():
    """讀取快照，算出每個時間點、每個貨幣的強弱分數。

    重點：不能直接拿各貨幣「原始數值的等級」互相比較——因為不同貨幣
    天生換算單位差異極大（例如 1 GBP≈1.3 美元，但 1 IDR≈0.00006 美元），
    這個量級差異幾乎不隨時間變化，會完全蓋掉真正的強弱波動。
    正確做法是先算出每個貨幣「相對自己最早一筆快照」的漲跌幅（log
    報酬率），再用這個漲跌幅去跟其他貨幣的漲跌幅做籃子平均。"""
    df = pd.read_csv(SNAPSHOT_CSV, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    currencies = list(df.columns)
    log_v = np.log(df[currencies].astype(float))
    log_return = log_v.sub(log_v.iloc[0])  # 相對第一筆快照的漲跌幅（log）
    strength = log_return.sub(log_return.mean(axis=1), axis=0) * SCALE
    return strength


def ohlc_by_count(series, n):
    """把『連續 n 筆快照』湊成一根 K 棒，用資料筆數切、不用日曆時間切。
    這樣就算排程時快時慢、中間漏跑過幾次，每根 K 棒依然是真正抓到的
    n 筆資料聚合出來的，不會因為某個時段剛好資料太少而變成沒有漲跌
    的扁平線。

    n=1（1H）：每根代表『這一筆比上一筆』的漲跌，開盤＝上一筆、
    收盤＝這一筆，這是資料本身的最小顆粒度，不會再更細了。
    n>1（4H／1D）：每 n 筆分成一組，開盤＝這組第一筆、收盤＝這組
    最後一筆、最高／最低＝這組裡的最大最小值。最後湊不滿 n 筆的
    尾巴會先丟掉，避免出現一根異常短小的 K 棒。"""
    series = series.dropna()
    if n == 1:
        o = series.shift(1)
        c = series
        high = pd.concat([o, c], axis=1).max(axis=1)
        low = pd.concat([o, c], axis=1).min(axis=1)
        return pd.DataFrame({"open": o, "high": high, "low": low, "close": c}).dropna()

    group_id = np.arange(len(series)) // n
    grouped = series.groupby(group_id)
    ohlc = pd.DataFrame({
        "open": grouped.first(),
        "high": grouped.max(),
        "low": grouped.min(),
        "close": grouped.last(),
    })
    complete = grouped.size() == n
    return ohlc[complete]


def pick_top_bottom(strength, n=7):
    latest = strength.iloc[-1].sort_values(ascending=False)
    return latest.head(n).index.tolist(), latest.tail(n).index.tolist()


def draw_candles(ax, ohlc, title):
    ohlc = ohlc.dropna().tail(MAX_CANDLES)
    if ohlc.empty:
        ax.set_title(f"{title}\n(not enough data yet)", fontsize=8)
        ax.axis("off")
        return
    for i, (_, row) in enumerate(ohlc.iterrows()):
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        up = c >= o
        color = "#26d97a" if up else "#ff4757"
        ax.plot([i, i], [l, h], color=color, linewidth=1)
        bottom = min(o, c)
        height = max(abs(c - o), 0.001)
        ax.add_patch(Rectangle((i - 0.3, bottom), 0.6, height, color=color))
    ax.set_title(title, fontsize=9)
    ax.axhline(0, color="#888888", linewidth=0.5, linestyle="--")
    ax.set_xlim(-1, len(ohlc))
    ax.set_xticks([])
    ax.tick_params(labelsize=7)


def render_timeframe(strength, label, n):
    top, bottom = pick_top_bottom(strength)
    codes = top + bottom

    fig, axes = plt.subplots(7, 2, figsize=(11, 16))
    fig.suptitle(f"Currency Strength — {label} (Left: Top 7 / Right: Bottom 7)", fontsize=13)

    for i, code in enumerate(codes):
        row_i, col_i = i % 7, i // 7
        ax = axes[row_i][col_i]
        ohlc = ohlc_by_count(strength[code], n)
        draw_candles(ax, ohlc, code)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    CHART_DIR.mkdir(exist_ok=True)
    fig.savefig(CHART_DIR / f"strength_{label}.png", dpi=130)
    plt.close(fig)


def main():
    if not SNAPSHOT_CSV.exists():
        print("還沒有 data/snapshots.csv，先讓排程多跑幾次再產圖")
        return
    strength = load_strength()
    for label, n in TIMEFRAMES:
        render_timeframe(strength, label, n)
        print(f"已產生 charts/strength_{label}.png")


if __name__ == "__main__":
    main()
