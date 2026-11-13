"""
Breakout study: which 2025 stats flagged hitters who broke out in 2026?
Uses statcast_2025_hitting.parquet and statcast_2026_hitting.parquet (regular season).

Breakout = 2026 wOBA beat what his 2025 wOBA predicted (regression-to-the-mean adjusted).
Tests ~20 candidate 2025 stats: raw signal, and signal BEYOND expected stats (xwOBA).
Then builds a 2027 breakout watch list from 2026 data.
Outputs: breakout_signals.csv, breakouts_2026.csv, watch_list_2027.csv, chart_breakout_signals.png
"""
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MIN_PA_BEFORE, MIN_PA_AFTER = 150, 200
# Usage: python3 02_breakout_predictors.py 2025 2026   (or 2024 2025 to validate)
Y1 = sys.argv[1] if len(sys.argv) > 2 else "2025"
Y2 = sys.argv[2] if len(sys.argv) > 2 else "2026"
TAG = f"{Y1}_{Y2}"
SWING = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play",
         "foul_bunt", "missed_bunt", "bunt_foul_tip"}
WHIFF = {"swinging_strike", "swinging_strike_blocked", "foul_tip", "missed_bunt"}

def hitter_stats(path):
    d = pd.read_parquet(path)
    d = d[d.game_type == "R"].copy()
    d["swing"] = d.description.isin(SWING) | d.description.str.startswith("hit_into_play", na=False)
    d["whiff"] = d.description.isin(WHIFF)
    d["in_zone"] = d.zone.between(1, 9)
    pa = d[d.events.notna() & (d.woba_denom > 0)].copy()
    # xwOBA per PA: expected value on balls in play, actual value otherwise (K, BB, HBP)
    pa["xwoba_pa"] = np.where(pa.estimated_woba_using_speedangle.notna(), pa.estimated_woba_using_speedangle, pa.woba_value)
    bip = d[d.bb_type.notna() & d.launch_speed.notna()]
    spray = np.degrees(np.arctan2(bip.hc_x - 125.42, 198.27 - bip.hc_y))
    pull = ((bip.stand == "R") & (spray < -15)) | ((bip.stand == "L") & (spray > 15))
    bip = bip.assign(pulled_air=pull & bip.bb_type.isin(["line_drive", "fly_ball"]))
    sw = d[d.swing]
    g_pa, g_bip, g_sw = pa.groupby("batter"), bip.groupby("batter"), sw.groupby("batter")
    s = pd.DataFrame({
        "PA": g_pa.size(),
        "woba": g_pa.woba_value.sum() / g_pa.woba_denom.sum(),
        "xwoba": g_pa.xwoba_pa.sum() / g_pa.woba_denom.sum(),
        "k_rate": g_pa.events.apply(lambda e: e.isin(["strikeout", "strikeout_double_play"]).mean()),
        "bb_rate": g_pa.events.apply(lambda e: (e == "walk").mean()),
        "avg_exit_velo": g_bip.launch_speed.mean(),
        "ev90": g_bip.launch_speed.quantile(0.9),
        "max_exit_velo": g_bip.launch_speed.max(),
        "hard_hit_rate": g_bip.launch_speed.apply(lambda v: (v >= 95).mean()),
        "barrel_rate": g_bip.launch_speed_angle.apply(lambda v: (v == 6).mean()),
        "sweet_spot_rate": g_bip.launch_angle.apply(lambda v: v.between(8, 32).mean()),
        "avg_launch_angle": g_bip.launch_angle.mean(),
        "ld_rate": g_bip.bb_type.apply(lambda v: (v == "line_drive").mean()),
        "gb_rate": g_bip.bb_type.apply(lambda v: (v == "ground_ball").mean()),
        "pulled_air_rate": g_bip.pulled_air.mean(),
        "whiff_rate": g_sw.whiff.mean(),
        "zone_contact_rate": 1 - d[d.swing & d.in_zone].groupby("batter").whiff.mean(),
        "chase_rate": d[~d.in_zone & d.zone.notna()].groupby("batter").swing.mean(),
    })
    if "bat_speed" in d:
        bs = d[d.bat_speed.notna()].groupby("batter")
        s["bat_speed"] = bs.bat_speed.mean()
        s["fast_swing_rate"] = bs.bat_speed.apply(lambda v: (v >= 75).mean())
        s["swing_length"] = bs.swing_length.mean()
    s["luck_gap"] = s.xwoba - s.woba            # positive = results lagged contact quality
    return s

h25, h26 = hitter_stats(f"statcast_{Y1}_hitting.parquet"), hitter_stats(f"statcast_{Y2}_hitting.parquet")
df = h25[h25.PA >= MIN_PA_BEFORE].join(h26[h26.PA >= MIN_PA_AFTER][["PA", "woba", "chase_rate"]], rsuffix="_26", how="inner")
df = df.rename(columns={"woba_26": "woba26", "PA_26": "PA26", "chase_rate_26": "chase26"})
print(f"SEASONS: {Y1} -> {Y2}")
print(f"Hitters with {MIN_PA_BEFORE}+ PA in {Y1} and {MIN_PA_AFTER}+ PA in {Y2}: {len(df)}")

# ---------- 1. Breakout = beat the regression-based expectation ----------
b = np.polyfit(df.woba, df.woba26, 1)
df["expected_26"] = np.polyval(b, df.woba)
df["breakout_score"] = df.woba26 - df.expected_26
# Beyond expected stats: what's left after also using 2025 xwOBA
X = np.column_stack([np.ones(len(df)), df.woba, df.xwoba])
coef, *_ = np.linalg.lstsq(X, df.woba26, rcond=None)
df["beyond_xwoba"] = df.woba26 - X @ coef

# ---------- 2. Test each 2025 stat ----------
candidates = [c for c in df.columns if c not in
              ("PA", "woba", "PA26", "woba26", "chase26", "expected_26", "breakout_score", "beyond_xwoba")]
n = len(df)
rows = []
for c in candidates:
    x = df[c]
    ok = x.notna()
    r1 = np.corrcoef(x[ok], df.breakout_score[ok])[0, 1]
    r2 = np.corrcoef(x[ok], df.beyond_xwoba[ok])[0, 1]
    se = 1 / np.sqrt(ok.sum() - 3)
    lo, hi = np.tanh(np.arctanh(r2) - 1.96 * se), np.tanh(np.arctanh(r2) + 1.96 * se)
    rows.append({"stat_2025": c, "r_with_breakout": r1, "r_beyond_xwoba": r2,
                 "beyond_ci_low": lo, "beyond_ci_high": hi, "significant": not (lo < 0 < hi)})
sig = pd.DataFrame(rows).set_index("stat_2025").sort_values("r_beyond_xwoba", key=abs, ascending=False)
print(f"\nWHICH {Y1} STATS FLAGGED {Y2} BREAKOUTS?")
print("r_with_breakout: raw signal | r_beyond_xwoba: signal after accounting for expected stats (the 'overlooked' test)")
print(sig.round(3).to_string())
sig.round(4).to_csv(f"breakout_signals_{TAG}.csv")
print("\nFAST SWING RATE CHECK: r_beyond_xwoba = "
      f"{sig.loc['fast_swing_rate', 'r_beyond_xwoba']:.3f} (95% CI {sig.loc['fast_swing_rate', 'beyond_ci_low']:.3f} to {sig.loc['fast_swing_rate', 'beyond_ci_high']:.3f})"
      if "fast_swing_rate" in sig.index else "\nfast_swing_rate not available for these seasons")

# ---------- 3. Who broke out, and what did they look like in 2025? ----------
try:
    from pybaseball import playerid_reverse_lookup
    def names_for(ids):
        nm = playerid_reverse_lookup(list(ids), key_type="mlbam")
        nm["name"] = nm.name_first.str.title() + " " + nm.name_last.str.title()
        return nm.set_index("key_mlbam")["name"]
except Exception:
    def names_for(ids):
        return pd.Series([str(i) for i in ids], index=list(ids))

top = [c for c in ["fast_swing_rate", "max_exit_velo", "zone_contact_rate"] if c in df.columns]
pct = df[top].rank(pct=True)
bo = df.sort_values("breakout_score", ascending=False).head(15).copy()
bo.insert(0, "name", names_for(bo.index).reindex(bo.index).values)
for c in top:
    bo[f"{c}_pctile_{Y1}"] = (pct.loc[bo.index, c] * 100).round()
print(f"\nTOP {Y2} BREAKOUTS (beat expectation by the most):")
print(bo[["name", "woba", "woba26", "breakout_score"] + [f"{c}_pctile_{Y1}" for c in top]].round(3).to_string(index=False))
bo.round(4).to_csv(f"breakouts_{Y2}.csv")

# ---------- 4. Your idea: does discipline unlock the ceiling? ----------
if "fast_swing_rate" in df.columns:
    df["chase_change"] = df.chase26 - df.chase_rate                    # negative = became more disciplined
    league_change = df.chase_change.median()
    df["high_ceiling"] = df.fast_swing_rate >= df.fast_swing_rate.median()
    df["got_disciplined"] = df.chase_change < league_change            # cut chase more than the typical hitter
    df["already_disciplined"] = df.chase_rate <= df.chase_rate.median()

    def cell_table(col, labels):
        t = df.groupby(["high_ceiling", col]).breakout_score.agg(["mean", "count", "std"])
        t["ci95"] = 1.96 * t["std"] / np.sqrt(t["count"])
        t.index = [f"{'High' if h else 'Low'} fast-swing rate | {labels[d]}" for h, d in t.index]
        return t[["mean", "ci95", "count"]].rename(columns={"mean": "avg_breakout_score"})

    print(f"\nYOUR IDEA (coaching version): breakout score by ceiling x change in chase rate, {Y1} -> {Y2}")
    print(cell_table("got_disciplined", {True: "cut chase", False: "did not cut chase"}).round(3).to_string())
    print(f"\nSCOUTING VERSION: breakout score by ceiling x {Y1} chase rate")
    print(cell_table("already_disciplined", {True: "low chase", False: "high chase"}).round(3).to_string())

    # Regression with an interaction term (standardized), with standard errors
    z = lambda v: (v - v.mean()) / (v.std() if v.std() > 0 else 1)
    d2 = df[["breakout_score", "fast_swing_rate", "chase_change"]].dropna()
    fsr, dch = z(d2.fast_swing_rate), z(-d2.chase_change)              # flip sign: higher = more discipline gained
    Xi = np.column_stack([np.ones(len(d2)), fsr, dch, fsr * dch])
    beta, *_ = np.linalg.lstsq(Xi, d2.breakout_score, rcond=None)
    resid = d2.breakout_score - Xi @ beta
    s2 = (resid ** 2).sum() / (len(d2) - Xi.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(Xi.T @ Xi)))
    reg = pd.DataFrame({"coef": beta, "se": se, "t": beta / se},
                       index=["intercept", "fast_swing_rate", "discipline_gained", "fast_swing x discipline"])
    print("\nREGRESSION: breakout score ~ fast swing rate + discipline gained + interaction (|t| > 2 ~ significant)")
    print(reg.round(4).to_string())
    pd.concat([cell_table("got_disciplined", {True: "cut chase", False: "did not cut chase"}),
               cell_table("already_disciplined", {True: "low chase", False: "high chase"})]).round(4).to_csv(f"discipline_test_{TAG}.csv")

# ---------- 4b. Watch list: elite fast swings, results haven't caught up ----------
if "fast_swing_rate" in h26.columns:
    cur = h26[h26.PA >= MIN_PA_AFTER].copy()
    elite = cur.fast_swing_rate >= cur.fast_swing_rate.quantile(0.75)
    lagging = cur.woba <= cur.woba.median()
    watch = cur[elite & lagging].sort_values("fast_swing_rate", ascending=False).head(20).copy()
    watch.insert(0, "name", names_for(watch.index).reindex(watch.index).values)
    nxt = int(Y2) + 1
    print(f"\n{nxt} WATCH LIST: top-25% fast swing rate, but wOBA at or below the median in {Y2}")
    print("(lower chase rate = closer to unlocking it, per the discipline idea)")
    print(watch[["name", "PA", "woba", "xwoba", "fast_swing_rate", "chase_rate", "zone_contact_rate"]].round(3).to_string(index=False))
    watch.round(4).to_csv(f"watch_list_{nxt}.csv")

# ---------- 5. Chart ----------
show = sig.head(12).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 5.2))
yy = np.arange(len(show))
ax.barh(yy + 0.18, show.r_with_breakout, height=0.36, color="#9aa9c4", label="Raw signal")
ax.barh(yy - 0.18, show.r_beyond_xwoba, height=0.36, color="#333366", label="Beyond expected stats")
ax.errorbar(show.r_beyond_xwoba, yy - 0.18,
            xerr=[show.r_beyond_xwoba - show.beyond_ci_low, show.beyond_ci_high - show.r_beyond_xwoba],
            fmt="none", ecolor="#999999", capsize=3)
ax.set_yticks(yy); ax.set_yticklabels([s.replace("_", " ") for s in show.index])
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel(f"Correlation of {Y1} stat with {Y2} breakout (95% interval on dark bars)")
ax.set_title(f"Which {Y1} stats flagged {Y2} breakouts? (n = {n} hitters)")
ax.legend(frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(f"chart_breakout_signals_{TAG}.png", dpi=200)
print(f"\nSaved CSVs and chart_breakout_signals_{TAG}.png")
