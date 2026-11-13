"""
Brewers study, step 1: pull a season (regular season + postseason) with the
extra columns hitting analysis needs: strike zone, count, batter side, bat tracking,
spray location, and which half-inning (to know the batting team).
Uses the pybaseball cache where possible.
"""
import sys
from pybaseball import statcast, cache
cache.enable()

# Usage: python3 01_pull_hitting.py 2026   (or 2025)
YEAR = sys.argv[1] if len(sys.argv) > 1 else "2026"
START, END = f"{YEAR}-03-01", f"{YEAR}-11-05"   # regular season through the World Series
df = statcast(start_dt=START, end_dt=END)

wanted = [
    "game_date", "game_pk", "game_type", "home_team", "away_team", "inning_topbot",
    "batter", "pitcher", "stand", "p_throws", "balls", "strikes",
    "pitch_type", "release_speed", "zone", "plate_x", "plate_z",
    "description", "events", "bb_type", "launch_speed", "launch_angle", "launch_speed_angle",
    "hc_x", "hc_y", "woba_value", "woba_denom", "estimated_woba_using_speedangle",
    "bat_speed", "swing_length",
]
missing = [c for c in wanted if c not in df.columns]
if missing:
    print("Columns not available in this pull (skipped):", missing)
df = df[[c for c in wanted if c in df.columns]]
print(f"Rows: {len(df):,}")
print(df.game_type.value_counts().to_string())
df.to_parquet(f"statcast_{YEAR}_hitting.parquet", index=False)
print(f"Saved statcast_{YEAR}_hitting.parquet")
