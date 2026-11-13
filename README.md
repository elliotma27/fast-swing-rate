# Fast Swing Rate: An Overlooked Signal of Future Hitting Success

**Elliot Malsam** · [elliotma27.github.io](https://elliotma27.github.io) · Statcast bat-tracking analysis, 2024 to 2026, in Python · [Full study (PDF)](Elliot_Malsam_Fast_Swing_Rate_Study.pdf)

**In short:** fast swing rate, the share of a hitter's swings at 75+ mph, flagged hitters who went on to beat their next-season projections in two straight season pairs, more consistently than average bat speed or exit velocity. Hitters who paired it with improved plate discipline did best of all.

## Why this question
Most bat-tracking discussion focuses on average bat speed and exit velocity. Fast swing rate gets far less attention. It measures how often a hitter reaches his top gear, which might say more about his ceiling than his average does.

## Method
For each hitter (150+ PA in year 1, 200+ in year 2), I projected next-season wOBA from this season's wOBA, so normal regression toward the mean is built in, then measured how much he beat that projection. I tested 20 prior-season stats, on their own and beyond what expected stats (xwOBA) already capture, on two separate season pairs: 2024 to 2025 (283 hitters) and 2025 to 2026 (276 hitters).

## Finding 1: Fast swing rate was the most consistent bat-tracking signal, and it beat average bat speed both years

| Prior-season stat (signal beyond xwOBA) | 2024 to 2025 | 2025 to 2026 |
|---|---|---|
| Fast swing rate (75+ mph swings) | +0.10 | +0.14 (significant) |
| Average bat speed | +0.09 | +0.07 |
| 90th-percentile exit velocity | +0.09 | +0.09 |
| Average exit velocity | +0.05 | +0.06 |
| Hard-hit rate | +0.06 | +0.03 |

![2024 to 2025](results/chart_breakout_signals_2024_2025.png)
![2025 to 2026](results/chart_breakout_signals_2025_2026.png)

## Finding 2: Plate discipline stacks on top of it
Hitters who cut their chase rate also beat their projections, independently of fast swing rate, in both season pairs (t = 2.8 and 3.8). The best results both years came from hitters with high fast swing rates who also cut their chase rate. The effects add together; a multiplying effect leaned positive both years but wasn't significant. Becoming more disciplined mattered, while already being disciplined did not, which points toward development rather than just scouting.

## Finding 3: It flags future success, not only classic breakouts
Many hitters it identifies are already great players (Aaron Judge, Giancarlo Stanton, Yordan Álvarez) who beat even their own strong projections. That's the point: fast swing rate identifies hitters whose ceiling sits above their current numbers, whether unproven or established. It also caught real breakouts: a watch list built only from 2025 data (top-quartile fast swing rate, below-median results) put 3 of its 20 hitters (Oneil Cruz, Jac Caglianone, Randal Grichuk) on the 2026 top-15 outperformers list, about three times what a random 20 would produce.

## 2027 watch list
Full list: [`results/watch_list_2027.csv`](results/watch_list_2027.csv)
- **Closest to unlocking** (elite fast swing rate, already low chase): Matt Chapman, Jorge Soler, Brett Baty, Taylor Trammell, Agustín Ramírez, Cam Smith
- **Development targets** (elite fast swing rate, high chase): Julio Rodríguez, Elías Díaz, Cal Raleigh, Jake Burger, Jo Adell, Spencer Jones

Many of these hitters, such as Vladimir Guerrero Jr., are established stars who aren't realistic targets for most teams, and some are rebound candidates rather than first-time breakouts. The real value is the profile itself: an elite fast swing rate paired with results that haven't caught up yet. The same screen can be applied to prospects, using minor league tracking data where it's available, to find this profile before it shows up in major league numbers.

## What this means for the Rockies
The Rockies were one of the most aggressive lineups in baseball in 2026. This suggests a concrete development approach: find hitters in the organization, including prospects, with elite fast swing rates but high chase rates, and prioritize plate discipline work with them. Their top gear is already there; the goal is to use it on better pitches.

## Caveats
Effects are modest; any single stat explains only a small share of year-to-year change. With 20 stats tested, some results are noise (line-drive and sweet-spot rates looked significantly negative in 2024 to 2025 but showed nothing in 2025 to 2026, so I set them aside). The discipline result measures change in the same season as the outperformance, so it shows association, not cause. Many hitters appear in both season pairs, and injury rebounds can look like breakouts.

## Reproduce
```
pip install -r requirements.txt
python3 01_pull_hitting.py 2024     # repeat for 2025 and 2026 (slow; uses the pybaseball cache)
python3 02_breakout_predictors.py 2024 2025
python3 02_breakout_predictors.py 2025 2026
```
