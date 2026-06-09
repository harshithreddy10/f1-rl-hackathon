# Research Notes

## FastF1 Key Fields
| Field | Description | Used In |
|---|---|---|
| `LapTime` | Lap duration (timedelta) | Tyre model target |
| `TyreLife` | Laps on current set | State + model feature |
| `Compound` | SOFT / MEDIUM / HARD / WET | State + action |
| `TrackStatus` | 1=green, 4=SC, 5=VSC | State variable |
| `PitInTime` | Time of pit entry | Pit lap detection |
| `LapNumber` | Lap in race | Fuel load proxy |

## Monaco 2023 — Carlos Sainz Actual Strategy
- Start: MEDIUM
- Pit 1: Lap 22 → HARD
- Pit 2: Lap 43 → SOFT
- Finish: P2

This is the ground truth. Agent must produce a competitive total time.

## RL Reference
- DQN paper: Mnih et al. 2015 (discrete action spaces)
- SB3 DQN docs: https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
- Gymnasium custom env guide: https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/

## Similar Work
- "F1 Strategy Simulation" — various Kaggle notebooks using FastF1
- No public RL-based F1 strategy optimizer found (novelty confirmed)
