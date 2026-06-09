# Plan — Technical Blueprint

## Stack
| Layer | Tool | Reason |
|---|---|---|
| Data | `fastf1` | Free, official F1 telemetry, 2018–2024 |
| ML Model | `scikit-learn` / `xgboost` | Tyre degradation regression |
| RL Env | `gymnasium` | Standard interface, compatible with SB3 |
| RL Agent | `stable-baselines3` DQN | Fast to train, good for discrete actions |
| Visualization | `plotly` + `streamlit` | Interactive charts, easy to demo |
| Language | Python 3.10+ | Team proficiency |

---

## Architecture — 3 Modules

### Module 1: Data Pipeline (`data/`)
- Load Monaco 2023 Race via FastF1
- Extract per-lap features: `LapNumber`, `TyreLife`, `Compound`, `LapTime`, `TrackStatus`
- Encode `Compound`: Soft=0, Medium=1, Hard=2
- Export clean DataFrame as `race_data.csv` (offline cache)
- Train/test split: laps 1–45 train, laps 46–57 test

### Module 2: Tyre Degradation Model (`model/`)
- **Input:** `[compound_encoded, tyre_age, lap_number]`
- **Output:** predicted lap time in seconds
- **Algorithm:** GradientBoostingRegressor (sklearn) or XGBRegressor
- **Target metric:** R² > 0.80 on test split
- **Artifact:** `tyre_model.pkl` (serialized, loaded by env)

### Module 3: RL Environment + Agent (`rl/`)

**Environment (`F1RaceEnv`):**
- Observation: `[lap, tyre_age, compound, gap_ahead, gap_behind, safety_car]` — shape (6,)
- Action space: Discrete(4) — stay / pit+soft / pit+medium / pit+hard
- Reward: `-lap_time - pit_penalty + position_bonus`
- Episode: 15 laps (training), 57 laps (evaluation)
- Termination: lap >= total_laps OR 3+ pit stops (hard cap)

**Agent:**
- Algorithm: DQN (`MlpPolicy`, 2-layer MLP)
- Training: 50,000 timesteps (~15 min on CPU)
- Hyperparameters: lr=1e-3, buffer=10000, exploration_fraction=0.3
- Artifact: `dqn_f1_agent.zip`

---

## Visualization (`viz/`)
Streamlit app with 3 panels:

1. **Tyre Degradation Chart** — lap time vs tyre age per compound (from model)
2. **Strategy Comparison** — stacked bar chart of compounds per stint, agent vs. real
3. **Cumulative Time Delta** — line chart showing agent vs. Sainz time gap per lap

---

## Data Model

```
RaceLap:
  lap_number: int
  tyre_life: int
  compound: str        # "SOFT" | "MEDIUM" | "HARD"
  lap_time_sec: float
  track_status: int    # 1=green, 4=safety car, 5=VSC
  is_pit_lap: bool
  pit_compound: str | None
```

---

## Risk Register
| Risk | Likelihood | Mitigation |
|---|---|---|
| RL doesn't converge | Medium | Pre-train before hackathon, bring saved weights |
| FastF1 API unavailable | Low | Cache `race_data.csv` before hackathon |
| Gym env bugs | Medium | Validate with `check_env()` in Hour 14 |
| Training too slow | Low | Use 15-lap episodes, not 57 |
| Demo crashes | Low | Record backup video of working demo |
