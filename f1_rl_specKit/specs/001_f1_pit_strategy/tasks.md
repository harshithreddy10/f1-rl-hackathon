# Tasks — Ordered Implementation Plan

> Tasks marked **[P]** can run in parallel with the previous task.

---

## Phase 1 — Data (Hours 0–8)

- [ ] **T-01** Install and verify all dependencies (`fastf1`, `gymnasium`, `stable-baselines3`, `plotly`, `streamlit`)
- [ ] **T-02** Load Monaco 2023 Race via FastF1, inspect raw DataFrame columns
- [ ] **T-03** Extract and clean lap features → save `race_data.csv` to disk
- [ ] **T-04 [P]** Load Monza 2023 Race as validation dataset → save `monza_data.csv`
- [ ] **T-05** Encode compound column (Soft=0, Medium=1, Hard=2), handle nulls, verify no data leakage
- [ ] **T-06** Plot tyre degradation curves per compound (exploratory) — confirm degradation signal exists

---

## Phase 2 — Tyre Model (Hours 8–14)

- [ ] **T-07** Train GradientBoostingRegressor on Monaco lap data
- [ ] **T-08** Evaluate model: R² score on test split — must be > 0.80 to proceed
- [ ] **T-09** If R² < 0.80: try XGBRegressor, add fuel load proxy feature (lap number)
- [ ] **T-10** Serialize trained model → `tyre_model.pkl`
- [ ] **T-11 [P]** Plot predicted vs. actual lap times — sanity check visual

---

## Phase 3 — RL Environment (Hours 14–20)

- [ ] **T-12** Implement `F1RaceEnv(gymnasium.Env)` with observation space, action space, reset, step
- [ ] **T-13** Implement reward function: `-lap_time - pit_penalty + position_bonus`
- [ ] **T-14** Load `tyre_model.pkl` inside env for lap time prediction
- [ ] **T-15** Run `check_env(env)` — fix all validation errors before proceeding
- [ ] **T-16** Manual episode test: step through 15 laps with random actions, verify no crashes
- [ ] **T-17** Implement FIA compound rule: episode ends with penalty if only 1 compound used

---

## Phase 4 — RL Agent (Hours 20–30)

- [ ] **T-18** Initialize DQN agent with `MlpPolicy` on `F1RaceEnv`
- [ ] **T-19** Train for 50,000 timesteps, log episode rewards every 1,000 steps
- [ ] **T-20** Plot reward curve — verify it's trending upward (learning signal)
- [ ] **T-21** Save trained agent → `dqn_f1_agent.zip`
- [ ] **T-22** Run trained agent on full 57-lap Monaco episode, record pit decisions
- [ ] **T-23 [P]** Implement rule-based baseline agent (pit lap 18 → Medium, lap 38 → Hard)
- [ ] **T-24** Compare agent vs. baseline vs. real Sainz strategy — compute total time deltas

---

## Phase 5 — Visualization (Hours 30–40)

- [ ] **T-25** Build Streamlit app scaffold with 3-tab layout
- [ ] **T-26** Tab 1: Tyre degradation chart — predicted lap time vs tyre age per compound
- [ ] **T-27** Tab 2: Strategy comparison bar chart — agent vs. Sainz stints and compounds
- [ ] **T-28** Tab 3: Cumulative time delta line chart — agent vs. real, lap by lap
- [ ] **T-29 [P]** Add weather toggle (Dry/Wet) — rerun agent inference on wet conditions
- [ ] **T-30** End-to-end demo run: open Streamlit, walk through all 3 tabs, time it (target < 5 min)

---

## Phase 6 — Polish & Pitch (Hours 40–48)

- [ ] **T-31** Fix any UI bugs found in T-30 demo run
- [ ] **T-32** Pre-cache all data and pre-load model weights so demo starts instantly
- [ ] **T-33** Record backup screen capture video of working demo
- [ ] **T-34** Write 5-minute pitch script (see `demo-script.md`)
- [ ] **T-35** Create 5-slide pitch deck: Problem → Solution → Demo → Results → Team
- [ ] **T-36** Full rehearsal — one person presents, others time and give feedback

---

## Dependency Graph

```
T-01 → T-02 → T-03 → T-07 → T-12 → T-18 → T-25
              T-04 ↗          ↑               ↑
              T-05 ──────────T-14           T-22
              T-06                          T-23
```
