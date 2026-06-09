# Spec — F1 Pit Stop Strategy Optimizer

## Problem Statement
F1 race outcomes are heavily influenced by pit stop strategy — when to pit and which tyre compound to switch to. Teams with better strategy consistently gain positions. Real teams spend millions on proprietary strategy software. There is no open, accessible tool that learns optimal strategy from historical race data.

## Who Is This For?
- Hackathon judges evaluating AI innovation in sports
- F1 fans who want to understand strategy decisions
- ML/DS practitioners exploring RL in real-world domains

---

## User Stories

### Core

**US-01 — Strategy Comparison**
> As a judge, I want to see the RL agent's pit strategy side-by-side with the real team's actual strategy, so I can immediately understand the value the system adds.

**US-02 — Race Simulation**
> As a user, I want the system to simulate a race lap-by-lap, so I can see how tyre degradation and pit decisions affect total race time.

**US-03 — Agent Decision Visibility**
> As a user, I want to see WHAT the agent decided each lap (stay out / pit + compound), so the system is interpretable, not a black box.

**US-04 — Time Delta Output**
> As a user, I want to see how many seconds the agent saved (or lost) vs. the real strategy, so I have a concrete performance metric.

### Stretch

**US-05 — Weather Toggle**
> As a demo presenter, I want to switch between Dry and Wet conditions live, so I can show the agent adapting its strategy in real time.

**US-06 — Multi-Race Generalization**
> As a judge, I want to see the agent tested on a race it wasn't trained on, so I know it learned a general strategy, not just memorized one race.

---

## What the System Must Do

1. Load real F1 race telemetry (lap times, tyre compound, tyre age, pit laps)
2. Model tyre degradation — predict lap time given compound and tyre age
3. Simulate a race environment where an agent can make pit/stay decisions each lap
4. Train an agent that learns to minimize total race time
5. Compare the agent's strategy against the real team strategy from the same race
6. Visualize the comparison in a clear, non-technical way

## What the System Must NOT Do
- Simulate all 20 drivers (single driver only for hackathon scope)
- Predict live race outcomes (historical races only)
- Replace commercial strategy software (this is a proof-of-concept)

---

## Success Criteria
- Agent completes a full race simulation without errors
- Agent strategy differs meaningfully from a random/naive baseline
- Visualization is understandable to a non-technical judge in under 60 seconds
- Demo runs end-to-end in under 5 minutes
