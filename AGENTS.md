# 🤖 AGENTS.md — AI Agents Documentation

This document describes all AI agents and models used in this project.

---

## 1. DQN Pit Stop Agent

### Overview
A Deep Q-Network (DQN) agent trained to make optimal pit stop decisions during an F1 race.

### Architecture
- **Algorithm:** DQN (Deep Q-Network)
- **Framework:** Stable-Baselines3
- **Policy:** MlpPolicy (Multi-layer Perceptron)
- **Training Steps:** 200,000
- **File:** `models/dqn_f1_agent.zip`

### Observation Space (8 features)
```python
[
    lap / total_laps,          # Race progress
    tyre_life / 50,            # Tyre age (normalised)
    compound_index / 2,        # Current compound
    fuel_load / 100,           # Remaining fuel
    track_temp / 60,           # Track temperature
    weather,                   # 0=dry, 1=wet
    gap_to_leader / 60,        # Position gap
    safety_car,                # 0=normal, 1=SC
]
```

### Action Space (4 discrete actions)
```
0 → Stay out
1 → Pit and fit SOFT tyres
2 → Pit and fit MEDIUM tyres
3 → Pit and fit HARD tyres
```

### Reward Function
```python
reward = -lap_time / 78.0     # Minimise race time

# Terminal rewards
if pit_count == 0:
    reward -= 500.0            # Penalise no-stop (rule violation)
elif compounds_used < 2:
    reward -= 300.0            # Penalise single compound
elif pit_count > 3:
    reward -= pit_count * 10   # Penalise excessive stops
else:
    reward += 100.0            # Reward valid clean strategy
```

### Training Hyperparameters
```python
learning_rate = 3e-4
buffer_size = 100_000
batch_size = 64
gamma = 0.99
exploration_fraction = 0.6
exploration_final_eps = 0.05
target_update_interval = 1000
```

### Performance
| Metric | Value |
|--------|-------|
| Race Time | 6178.4s |
| Pit Stops | 1 |
| Pit Lap | 38 |
| vs Baseline | -47.7s |

---

## 2. Rule-Based Baseline Agent

### Overview
A deterministic agent that pits at fixed laps — used as a benchmark.

### Strategy
- **Pit 1:** Lap 25 → HARD tyres
- **Pit 2:** Lap 60 → SOFT tyres
- **File:** `agents/rule_agent.py`

### Performance
| Metric | Value |
|--------|-------|
| Race Time | 6226.1s |
| Pit Stops | 2 |
| Pit Laps | 25, 60 |

---

## 3. Tyre Degradation Model

### Overview
A supervised ML model predicting lap time based on tyre state.

### Architecture
- **Algorithm:** GradientBoostingRegressor
- **Framework:** scikit-learn
- **File:** `models/tyre_model.pkl`

### Features
```python
[
    compound_encoded,   # Label encoded compound
    tyre_life,          # Laps on current tyre
    tyre_life ** 2,     # Non-linear degradation
    lap_number,         # Current lap
    pit_out,            # First lap after pit stop
]
```

### Performance
- **MAE:** 0.776 seconds
- **Training Data:** Monaco 2024, 770 laps, 10 drivers

---

## 4. AI Strategist

### Overview
A conversational AI that answers F1 strategy questions.
Supports multiple backends via BYOK or local inference.

### Backends
| Backend | Model | Type |
|---------|-------|------|
| Ollama | llama3 | Local |
| Gemini | gemini-1.5-flash | Cloud (free) |
| Anthropic | claude-haiku-4-5 | Cloud (paid) |

### System Prompt
```
You are an expert F1 pit stop strategist.
You have access to Monaco 2024 race data.
Our RL agent found that a 1-stop strategy (pit lap 38, MEDIUM→HARD)
saves 47.7 seconds over a 2-stop baseline.
Answer concisely and accurately.
```

### File
`strategist/ai_strategist.py`
