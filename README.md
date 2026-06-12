# 🏎️ F1 Pit Stop Strategy — Reinforcement Learning

> An RL-powered F1 pit stop strategy optimizer trained on real Monaco 2024 race data.
> The DQN agent discovers a 1-stop strategy saving **47.7 seconds** over a rule-based baseline.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project uses Deep Q-Network (DQN) reinforcement learning to optimize F1 pit stop strategies using real telemetry data from the Monaco 2024 Grand Prix via FastF1.

The agent learns when to pit and which tyre compound to use by simulating thousands of race scenarios, ultimately outperforming a rule-based baseline strategy.

---

## Features

- 📊 **Real F1 Data** — Monaco 2024 race telemetry via FastF1
- 🧠 **DQN Agent** — Trained with Stable-Baselines3
- 📈 **Tyre Degradation Model** — GradientBoosting regression (MAE: 0.776s)
- 🌐 **Multi-language Support** — English, Hindi, Telugu
- 🤖 **AI Strategist** — Powered by Gemini, Anthropic, or local Ollama
- 🔑 **BYOK Support** — Bring Your Own Key for cloud AI
- 🖥️ **Local AI** — Ollama integration for privacy-first inference

---

## Project Structure

```
f1-rl-strategy/
├── data/
│   └── load_data.py          # FastF1 data loader
├── models/
│   ├── tyre_model.py         # Tyre degradation regression
│   ├── tyre_model.pkl        # Trained model
│   └── dqn_f1_agent.zip      # Trained DQN agent
├── env/
│   └── f1_env.py             # Custom Gymnasium environment
├── agents/
│   ├── rule_agent.py         # Rule-based baseline
│   └── train_dqn.py          # DQN training script
├── strategist/
│   └── ai_strategist.py      # AI backend (Gemini/Anthropic/Ollama)
├── visualization/
│   └── compare_agents.py     # Comparison plots
├── app.py                    # Streamlit dashboard
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/harshithreddy10/f1-rl-hackathon.git
cd f1-rl-hackathon/f1-rl-strategy

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run the Streamlit App
```bash
streamlit run app.py
```

### Run Phase by Phase
```bash
# Phase 1: Load Monaco 2024 data
python data/load_data.py

# Phase 2: Train tyre degradation model
python models/tyre_model.py

# Phase 3: Test gym environment
python env/f1_env.py

# Phase 4: Train DQN agent (~45 mins)
python agents/train_dqn.py

# Phase 5: Generate comparison plots
python visualization/compare_agents.py
```

### Local AI with Ollama
```bash
# Install Ollama
brew install ollama

# Pull model
ollama pull llama3

# Start server
ollama serve

# Then run the app and select Ollama in sidebar
streamlit run app.py
```

---

## How It Works

```
Real Monaco 2024 Data
        ↓
Tyre Model learns lap time vs tyre age
        ↓
RL Environment simulates 78-lap race
        ↓
DQN Agent trains for 200K steps
        ↓
Agent finds optimal pit timing
```

### Observation Space (8 features)
| Feature | Description |
|---------|-------------|
| lap_progress | Current lap / total laps |
| tyre_age | Normalised tyre life |
| compound | SOFT=0, MEDIUM=0.5, HARD=1 |
| fuel_load | Remaining fuel |
| track_temp | Track temperature |
| weather | Dry=0, Wet=1 |
| gap_to_leader | Race gap |
| safety_car | Safety car deployed |

### Action Space (4 discrete)
| Action | Meaning |
|--------|---------|
| 0 | Stay out |
| 1 | Pit → SOFT |
| 2 | Pit → MEDIUM |
| 3 | Pit → HARD |

---

## Results

| Strategy | Total Time | Pit Stops | Pit Lap |
|----------|-----------|-----------|---------|
| Rule-Based (2-stop) | 6226.1s | 2 | 25, 60 |
| **DQN Agent (1-stop)** | **6178.4s** | **1** | **38** |
| **Time Saved** | **+47.7s** | | |

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

This project is licensed under the **GNU Affero General Public License v3.0**.
See [LICENSE](LICENSE) for details.
