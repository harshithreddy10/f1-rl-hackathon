# Project Constitution — F1 RL Strategy Optimizer

## 1. Core Principles
* **Scope Discipline:** Adhere strictly to the single-driver constraint (Carlos Sainz) for historical races (Monaco 2023). Do not introduce multi-agent interactions, full-grid simulations, or live race outcome predictions.
* **Explainability:** The system is built for hackathon judges and F1 fans. The Streamlit visualizations and time delta outputs are just as critical as the underlying DQN agent. The AI's decisions must not be a black box.
* **Fallback Readiness:** Always maintain a working demo state. Prioritize saving model weights (`tyre_model.pkl`, `dqn_f1_agent.zip`) so the demo can run offline instantly if training fails to converge live.

## 2. Coding Standards
* **Language & Stack:** Python 3.10+ only. Limit primary external dependencies to `fastf1`, `scikit-learn`, `gymnasium`, `stable-baselines3`, `streamlit`, and `plotly`.
* **Typing:** Mandatory type hints for all function signatures, methods, and complex data structures to ensure the Gym environment and Streamlit frontend pass data seamlessly.
* **Naming Conventions:** Standard PEP 8 conventions. Use `snake_case` for variables/functions (e.g., `lap_time_sec`) and `PascalCase` for classes (e.g., `F1RaceEnv`).
* **Documentation:** All classes and core functions must include concise docstrings explaining their purpose, inputs, and outputs.

## 3. Machine Learning & RL Rules
* **Reproducibility First:** Always set explicit random seeds (e.g., `seed=42`) for the `gymnasium` environment, `stable-baselines3` models, and `scikit-learn` splits to ensure consistent demo results.
* **Strict Data Boundaries:** Respect the train/test split (Laps 1–45 train, Laps 46–57 test). Ensure zero data leakage when training the tyre degradation model and validating the RL agent.
* **Modular Architecture:** The Gym environment (`F1RaceEnv`) must remain decoupled from the raw data pipeline. It must only ingest the clean, pre-processed `race_data.csv` and the serialized `tyre_model.pkl`.

## 4. AI Execution Directives
* **Sequential Execution:** Strictly follow the ordered implementation plan in `tasks.md`. Do not start RL agent tasks until the data pipeline and tyre model are fully validated.
* **Validation Checkpoints:** You must run `check_env()` on the custom Gym environment and fix all validation errors before initializing the Stable-Baselines3 agent.
* **No Dummy Data:** Rely exclusively on the FastF1 historical data. Do not generate or mock telemetry data unless the API completely fails.