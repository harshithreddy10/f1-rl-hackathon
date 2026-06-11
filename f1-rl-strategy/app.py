import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="F1 Pit Strategy RL", layout="wide", page_icon="🏎️")

# ── Header ────────────────────────────────────────────
st.title("🏎️ F1 Pit Stop Strategy — Reinforcement Learning")
st.markdown("*Monaco 2024 · DQN Agent vs Rule-Based Baseline*")
st.divider()

# ── Sidebar ───────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
weather = st.sidebar.radio("Weather", ["dry", "wet"]).lower()
st.sidebar.divider()
st.sidebar.markdown("### 📊 Model Info")
st.sidebar.success("✅ Tyre Model MAE: 0.776s")
st.sidebar.success("✅ DQN Training: 200K steps")
st.sidebar.info("🏁 Race: Monaco 2024 (78 laps)")

# ── Load models & run agents ──────────────────────────
@st.cache_resource
def load_everything():
    from models.tyre_model import load_model, predict_lap_time
    from env.f1_env import F1PitEnv, TOTAL_LAPS
    from agents.rule_agent import RuleBasedAgent
    from stable_baselines3 import DQN
    return load_model, predict_lap_time, F1PitEnv, TOTAL_LAPS, RuleBasedAgent, DQN

@st.cache_data
def run_simulation(weather):
    from env.f1_env import F1PitEnv, TOTAL_LAPS
    from agents.rule_agent import RuleBasedAgent
    from stable_baselines3 import DQN

    # Rule agent
    rule_env = F1PitEnv(weather=weather)
    agent = RuleBasedAgent()
    obs, _ = rule_env.reset()
    while True:
        action = agent.act(obs, rule_env)
        obs, _, term, _, _ = rule_env.step(action)
        if term:
            break

    # DQN agent
    dqn_env = F1PitEnv(weather=weather)
    model = DQN.load("models/dqn_f1_agent")
    obs, _ = dqn_env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, _, _ = dqn_env.step(int(action))
        if term:
            break

    return rule_env, dqn_env

# ── Run simulation ─────────────────────────────────────
with st.spinner("🔄 Running simulation..."):
    try:
        rule_env, dqn_env = run_simulation(weather)
        simulation_ok = True
    except Exception as e:
        st.error(f"Simulation error: {e}")
        simulation_ok = False

if simulation_ok:
    TOTAL_LAPS = len(rule_env.lap_times)
    laps = list(range(1, TOTAL_LAPS + 1))

    # ── Scoreboard ─────────────────────────────────────
    st.subheader("📊 Race Results")
    col1, col2, col3, col4 = st.columns(4)
    delta = rule_env.total_time - dqn_env.total_time
    col1.metric("Rule-Based Total", f"{rule_env.total_time:.1f}s")
    col2.metric("DQN Agent Total",  f"{dqn_env.total_time:.1f}s")
    col3.metric("Time Saved by DQN", f"{delta:+.1f}s",
                delta=f"{delta:+.1f}s", delta_color="inverse")
    col4.metric("DQN Pit Laps", str(dqn_env.pit_laps))
    st.divider()

    # ── Lap time chart ──────────────────────────────────
    st.subheader("📈 Lap Time Comparison")
    DARK_BG = "#0d0d1a"
    CARD_BG = "#1a1a2e"
    WHITE   = "#f0f0f0"
    YELLOW  = "#FFF200"
    ACCENT  = "#00d2ff"
    RED     = "#E8002D"

    fig, ax = plt.subplots(figsize=(14, 4), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.plot(laps, rule_env.lap_times, color=YELLOW, linewidth=1.8,
            label="Rule-Based (2-stop)", marker="o", ms=2)
    ax.plot(laps, dqn_env.lap_times,  color=ACCENT, linewidth=1.8,
            label="DQN Agent (1-stop)", marker="s", ms=2)
    for p in rule_env.pit_laps:
        ax.axvline(p, color=YELLOW, alpha=0.4, linestyle="--", linewidth=1)
    for p in dqn_env.pit_laps:
        ax.axvline(p, color=ACCENT, alpha=0.4, linestyle="--", linewidth=1)
    ax.set_xlabel("Lap", color=WHITE)
    ax.set_ylabel("Lap Time (s)", color=WHITE)
    ax.tick_params(colors=WHITE)
    ax.legend(facecolor=CARD_BG, labelcolor=WHITE)
    ax.grid(True, alpha=0.15, color="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    st.pyplot(fig)

    # ── Cumulative gap ──────────────────────────────────
    st.subheader("⏱️ Cumulative Time Gap (Rule - DQN)")
    fig2, ax2 = plt.subplots(figsize=(14, 3), facecolor=DARK_BG)
    ax2.set_facecolor(CARD_BG)
    gap = np.cumsum(rule_env.lap_times) - np.cumsum(dqn_env.lap_times)
    ax2.fill_between(laps, gap, 0, where=(gap >= 0),
                     color=ACCENT, alpha=0.5, label="DQN ahead")
    ax2.fill_between(laps, gap, 0, where=(gap < 0),
                     color=RED, alpha=0.5, label="Rule ahead")
    ax2.plot(laps, gap, color=WHITE, linewidth=1.5)
    ax2.axhline(0, color="#555", linewidth=1)
    ax2.set_xlabel("Lap", color=WHITE)
    ax2.set_ylabel("Gap (s)", color=WHITE)
    ax2.tick_params(colors=WHITE)
    ax2.legend(facecolor=CARD_BG, labelcolor=WHITE)
    ax2.grid(True, alpha=0.15, color="gray")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#444")
    st.pyplot(fig2)

    # ── Tyre strategy ───────────────────────────────────
    st.subheader("🏁 Tyre Strategy Timeline")
    comp_colors = {"SOFT": RED, "MEDIUM": YELLOW, "HARD": WHITE}

    fig3, ax3 = plt.subplots(figsize=(14, 2.5), facecolor=DARK_BG)
    ax3.set_facecolor(CARD_BG)

    def draw_strategy(env, y, label):
        boundaries = [1] + sorted(env.pit_laps) + [TOTAL_LAPS + 1]
        compounds  = list(env.compounds_used)
        while len(compounds) < len(boundaries) - 1:
            compounds.append(compounds[-1])
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end   = boundaries[i+1] - 1
            comp  = compounds[i]
            color = comp_colors.get(comp, "#888")
            ax3.barh(y, end-start+1, left=start, height=0.35,
                     color=color, edgecolor="#000", linewidth=0.5)
            ax3.text(start+(end-start)/2, y, comp[:1],
                     ha="center", va="center", fontsize=9,
                     color="#000", fontweight="bold")
        ax3.text(0.5, y, label, ha="right", va="center",
                 color=WHITE, fontsize=10,
                 transform=ax3.get_yaxis_transform())

    draw_strategy(rule_env, 0.7, "Rule")
    draw_strategy(dqn_env,  0.3, " DQN")
    ax3.set_xlim(0, TOTAL_LAPS+1)
    ax3.set_ylim(0, 1)
    ax3.set_xlabel("Lap", color=WHITE)
    ax3.set_yticks([])
    ax3.tick_params(colors=WHITE)
    for spine in ax3.spines.values():
        spine.set_edgecolor("#444")
    st.pyplot(fig3)

    # ── Tyre degradation curves ─────────────────────────
    st.divider()
    st.subheader("📉 Tyre Degradation Model")
    from models.tyre_model import load_model, predict_lap_time
    tyre_model, le = load_model()

    fig4, ax4 = plt.subplots(figsize=(14, 4), facecolor=DARK_BG)
    ax4.set_facecolor(CARD_BG)
    colors = {"SOFT": RED, "MEDIUM": YELLOW, "HARD": WHITE}
    ages = np.arange(1, 55)
    for compound, color in colors.items():
        times = [predict_lap_time(tyre_model, le, compound, age, 40) for age in ages]
        ax4.plot(ages, times, color=color, linewidth=2.5, label=compound)
    ax4.set_xlabel("Tyre Age (laps)", color=WHITE)
    ax4.set_ylabel("Predicted Lap Time (s)", color=WHITE)
    ax4.tick_params(colors=WHITE)
    ax4.legend(facecolor=CARD_BG, labelcolor=WHITE)
    ax4.grid(True, alpha=0.2, color="gray")
    for spine in ax4.spines.values():
        spine.set_edgecolor("#444")
    st.pyplot(fig4)

    # ── Raw data ────────────────────────────────────────
    st.divider()
    with st.expander("🔍 View Raw Lap Data"):
        df = pd.DataFrame({
            "Lap": laps,
            "Rule LapTime (s)": [f"{t:.2f}" for t in rule_env.lap_times],
            "DQN LapTime (s)":  [f"{t:.2f}" for t in dqn_env.lap_times],
        })
        st.dataframe(df, use_container_width=True)