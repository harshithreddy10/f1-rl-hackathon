import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="F1 Pit Strategy RL", layout="wide", page_icon="🏎️")

st.title("🏎️ F1 Pit Stop Strategy — Reinforcement Learning")
st.markdown("*Monaco 2024 · DQN Agent vs Rule-Based Baseline*")

# ── Sidebar ──────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
weather = st.sidebar.radio("Weather", ["Dry", "Wet"])
total_laps = st.sidebar.slider("Race Laps", 10, 20, 15)

# ── Mock tyre degradation chart ───────────────────────
st.subheader("📈 Tyre Degradation Model")
st.caption("Trained on real Monaco 2024 lap data via FastF1")

fig, ax = plt.subplots(figsize=(10, 4))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#1a1a2e")

ages = np.arange(1, 40)
# Mock curves (will be replaced by real model in Phase 2)
ax.plot(ages, 81 + ages * 0.08 + (ages**2) * 0.003, color="#E8002D", label="SOFT", linewidth=2.5)
ax.plot(ages, 82 + ages * 0.05 + (ages**2) * 0.002, color="#FFF200", label="MEDIUM", linewidth=2.5)
ax.plot(ages, 83 + ages * 0.03 + (ages**2) * 0.001, color="#CCCCCC", label="HARD", linewidth=2.5)

ax.tick_params(colors="white")
ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")
ax.title.set_color("white")
for spine in ax.spines.values():
    spine.set_edgecolor("#444")
ax.set_xlabel("Tyre Age (laps)")
ax.set_ylabel("Lap Time (s)")
ax.legend(facecolor="#2a2a4e", labelcolor="white")
ax.grid(True, alpha=0.2, color="gray")
st.pyplot(fig)

# ── Mock agent comparison ─────────────────────────────
st.subheader("🤖 DQN Agent vs Rule-Based Baseline")
col1, col2, col3 = st.columns(3)
col1.metric("Rule-Based Total Time", "1243.4s")
col2.metric("DQN Agent Total Time", "1238.1s")
col3.metric("Time Saved", "5.3s", delta="-5.3s", delta_color="inverse")

st.info("⏳ RL agent training in progress — results will update after Phase 4")

# ── Status tracker ────────────────────────────────────
st.subheader("🔧 Build Progress")
st.success("✅ Phase 1 — Monaco 2024 data loaded (770 laps)")
st.warning("⏳ Phase 2 — Tyre degradation model (in progress)")
st.error("❌ Phase 3 — Gym environment (not started)")
st.error("❌ Phase 4 — DQN training (not started)")