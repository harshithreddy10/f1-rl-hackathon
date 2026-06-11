import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from env.f1_env import F1PitEnv, TOTAL_LAPS
from agents.rule_agent import RuleBasedAgent

os.makedirs("visualization", exist_ok=True)

DARK_BG = "#0d0d1a"
CARD_BG = "#1a1a2e"
RED     = "#E8002D"
YELLOW  = "#FFF200"
WHITE   = "#f0f0f0"
ACCENT  = "#00d2ff"


def run_rule_agent(weather="dry"):
    env   = F1PitEnv(weather=weather)
    obs, _ = env.reset()
    lap = 0
    while True:
        # 2-stop: pit lap 25 → HARD, lap 55 → SOFT
        if lap == 25:
            action = 3
        elif lap == 55:
            action = 1
        else:
            action = 0
        obs, _, term, _, _ = env.step(action)
        lap += 1
        if term:
            break
    return env


def run_dqn_agent(weather="dry"):
    try:
        from stable_baselines3 import DQN
        model = DQN.load("models/dqn_f1_agent")
        env   = F1PitEnv(weather=weather)
        obs, _ = env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, _, _ = env.step(int(action))
            if term:
                break
        return env
    except Exception as e:
        print(f"⚠️  DQN model not found ({e}), using rule-based as fallback")
        return run_rule_agent(weather)


def plot_comparison(weather="dry"):
    rule_env = run_rule_agent(weather)
    dqn_env  = run_dqn_agent(weather)

    laps = list(range(1, TOTAL_LAPS + 1))

    fig = plt.figure(figsize=(18, 10), facecolor=DARK_BG)
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle(
        f"F1 Pit Strategy — RL Agent vs Baseline  |  Monaco 2024  |  Weather: {weather.upper()}",
        fontsize=15, color=WHITE, fontweight="bold", y=0.98
    )

    # ── Plot 1: Lap times ─────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor(CARD_BG)
    ax1.plot(laps, rule_env.lap_times, color=YELLOW, linewidth=2,
             label="Rule-Based (2-stop)", marker="o", ms=2)
    ax1.plot(laps, dqn_env.lap_times, color=ACCENT, linewidth=2,
             label="DQN Agent (1-stop)", marker="s", ms=2)

    for pit_lap in rule_env.pit_laps:
        ax1.axvline(pit_lap, color=YELLOW, alpha=0.4, linestyle="--", linewidth=1)
    for pit_lap in dqn_env.pit_laps:
        ax1.axvline(pit_lap, color=ACCENT, alpha=0.4, linestyle="--", linewidth=1)

    _style_ax(ax1, "Lap", "Lap Time (s)", "Lap Time Per Lap")
    ax1.legend(facecolor=CARD_BG, labelcolor=WHITE, fontsize=10)

    # ── Plot 2: Cumulative gap ─────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(CARD_BG)
    rule_cum = np.cumsum(rule_env.lap_times)
    dqn_cum  = np.cumsum(dqn_env.lap_times)
    gap      = rule_cum - dqn_cum
    ax2.fill_between(laps, gap, 0, where=(gap >= 0), color=ACCENT, alpha=0.5, label="DQN ahead")
    ax2.fill_between(laps, gap, 0, where=(gap < 0),  color=RED,   alpha=0.5, label="Rule ahead")
    ax2.plot(laps, gap, color=WHITE, linewidth=1.5)
    ax2.axhline(0, color="#555", linewidth=1)
    _style_ax(ax2, "Lap", "Time Gap (s)", "Cumulative Gap")
    ax2.legend(facecolor=CARD_BG, labelcolor=WHITE, fontsize=9)

    # ── Plot 3: Tyre strategy bars ─────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.set_facecolor(CARD_BG)
    comp_colors = {"SOFT": RED, "MEDIUM": YELLOW, "HARD": WHITE}

    def draw_strategy(env, y, label):
        boundaries = [1] + sorted(env.pit_laps) + [TOTAL_LAPS + 1]
        compounds  = env.compounds_used
        while len(compounds) < len(boundaries) - 1:
            compounds.append(compounds[-1])
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end   = boundaries[i + 1] - 1
            comp  = compounds[i]
            color = comp_colors.get(comp, "#888")
            ax3.barh(y, end - start + 1, left=start, height=0.35,
                     color=color, edgecolor="#000", linewidth=0.5)
            ax3.text(start + (end - start) / 2, y, comp[:1],
                     ha="center", va="center", fontsize=8,
                     color="#000", fontweight="bold")
        ax3.text(0.5, y, label, ha="right", va="center",
                 color=WHITE, fontsize=10,
                 transform=ax3.get_yaxis_transform())

    draw_strategy(rule_env, 0.7, "Rule")
    draw_strategy(dqn_env,  0.3, " DQN")

    ax3.set_xlim(0, TOTAL_LAPS + 1)
    ax3.set_ylim(0, 1)
    ax3.set_xlabel("Lap", color=WHITE)
    ax3.set_title("Tyre Strategy Timeline", color=WHITE,
                  fontsize=12, fontweight="bold")
    ax3.set_yticks([])
    for spine in ax3.spines.values():
        spine.set_edgecolor("#444")
    ax3.tick_params(colors=WHITE)

    # ── Plot 4: Scoreboard ────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor(CARD_BG)
    ax4.axis("off")

    rule_total = rule_env.total_time
    dqn_total  = dqn_env.total_time
    delta      = rule_total - dqn_total

    lines = [
        ("RULE-BASED",   f"{rule_total:.1f}s",  YELLOW),
        ("DQN AGENT",    f"{dqn_total:.1f}s",   ACCENT),
        ("",             "",                     WHITE),
        ("TIME SAVED",   f"{delta:+.1f}s",       ACCENT if delta > 0 else RED),
        ("DQN PITS",     str(len(dqn_env.pit_laps)),   WHITE),
        ("RULE PITS",    str(len(rule_env.pit_laps)),   WHITE),
        ("DQN PIT LAP",  str(dqn_env.pit_laps),         ACCENT),
    ]
    for i, (label, val, color) in enumerate(lines):
        y = 0.92 - i * 0.13
        ax4.text(0.05, y, label, transform=ax4.transAxes,
                 color="#aaa", fontsize=9)
        ax4.text(0.95, y, val, transform=ax4.transAxes,
                 color=color, fontsize=11, fontweight="bold", ha="right")
    ax4.set_title("📊 Scoreboard", color=WHITE,
                  fontsize=12, fontweight="bold")
    for spine in ax4.spines.values():
        spine.set_edgecolor("#444")

    out = f"visualization/comparison_{weather}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"✅ Saved → {out}")
    plt.show()
    return fig


def _style_ax(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, color=WHITE, fontsize=10)
    ax.set_ylabel(ylabel, color=WHITE, fontsize=10)
    ax.set_title(title, color=WHITE, fontsize=12, fontweight="bold")
    ax.tick_params(colors=WHITE)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.grid(True, alpha=0.15, color="gray")


if __name__ == "__main__":
    print("🎨 Generating dry race comparison...")
    plot_comparison("dry")
    print("\n🎨 Generating wet race comparison...")
    plot_comparison("wet")
    print("\n✅ All plots saved to visualization/")