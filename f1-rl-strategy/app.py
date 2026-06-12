import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os

# ── Fix paths for Streamlit Cloud ────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

st.set_page_config(page_title="F1 Pit Strategy RL", layout="wide", page_icon="🏎️")

# ── Language Definitions ──────────────────────────────
LANGUAGES = {
    "English": {
        "title":        "🏎️ F1 Pit Stop Strategy — Reinforcement Learning",
        "subtitle":     "Monaco 2024 · DQN Agent vs Rule-Based Baseline",
        "weather":      "Weather",
        "model_info":   "📊 Model Info",
        "language":     "🌐 Language",
        "results":      "📊 Race Results",
        "rule_total":   "Rule-Based Total",
        "dqn_total":    "DQN Agent Total",
        "time_saved":   "Time Saved by DQN",
        "dqn_pit":      "DQN Pit Laps",
        "lap_chart":    "📈 Lap Time Comparison",
        "gap_chart":    "⏱️ Cumulative Time Gap (Rule - DQN)",
        "tyre_timeline":"🏁 Tyre Strategy Timeline",
        "tyre_deg":     "📉 Tyre Degradation Model",
        "raw_data":     "🔍 View Raw Lap Data",
        "lap_col":      "Lap",
        "rule_col":     "Rule LapTime (s)",
        "dqn_col":      "DQN LapTime (s)",
        "running":      "🔄 Running simulation...",
        "dqn_ahead":    "DQN ahead",
        "rule_ahead":   "Rule ahead",
        "rule_label":   "Rule-Based (2-stop)",
        "dqn_label":    "DQN Agent (1-stop)",
        "tyre_age":     "Tyre Age (laps)",
        "lap_time":     "Lap Time (s)",
        "gap_label":    "Gap (s)",
        "lap_label":    "Lap",
        "ai_title":     "🤖 Ask the AI Strategist",
        "ai_caption":   "Local AI via Ollama or Cloud AI with your own API key",
        "ai_quick":     "Quick questions:",
        "ai_q1":        "⏱️ When to pit?",
        "ai_q2":        "🌧️ Rain strategy?",
        "ai_q3":        "🏆 Why 1-stop wins?",
        "ai_input":     "Ask anything about F1 strategy:",
        "ai_placeholder":"e.g. Should I pit early or late at Monaco?",
        "ai_button":    "🏎️ Ask Strategist",
        "ai_answer":    "Answer",
        "ai_thinking":  "Thinking...",
    },
    "हिंदी (Hindi)": {
        "title":        "🏎️ F1 पिट स्टॉप रणनीति — रीइन्फोर्समेंट लर्निंग",
        "subtitle":     "मोनाको 2024 · DQN एजेंट vs नियम-आधारित बेसलाइन",
        "weather":      "मौसम",
        "model_info":   "📊 मॉडल जानकारी",
        "language":     "🌐 भाषा",
        "results":      "📊 रेस परिणाम",
        "rule_total":   "नियम-आधारित कुल",
        "dqn_total":    "DQN एजेंट कुल",
        "time_saved":   "DQN द्वारा बचाया समय",
        "dqn_pit":      "DQN पिट लैप्स",
        "lap_chart":    "📈 लैप टाइम तुलना",
        "gap_chart":    "⏱️ संचयी समय अंतर (नियम - DQN)",
        "tyre_timeline":"🏁 टायर रणनीति समयरेखा",
        "tyre_deg":     "📉 टायर क्षरण मॉडल",
        "raw_data":     "🔍 कच्चा लैप डेटा देखें",
        "lap_col":      "लैप",
        "rule_col":     "नियम लैप टाइम (s)",
        "dqn_col":      "DQN लैप टाइम (s)",
        "running":      "🔄 सिमुलेशन चल रहा है...",
        "dqn_ahead":    "DQN आगे",
        "rule_ahead":   "नियम आगे",
        "rule_label":   "नियम-आधारित (2-स्टॉप)",
        "dqn_label":    "DQN एजेंट (1-स्टॉप)",
        "tyre_age":     "टायर आयु (लैप्स)",
        "lap_time":     "लैप टाइम (s)",
        "gap_label":    "अंतर (s)",
        "lap_label":    "लैप",
        "ai_title":     "🤖 AI रणनीतिकार से पूछें",
        "ai_caption":   "Ollama के साथ लोकल AI या अपनी API key के साथ क्लाउड AI",
        "ai_quick":     "त्वरित प्रश्न:",
        "ai_q1":        "⏱️ पिट कब करें?",
        "ai_q2":        "🌧️ बारिश में रणनीति?",
        "ai_q3":        "🏆 1-स्टॉप क्यों जीतता है?",
        "ai_input":     "F1 रणनीति के बारे में कुछ भी पूछें:",
        "ai_placeholder":"जैसे: मोनाको में जल्दी या देर से पिट करें?",
        "ai_button":    "🏎️ रणनीतिकार से पूछें",
        "ai_answer":    "उत्तर",
        "ai_thinking":  "सोच रहा है...",
    },
    "తెలుగు (Telugu)": {
        "title":        "🏎️ F1 పిట్ స్టాప్ వ్యూహం — రీన్‌ఫోర్స్‌మెంట్ లెర్నింగ్",
        "subtitle":     "మొనాకో 2024 · DQN ఏజెంట్ vs నియమ-ఆధారిత బేస్‌లైన్",
        "weather":      "వాతావరణం",
        "model_info":   "📊 మోడల్ సమాచారం",
        "language":     "🌐 భాష",
        "results":      "📊 రేస్ ఫలితాలు",
        "rule_total":   "నియమ-ఆధారిత మొత్తం",
        "dqn_total":    "DQN ఏజెంట్ మొత్తం",
        "time_saved":   "DQN ఆదా చేసిన సమయం",
        "dqn_pit":      "DQN పిట్ ల్యాప్స్",
        "lap_chart":    "📈 ల్యాప్ టైమ్ పోలిక",
        "gap_chart":    "⏱️ సంచిత సమయ వ్యత్యాసం (నియమం - DQN)",
        "tyre_timeline":"🏁 టైర్ వ్యూహం టైమ్‌లైన్",
        "tyre_deg":     "📉 టైర్ క్షీణత మోడల్",
        "raw_data":     "🔍 రా ల్యాప్ డేటా చూడండి",
        "lap_col":      "ల్యాప్",
        "rule_col":     "నియమం ల్యాప్ టైమ్ (s)",
        "dqn_col":      "DQN ల్యాప్ టైమ్ (s)",
        "running":      "🔄 సిమ్యులేషన్ నడుస్తోంది...",
        "dqn_ahead":    "DQN ముందు",
        "rule_ahead":   "నియమం ముందు",
        "rule_label":   "నియమ-ఆధారిత (2-స్టాప్)",
        "dqn_label":    "DQN ఏజెంట్ (1-స్టాప్)",
        "tyre_age":     "టైర్ వయస్సు (ల్యాప్స్)",
        "lap_time":     "ల్యాప్ టైమ్ (s)",
        "gap_label":    "వ్యత్యాసం (s)",
        "lap_label":    "ల్యాప్",
        "ai_title":     "🤖 AI వ్యూహకర్తను అడగండి",
        "ai_caption":   "Ollama తో లోకల్ AI లేదా మీ API కీతో క్లౌడ్ AI",
        "ai_quick":     "త్వరిత ప్రశ్నలు:",
        "ai_q1":        "⏱️ పిట్ ఎప్పుడు?",
        "ai_q2":        "🌧️ వర్షంలో వ్యూహం?",
        "ai_q3":        "🏆 1-స్టాప్ ఎందుకు గెలుస్తుంది?",
        "ai_input":     "F1 వ్యూహం గురించి ఏదైనా అడగండి:",
        "ai_placeholder":"ఉదా: మొనాకోలో ముందు లేదా తర్వాత పిట్ చేయాలా?",
        "ai_button":    "🏎️ వ్యూహకర్తను అడగండి",
        "ai_answer":    "సమాధానం",
        "ai_thinking":  "ఆలోచిస్తోంది...",
    }
}

# ── Sidebar ───────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
lang    = st.sidebar.selectbox("🌐 Language / भाषा / భాష", list(LANGUAGES.keys()))
t       = LANGUAGES[lang]
weather = st.sidebar.radio(t["weather"], ["dry", "wet"]).lower()
st.sidebar.divider()
st.sidebar.markdown(f"### {t['model_info']}")
st.sidebar.success("✅ Tyre Model MAE: 0.776s")
st.sidebar.success("✅ DQN Training: 200K steps")
st.sidebar.info("🏁 Race: Monaco 2024 (78 laps)")

# ── AI Backend Sidebar ────────────────────────────────
st.sidebar.divider()
st.sidebar.markdown("### 🤖 AI Strategist")
ai_mode = st.sidebar.selectbox(
    "AI Backend",
    ["🖥️ Ollama (Local)", "🔑 Gemini (BYOK)", "🔑 Anthropic (BYOK)"]
)

api_key      = None
ollama_model = "llama3"

if ai_mode == "🖥️ Ollama (Local)":
    ollama_model = st.sidebar.text_input("Ollama Model", value="llama3")
    st.sidebar.info("Run `ollama serve` locally on port 11434")
elif ai_mode == "🔑 Gemini (BYOK)":
    api_key = st.sidebar.text_input(
        "Gemini API Key", type="password", placeholder="AIza..."
    )
    st.sidebar.markdown("[🔗 Get free Gemini API key](https://aistudio.google.com/app/apikey)")
elif ai_mode == "🔑 Anthropic (BYOK)":
    api_key = st.sidebar.text_input(
        "Anthropic API Key", type="password", placeholder="sk-ant-..."
    )

# ── Header ────────────────────────────────────────────
st.title(t["title"])
st.markdown(f"*{t['subtitle']}*")
st.divider()

# ── Simulation ────────────────────────────────────────
@st.cache_data
def run_simulation(weather):
    from env.f1_env import F1PitEnv
    from agents.rule_agent import RuleBasedAgent
    from stable_baselines3 import DQN

    rule_env = F1PitEnv(weather=weather)
    agent    = RuleBasedAgent()
    obs, _   = rule_env.reset()
    while True:
        action = agent.act(obs, rule_env)
        obs, _, term, _, _ = rule_env.step(action)
        if term:
            break

    dqn_env = F1PitEnv(weather=weather)
    model   = DQN.load(os.path.join(ROOT, "models", "dqn_f1_agent"))
    obs, _  = dqn_env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, _, _ = dqn_env.step(int(action))
        if term:
            break

    return (
        rule_env.lap_times, rule_env.total_time,
        rule_env.pit_laps,  rule_env.compounds_used,
        dqn_env.lap_times,  dqn_env.total_time,
        dqn_env.pit_laps,   dqn_env.compounds_used,
    )

with st.spinner(t["running"]):
    try:
        (
            rule_laps, rule_total, rule_pits, rule_compounds,
            dqn_laps,  dqn_total,  dqn_pits,  dqn_compounds,
        ) = run_simulation(weather)
        sim_ok = True
    except Exception as e:
        st.error(f"Error: {e}")
        sim_ok = False

if sim_ok:
    TOTAL_LAPS = len(rule_laps)
    laps       = list(range(1, TOTAL_LAPS + 1))
    delta      = rule_total - dqn_total

    DARK_BG = "#0d0d1a"
    CARD_BG = "#1a1a2e"
    WHITE   = "#f0f0f0"
    YELLOW  = "#FFF200"
    ACCENT  = "#00d2ff"
    RED     = "#E8002D"

    # ── Scoreboard ─────────────────────────────────────
    st.subheader(t["results"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t["rule_total"], f"{rule_total:.1f}s")
    c2.metric(t["dqn_total"],  f"{dqn_total:.1f}s")
    c3.metric(t["time_saved"], f"{delta:+.1f}s",
              delta=f"{delta:+.1f}s", delta_color="inverse")
    c4.metric(t["dqn_pit"],    str(dqn_pits))
    st.divider()

    # ── Lap time chart ──────────────────────────────────
    st.subheader(t["lap_chart"])
    fig, ax = plt.subplots(figsize=(14, 4), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.plot(laps, rule_laps, color=YELLOW, linewidth=1.8,
            label=t["rule_label"], marker="o", ms=2)
    ax.plot(laps, dqn_laps,  color=ACCENT, linewidth=1.8,
            label=t["dqn_label"],  marker="s", ms=2)
    for p in rule_pits:
        ax.axvline(p, color=YELLOW, alpha=0.4, linestyle="--", linewidth=1)
    for p in dqn_pits:
        ax.axvline(p, color=ACCENT, alpha=0.4, linestyle="--", linewidth=1)
    ax.set_xlabel(t["lap_label"], color=WHITE)
    ax.set_ylabel(t["lap_time"],  color=WHITE)
    ax.tick_params(colors=WHITE)
    ax.legend(facecolor=CARD_BG, labelcolor=WHITE)
    ax.grid(True, alpha=0.15, color="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    st.pyplot(fig)

    # ── Cumulative gap ──────────────────────────────────
    st.subheader(t["gap_chart"])
    fig2, ax2 = plt.subplots(figsize=(14, 3), facecolor=DARK_BG)
    ax2.set_facecolor(CARD_BG)
    gap = np.cumsum(rule_laps) - np.cumsum(dqn_laps)
    ax2.fill_between(laps, gap, 0, where=(gap >= 0),
                     color=ACCENT, alpha=0.5, label=t["dqn_ahead"])
    ax2.fill_between(laps, gap, 0, where=(gap < 0),
                     color=RED,   alpha=0.5, label=t["rule_ahead"])
    ax2.plot(laps, gap, color=WHITE, linewidth=1.5)
    ax2.axhline(0, color="#555", linewidth=1)
    ax2.set_xlabel(t["lap_label"], color=WHITE)
    ax2.set_ylabel(t["gap_label"], color=WHITE)
    ax2.tick_params(colors=WHITE)
    ax2.legend(facecolor=CARD_BG, labelcolor=WHITE)
    ax2.grid(True, alpha=0.15, color="gray")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#444")
    st.pyplot(fig2)

    # ── Tyre strategy ───────────────────────────────────
    st.subheader(t["tyre_timeline"])
    comp_colors = {"SOFT": RED, "MEDIUM": YELLOW, "HARD": WHITE}
    fig3, ax3 = plt.subplots(figsize=(14, 2.5), facecolor=DARK_BG)
    ax3.set_facecolor(CARD_BG)

    def draw_strategy(pit_laps, compounds_used, y, label):
        boundaries = [1] + sorted(pit_laps) + [TOTAL_LAPS + 1]
        compounds  = list(compounds_used)
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

    draw_strategy(rule_pits, rule_compounds, 0.7, "Rule")
    draw_strategy(dqn_pits,  dqn_compounds,  0.3, " DQN")
    ax3.set_xlim(0, TOTAL_LAPS+1)
    ax3.set_ylim(0, 1)
    ax3.set_xlabel(t["lap_label"], color=WHITE)
    ax3.set_yticks([])
    ax3.tick_params(colors=WHITE)
    for spine in ax3.spines.values():
        spine.set_edgecolor("#444")
    st.pyplot(fig3)

    # ── Tyre degradation ────────────────────────────────
    st.divider()
    st.subheader(t["tyre_deg"])
    from models.tyre_model import load_model, predict_lap_time
    tyre_model, le = load_model(os.path.join(ROOT, "models", "tyre_model.pkl"))
    fig4, ax4 = plt.subplots(figsize=(14, 4), facecolor=DARK_BG)
    ax4.set_facecolor(CARD_BG)
    ages = np.arange(1, 55)
    for compound, color in {"SOFT": RED, "MEDIUM": YELLOW, "HARD": WHITE}.items():
        times = [predict_lap_time(tyre_model, le, compound, age, 40) for age in ages]
        ax4.plot(ages, times, color=color, linewidth=2.5, label=compound)
    ax4.set_xlabel(t["tyre_age"], color=WHITE)
    ax4.set_ylabel(t["lap_time"], color=WHITE)
    ax4.tick_params(colors=WHITE)
    ax4.legend(facecolor=CARD_BG, labelcolor=WHITE)
    ax4.grid(True, alpha=0.2, color="gray")
    for spine in ax4.spines.values():
        spine.set_edgecolor("#444")
    st.pyplot(fig4)

    # ── Raw data ────────────────────────────────────────
    st.divider()
    with st.expander(t["raw_data"]):
        df = pd.DataFrame({
            t["lap_col"]:  laps,
            t["rule_col"]: [f"{x:.2f}" for x in rule_laps],
            t["dqn_col"]:  [f"{x:.2f}" for x in dqn_laps],
        })
        st.dataframe(df, use_container_width=True)

    # ── AI Strategist ─────────────────────────────────
    st.divider()
    st.subheader(t["ai_title"])
    st.caption(t["ai_caption"])

    qc1, qc2, qc3 = st.columns(3)
    q1 = qc1.button(t["ai_q1"])
    q2 = qc2.button(t["ai_q2"])
    q3 = qc3.button(t["ai_q3"])

    default_q = ""
    if q1: default_q = "When is the optimal lap to pit at Monaco 2024?"
    if q2: default_q = "What is the best strategy if it rains at lap 40?"
    if q3: default_q = "Why is 1-stop strategy better than 2-stop at Monaco?"

    question = st.text_input(
        t["ai_input"],
        value=default_q,
        placeholder=t["ai_placeholder"]
    )

    if st.button(t["ai_button"]) and question:
        from utils.ai_strategist import get_ai_response
        with st.spinner(t["ai_thinking"]):
            answer = get_ai_response(
                question, ai_mode, api_key, ollama_model
            )
        st.success(f"**{t['ai_answer']}:** {answer}")