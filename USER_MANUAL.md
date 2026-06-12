# 📖 User Manual — F1 Pit Stop Strategy RL

---

## Getting Started

### 1. Open the App
Visit the deployed app or run locally:
```bash
streamlit run app.py
```

---

## Interface Guide

### Sidebar

#### 🌐 Language
Select your preferred language:
- **English**
- **हिंदी (Hindi)**
- **తెలుగు (Telugu)**

#### ☀️ Weather
Toggle between race conditions:
- **Dry** — Normal race conditions
- **Wet** — Wet weather (+15% lap times)

#### 📊 Model Info
Shows current model performance metrics.

#### 🤖 AI Strategist Backend
Choose your AI provider:

| Option | Description | Cost |
|--------|-------------|------|
| 🖥️ Ollama (Local) | Runs on your machine | Free |
| 🔑 Gemini (BYOK) | Google's free cloud API | Free tier |
| 🔑 Anthropic (BYOK) | Claude AI | Paid |

---

## Main Dashboard

### 📊 Race Results
Four metric cards showing:
- **Rule-Based Total** — Baseline 2-stop strategy time
- **DQN Agent Total** — RL agent race time
- **Time Saved** — How much faster the agent is
- **DQN Pit Laps** — When the agent pitted

### 📈 Lap Time Comparison
- **Yellow line** — Rule-based agent lap times
- **Blue line** — DQN agent lap times
- **Dashed vertical lines** — Pit stop moments

### ⏱️ Cumulative Time Gap
- **Blue area** — DQN is ahead
- **Red area** — Rule-based is ahead

### 🏁 Tyre Strategy Timeline
Visual bar showing compound used each lap:
- 🟡 **Yellow (M)** — Medium tyres
- ⬜ **White (H)** — Hard tyres
- 🔴 **Red (S)** — Soft tyres

### 📉 Tyre Degradation Model
Predicted lap time vs tyre age for each compound.

### 🔍 Raw Lap Data
Expandable table with lap-by-lap times for both agents.

---

## AI Strategist

### Using Gemini (Recommended for deployed app)
1. Go to [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Create a free API key
3. Select **Gemini (BYOK)** in sidebar
4. Paste your key
5. Ask any F1 strategy question

### Using Ollama (Local, Privacy-First)
1. Install Ollama: `brew install ollama`
2. Pull model: `ollama pull llama3`
3. Start server: `ollama serve`
4. Select **Ollama (Local)** in sidebar
5. Ask questions — data never leaves your machine

### Quick Questions
Click preset buttons for instant answers:
- ⏱️ **When to pit?**
- 🌧️ **Rain strategy?**
- 🏆 **Why 1-stop wins?**

---

## Interpreting Results

### Good Result
```
DQN Agent: 6178s
Rule-Based: 6226s
Time Saved: +47.7s ✅
```
DQN found a faster strategy!

### Understanding Pit Timing
- **Too early** (before lap 20) → Wastes fresh tyre life
- **Too late** (after lap 50) → Tyres degrade badly
- **Optimal** (lap 35-40) → Best balance at Monaco

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Simulation error | Check `models/tyre_model.pkl` exists |
| Ollama not responding | Run `ollama serve` in terminal |
| Gemini error | Check API key is valid |
