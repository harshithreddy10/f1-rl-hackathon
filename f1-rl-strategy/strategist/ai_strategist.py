import requests
import os


def ask_ollama(question, model="llama3"):
    """Local inference via Ollama — no API key needed."""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": f"""You are an expert F1 pit stop strategist.
You have access to Monaco 2024 race data.
Our RL agent found that a 1-stop strategy (pit lap 38, MEDIUM→HARD)
saves 47.7 seconds over a 2-stop baseline.
Answer concisely and accurately.

Question: {question}""",
                "stream": False
            },
            timeout=30
        )
        return response.json()["response"]
    except requests.exceptions.ConnectionError:
        return "⚠️ Ollama is not running. Start it with: `ollama serve`"
    except Exception as e:
        return f"⚠️ Ollama error: {e}"


def ask_gemini(question, api_key):
    """Cloud inference via Google Gemini — free tier available."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            f"""You are an expert F1 pit stop strategist.
You have access to Monaco 2024 race data.
Our RL agent found that a 1-stop strategy (pit lap 38, MEDIUM→HARD)
saves 47.7 seconds over a 2-stop baseline.
Answer concisely and accurately.

Question: {question}"""
        )
        return response.text
    except Exception as e:
        return f"⚠️ Gemini error: {e}"


def ask_anthropic(question, api_key):
    """Cloud inference via Anthropic BYOK."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": f"""You are an expert F1 pit stop strategist.
You have access to Monaco 2024 race data.
Our RL agent found that a 1-stop strategy (pit lap 38, MEDIUM→HARD)
saves 47.7 seconds over a 2-stop baseline.
Answer concisely.

Question: {question}"""
                }
            ]
        )
        return message.content[0].text
    except Exception as e:
        return f"⚠️ Anthropic error: {e}"


def get_ai_response(question, mode, api_key=None, ollama_model="llama3"):
    """Route question to correct AI backend."""
    if mode == "🖥️ Ollama (Local)":
        return ask_ollama(question, ollama_model)
    elif mode == "🔑 Gemini (BYOK)":
        if not api_key:
            return "⚠️ Please enter your Gemini API key in the sidebar."
        return ask_gemini(question, api_key)
    elif mode == "🔑 Anthropic (BYOK)":
        if not api_key:
            return "⚠️ Please enter your Anthropic API key in the sidebar."
        return ask_anthropic(question, api_key)
    return "⚠️ No AI mode selected."