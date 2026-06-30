"""
AI Copilot — powered by Google Gemini 2.0 Flash.
API key is read from the environment (Streamlit Cloud secrets / .env file),
with a hardcoded fallback for convenience during local development.
"""
import os
import requests

_HARDCODED_KEY = "AQ.Ab8RN6JjAtMnTc4zwxyyZ4FJTTkh6hOqgOkQGAKl3eIozpj9Lg"
_MODEL         = "gemini-2.0-flash"
_API_BASE      = "https://generativelanguage.googleapis.com/v1beta/models"

SYSTEM_PROMPT = (
    "You are SupplySenseAI Copilot, an expert supply chain risk analyst. "
    "Help supply chain managers understand demand forecasts, uncertainty bands, "
    "and procurement recommendations in plain English. "
    "Be specific — reference exact numbers from the provided context. "
    "End every response with a clear, actionable recommendation. "
    "Keep answers concise: 4 to 6 sentences."
)


def _get_api_key() -> str:
    """Return Gemini API key from environment variable or hardcoded fallback."""
    return os.environ.get("GEMINI_API_KEY", _HARDCODED_KEY).strip()


def _build_context_string(data: dict) -> str:
    """Format the SKU context dictionary into a readable string."""
    return "\n".join(f"  {k}: {v}" for k, v in data.items())


def ask(question: str, sku_data: dict, history: list) -> tuple:
    """
    Send a question to the Gemini API and return (answer, updated_history).

    Parameters
    ----------
    question  : The user's question.
    sku_data  : Dictionary of SKU metrics passed as context.
    history   : Prior conversation turns (role='user'|'assistant').

    Returns
    -------
    (answer_text, updated_history)
    """
    api_key = _get_api_key()
    context = _build_context_string(sku_data)

    # Build the contents array for Gemini (maps 'assistant' -> 'model')
    contents = []
    for msg in history:
        gemini_role = "model" if msg["role"] == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": [{"text": msg["content"]}]})

    # On the first turn, prepend full SKU context to the question
    user_text = (
        f"Current SKU Context:\n{context}\n\nQuestion: {question}"
        if not history
        else question
    )
    contents.append({"role": "user", "parts": [{"text": user_text}]})

    payload = {
        "system_instruction": {
            "parts": [{"text": f"{SYSTEM_PROMPT}\n\nSKU Context:\n{context}"}]
        },
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": 1024,
            "temperature": 0.4,
        },
    }

    url      = f"{_API_BASE}/{_MODEL}:generateContent?key={api_key}"
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )

    if not response.ok:
        try:
            error_msg = response.json().get("error", {}).get("message", response.text)
        except Exception:
            error_msg = response.text
        raise RuntimeError(f"Gemini API error {response.status_code}: {error_msg}")

    data       = response.json()
    candidates = data.get("candidates", [])

    if not candidates:
        block_reason = data.get("promptFeedback", {}).get("blockReason", "unknown")
        raise RuntimeError(f"Gemini returned no candidates (blockReason: {block_reason})")

    answer = candidates[0]["content"]["parts"][0]["text"]

    updated_history = list(history)
    updated_history.append({"role": "user",      "content": user_text})
    updated_history.append({"role": "assistant", "content": answer})
    return answer, updated_history


class CopilotSession:
    """Maintains a stateful conversation with Gemini for a single SKU session."""

    def __init__(self, sku_data: dict):
        self.sku_data = sku_data
        self.history: list = []

    def chat(self, message: str) -> str:
        answer, self.history = ask(message, self.sku_data, self.history)
        return answer

    def reset(self):
        self.history = []
