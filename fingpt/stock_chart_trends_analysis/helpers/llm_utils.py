import json, os
from typing import Any, Dict, List, Tuple, Optional
from config import OUTPUT_BEGIN, OUTPUT_END, OPENAI_MODEL, client
from helpers.general_utils import _between_markers

def simple_sentiment_from_patterns(preds: List[Dict[str, Any]]) -> str:
    bullish = {"morning_star_rise","hammer","bullish_engulfing","ascending_triangle","golden_cross"}
    bearish = {"evening_star_fall","shooting_star","bearish_engulfing","descending_triangle","death_cross"}
    score = sum(1 for p in preds if any(b in p.get("class","").lower() for b in bullish)) - \
            sum(1 for p in preds if any(b in p.get("class","").lower() for b in bearish))
    return "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"

# --- LLM wrapper (OpenAI Chat Completions) ---
def llm_generate(system_prompt: str,
                 user_prompt: str,
                 max_new_tokens: int = 500,
                 temperature: float = 0.2,
                 top_p: float = 0.95,
                 stop: Optional[List[str]] = None) -> str:
    """
    Thin wrapper over OpenAI Chat Completions returning assistant text.
    """
    stop = stop or None
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        stop=stop,
    )
    return (resp.choices[0].message.content or "").strip()


def build_llm_prompts_for_forecast(final_output, news_snippets) -> Tuple[str, str]:
    ctx = {
        "ticker": final_output.get("ticker"),
        "exchange": final_output.get("exchange"),
        "ohlc": final_output.get("ohlc"),
        "sessions": final_output.get("sessions"),
        "price_range": final_output.get("price_range"),
        "predictions": final_output.get("predictions"),
    }
    news_text = "\n".join(
        f"- {n['date']} | {n['headline']} :: {n['summary'][:200]}..."
        for n in (news_snippets or [])[:6]
    )
    system = (
        "You are a seasoned stock market analyst. Use ONLY the JSON (if available) and NEWS provided. "
        "Write exactly three markdown sections: Positive Developments, Potential Concerns, Forecast & Analysis."
    )
    user = (
        f"[JSON]\n{json.dumps(ctx, indent=2)}\n\n"
        f"[NEWS]\n{news_text if news_text else 'No recent news'}\n\n"
        f"{OUTPUT_BEGIN}\n<Your answer here>\n{OUTPUT_END}"
    )
    return system, user

def llm_brief_summary(final_output: Dict[str, Any], sentiment: str, mode: str = "image") -> str:
    ctx = {
        "ticker": final_output.get("ticker") or final_output.get("company_ticker"),
        "company_name": final_output.get("company_name"),
        "exchange": final_output.get("exchange"),
        "price_range": final_output.get("price_range"),
        "ohlc": final_output.get("ohlc"),
        "predictions": final_output.get("predictions"),
        "pattern_sentiment": sentiment,
    }
    title = "### Chart Summary" if mode == "image" else "### Text Query Summary"
    system = (
        "You are a precise market assistant. Using ONLY the provided JSON, "
        "return EXACTLY 5 concise markdown bullets. Each bullet must start with '- ' "
        "and include a bold label followed by a short fact."
    )
    user = (
        f"[JSON]\n{json.dumps(ctx, indent=2)}\n\n"
        f"{OUTPUT_BEGIN}\n"
        f"- **Label**: point\n"
        f"- **Label**: point\n"
        f"- **Label**: point\n"
        f"- **Label**: point\n"
        f"- **Label**: point\n"
        f"{OUTPUT_END}"
    )
    raw = llm_generate(system, user, max_new_tokens=280, temperature=0.2, top_p=0.95, stop=[OUTPUT_END])
    bullets = [ln.strip() for ln in _between_markers(raw).splitlines() if ln.strip().startswith("- ")]
    return title + "\n" + "\n".join(bullets)