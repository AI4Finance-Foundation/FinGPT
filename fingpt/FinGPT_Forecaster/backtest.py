"""Evaluate one-week FinGPT forecasts using archived closing prices."""

import argparse
import csv
import json
import math
import re
from bisect import bisect_right
from datetime import date, timedelta
from pathlib import Path


BASE_MODEL = "meta-llama/Llama-2-7b-chat-hf"
ADAPTER = "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora"
WEEK = timedelta(days=7)
PREDICTION = re.compile(
    r"^\s*(?:\*\*)?Prediction\s*:(?:\*\*)?\s*(up|down)\s+by\s+"
    r"(?:(more than|at least)\s+)?(\d+(?:\.\d+)?)"
    r"(?:\s*%?\s*[-–]\s*(\d+(?:\.\d+)?))?\s*\+?%(?=\s|[.*;,]|$)",
    re.IGNORECASE | re.MULTILINE,
)


def load_prices(path):
    prices = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"date", "close"}.issubset(reader.fieldnames or []):
            raise ValueError("Price CSV must have date and close columns.")
        for line, row in enumerate(reader, start=2):
            try:
                day = date.fromisoformat(row["date"])
                close = float(row["close"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid date or close on CSV line {line}.") from error
            if day in prices or not math.isfinite(close) or close <= 0:
                raise ValueError(f"Duplicate date or invalid close on CSV line {line}.")
            prices[day] = close
    return prices


def make_cases(prices, start, end, lookback_weeks):
    trading_days = sorted(prices)
    cutoff = start
    while cutoff <= end:
        boundaries = [cutoff - offset * WEEK for offset in range(lookback_weeks, -2, -1)]
        if not trading_days or boundaries[-1] > trading_days[-1]:
            break
        indices = [bisect_right(trading_days, boundary) - 1 for boundary in boundaries]
        if indices[0] >= 0 and len(set(indices)) == len(indices):
            days = [trading_days[index] for index in indices]
            history = [(day, prices[day]) for day in days[:-1]]
            actual = (prices[days[-1]] / history[-1][1] - 1) * 100
            yield cutoff, history, days[-1], actual
        cutoff += WEEK


def make_prompt(symbol, cutoff, history):
    movements = [
        f"From {start_day} to {end_day}, {symbol}'s closing price "
        f"moved from {start_close:.4f} to {end_close:.4f}."
        for (start_day, start_close), (end_day, end_close) in zip(history, history[1:])
    ]
    context = "\n".join(movements)
    return (
        "[INST] <<SYS>>\nYou are a stock market analyst. Forecast the upcoming week's "
        "price movement based on the historical information provided. Use the format "
        "'Prediction: up by 1-2%' or 'Prediction: down by 1-2%', followed by your analysis."
        f"\n<</SYS>>\n\n{context}\nForecast {symbol} from {cutoff} to {cutoff + WEEK}. [/INST]"
    )


def parse_prediction(answer):
    match = PREDICTION.search(answer)
    if not match:
        return None
    direction, lower_bound, first, second = match.groups()
    magnitude = float(first)
    if second is not None:
        upper = float(second)
        if lower_bound or upper < magnitude:
            return None
        magnitude = (magnitude + upper) / 2
    if not math.isfinite(magnitude):
        return None
    return magnitude if direction.lower() == "up" else -magnitude


def evaluate(symbol, cases, predict):
    rows = []
    squared_errors = []
    correct = 0
    for cutoff, history, target_date, actual in cases:
        answer = predict(make_prompt(symbol, cutoff, history))
        prediction = parse_prediction(answer)
        if prediction is not None:
            correct += (math.copysign(1, prediction) > 0) == (actual > 0)
            squared_errors.append((actual - prediction) ** 2)
        rows.append({
            "forecast_date": cutoff.isoformat(),
            "reference_close_date": history[-1][0].isoformat(),
            "target_close_date": target_date.isoformat(),
            "actual_return_pct": actual,
            "predicted_return_pct": prediction,
            "response": answer,
        })
    scored = len(squared_errors)
    return {
        "forecasts": len(rows),
        "scored": scored,
        "unparsed": len(rows) - scored,
        "directional_accuracy": correct / scored if scored else None,
        "mse_pct_points_squared": math.fsum(squared_errors) / scored if scored else None,
        "predictions": rows,
    }


def load_predictor(base_revision, adapter_revision):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=base_revision)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=base_revision, torch_dtype="auto", device_map="auto"
    )
    model = PeftModel.from_pretrained(base, ADAPTER, revision=adapter_revision).eval()

    def predict(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.get_input_embeddings().weight.device)
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=False, max_new_tokens=256)
        return tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

    return predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--lookback-weeks", type=int, default=3)
    parser.add_argument("--base-revision", required=True, help="Hugging Face base model commit SHA")
    parser.add_argument("--adapter-revision", required=True, help="Hugging Face adapter commit SHA")
    args = parser.parse_args()
    if args.start > args.end or args.lookback_weeks < 1:
        parser.error("Provide an ascending date range and a positive lookback.")
    if any(
        not re.fullmatch(r"[0-9a-fA-F]{40}", revision)
        for revision in (args.base_revision, args.adapter_revision)
    ):
        parser.error("Provide full commit SHAs for both checkpoint revisions.")
    try:
        cases = list(make_cases(load_prices(args.prices), args.start, args.end, args.lookback_weeks))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not cases:
        parser.error("No eligible weekly forecasts. Check the input history and target coverage.")
    results = evaluate(
        args.symbol, cases, load_predictor(args.base_revision, args.adapter_revision)
    )
    print(json.dumps({
        "base_model": BASE_MODEL,
        "base_revision": args.base_revision,
        "adapter": ADAPTER,
        "adapter_revision": args.adapter_revision,
        "symbol": args.symbol,
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "lookback_weeks": args.lookback_weeks,
        "horizon_days": WEEK.days,
        **results,
    }, indent=2))


if __name__ == "__main__":
    main()
