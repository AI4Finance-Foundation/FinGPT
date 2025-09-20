import os
import json
from rouge_score import rouge_scorer

def load_text(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data.get("text", "")

def main():
    refs_dir = r"C:\Users\rahul\OneDrive\Desktop\FinGPT-M\fingpt\stock_chart_trends_analysis\eval\refs\images_ref_json"
    preds_dir = r"C:\Users\rahul\OneDrive\Desktop\FinGPT-M\fingpt\stock_chart_trends_analysis\eval\preds\json"

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    results = []

    ref_files = [f for f in os.listdir(refs_dir) if f.endswith(".json")]
    for fname in ref_files:
        ref_path = os.path.join(refs_dir, fname)
        pred_path = os.path.join(preds_dir, fname)
        if not os.path.exists(pred_path):
            print(f"Prediction file missing for {fname}, skipping.")
            continue
        ref_text = load_text(ref_path)
        pred_text = load_text(pred_path)
        scores = scorer.score(ref_text, pred_text)
        result = {
            "file": fname,
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4)
        }
        results.append(result)
        print(f"{fname}: ROUGE-1={result['rouge1']}, ROUGE-2={result['rouge2']}, ROUGE-L={result['rougeL']}")

    with open("rouge_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()