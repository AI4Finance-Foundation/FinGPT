"""Evaluate the structure of A-share factor research outputs.

The evaluator intentionally does not claim that a recommendation is correct.
It checks whether a report contains the fields needed for reproducible review.
"""

import argparse
import json
from pathlib import Path


RECOMMENDATIONS = {"buy", "hold", "sell"}
REQUIRED_FIELDS = {"symbol", "as_of", "factors", "risks", "recommendation", "confidence"}


def evaluate_record(record):
    """Return structural metrics and validation errors for one report."""
    errors = []
    if not isinstance(record, dict):
        return {"valid": False, "score": 0.0, "errors": ["record must be an object"]}

    missing = sorted(REQUIRED_FIELDS - record.keys())
    errors.extend(f"missing field: {field}" for field in missing)

    factors = record.get("factors")
    if not isinstance(factors, list) or not factors:
        errors.append("factors must be a non-empty list")
    else:
        for index, factor in enumerate(factors):
            if not isinstance(factor, dict):
                errors.append(f"factors[{index}] must be an object")
                continue
            for field in ("name", "direction", "evidence"):
                if not factor.get(field):
                    errors.append(f"factors[{index}] missing field: {field}")

    risks = record.get("risks")
    if not isinstance(risks, list) or not risks:
        errors.append("risks must be a non-empty list")

    if record.get("recommendation") not in RECOMMENDATIONS:
        errors.append("recommendation must be buy, hold, or sell")

    confidence = record.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
        errors.append("confidence must be a number between 0 and 1")

    score = 1.0 - min(len(errors), 5) / 5
    return {"valid": not errors, "score": round(score, 3), "errors": errors}


def evaluate_file(path):
    """Evaluate newline-delimited JSON reports and return aggregate metrics."""
    results = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                result = evaluate_record(record)
            except json.JSONDecodeError as error:
                result = {"valid": False, "score": 0.0, "errors": [f"invalid JSON: {error.msg}"]}
            result["line"] = line_number
            results.append(result)

    if not results:
        raise ValueError("input file contains no JSON records")

    return {
        "records": len(results),
        "valid_records": sum(result["valid"] for result in results),
        "mean_score": round(sum(result["score"] for result in results) / len(results), 3),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, help="newline-delimited JSON report file")
    args = parser.parse_args()
    print(json.dumps(evaluate_file(args.reports), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()