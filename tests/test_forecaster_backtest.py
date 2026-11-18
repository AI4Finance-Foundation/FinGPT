from datetime import date

import pytest

from fingpt.FinGPT_Forecaster.backtest import evaluate, load_prices, make_cases, parse_prediction


def test_backtest_keeps_future_close_out_of_prompt_and_reports_coverage(tmp_path):
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text(
        "date,close\n2025-01-02,99\n2025-01-10,100\n"
        "2025-01-17,101\n2025-01-24,97\n2025-01-27,98\n"
    )
    # Saturday cutoffs use the last close known by that date.
    cases = list(make_cases(
        load_prices(csv_path), date(2025, 1, 11), date(2025, 1, 18), lookback_weeks=1
    ))
    prompts = []
    responses = iter(["Prediction: up by 1–3%", "There is no clear outlook."])

    def predict(prompt):
        prompts.append(prompt)
        return next(responses)

    report = evaluate("TEST", cases, predict)
    assert "100.0000" in prompts[0]
    assert "101.0000" not in prompts[0]
    assert "101.0000" in prompts[1]
    assert "97.0000" not in prompts[1]
    assert report["predictions"][0]["target_close_date"] == "2025-01-17"
    assert report["forecasts"] == 2
    assert report["unparsed"] == 1
    assert report["directional_accuracy"] == 1.0
    assert report["mse_pct_points_squared"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("answer", "prediction"),
    [
        ("Prediction: Down by more than 5%", -5.0),
        ("**Prediction:** down by 0-1%", -0.5),
        ("Prediction: Up by 1.25%", 1.25),
        ("Prediction: Up by 3-1%", None),
        ("Prediction: Up by 2 dollars", None),
    ],
)
def test_parse_prediction(answer, prediction):
    assert parse_prediction(answer) == prediction
