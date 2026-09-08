import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1]
    / "fingpt"
    / "FinGPT_Benchmark"
    / "benchmarks"
    / "ashare_factor.py"
)
SPEC = importlib.util.spec_from_file_location("ashare_factor", MODULE_PATH)
ashare_factor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ashare_factor)


class AShareFactorBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.valid_record = {
            "symbol": "600000.SH",
            "as_of": "2026-01-01",
            "factors": [
                {
                    "name": "earnings",
                    "direction": "positive",
                    "evidence": "revenue growth",
                }
            ],
            "risks": ["valuation"],
            "recommendation": "buy",
            "confidence": 0.8,
        }

    def test_complete_record_is_valid(self):
        result = ashare_factor.evaluate_record(self.valid_record)

        self.assertTrue(result["valid"])
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["errors"], [])

    def test_invalid_confidence_is_reported(self):
        record = dict(self.valid_record, confidence=1.2)

        result = ashare_factor.evaluate_record(record)

        self.assertFalse(result["valid"])
        self.assertIn("confidence must be a number between 0 and 1", result["errors"])


if __name__ == "__main__":
    unittest.main()