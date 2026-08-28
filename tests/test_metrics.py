import unittest

from metrics import evaluate_generations


class MetricTests(unittest.TestCase):
    def test_pass_at_k_and_format_rate(self):
        rows = [
            {
                "gold_answer": "42",
                "responses": ["Final answer: \\boxed{41}.", "Final answer: \\boxed{42}."],
            },
            {
                "gold_answer": "3/4",
                "responses": ["The result is 3/4.", "Final answer: \\boxed{3/4}."],
            },
        ]
        result = evaluate_generations(rows, k=2)
        self.assertAlmostEqual(result["pass@1"], 0.5)
        self.assertAlmostEqual(result["pass@k"], 1.0)
        self.assertAlmostEqual(result["format_rate@1"], 0.5)
        self.assertEqual(result["examples"], 2)

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_generations([])


if __name__ == "__main__":
    unittest.main()
