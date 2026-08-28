import math
import unittest

from verifier import extract_answer, normalize_answer, reward_response, verify_response


class VerifierTests(unittest.TestCase):
    def test_boxed_and_fraction_are_canonicalised(self):
        self.assertEqual(extract_answer("work\n\\boxed{3/4}"), "3/4")
        self.assertEqual(normalize_answer("0.75"), "3/4")
        self.assertEqual(normalize_answer("\\frac{6}{8}"), "3/4")

    def test_marker_and_unmarked_answer_are_distinguished(self):
        marked = verify_response("Final answer: \\boxed{42}.", "42")
        unmarked = verify_response("The result is 42.", "42")
        self.assertTrue(marked.is_correct)
        self.assertTrue(marked.format_valid)
        self.assertTrue(unmarked.is_correct)
        self.assertFalse(unmarked.format_valid)

    def test_invalid_response_gets_penalty(self):
        result = verify_response("I cannot parse this.", "42")
        self.assertIsNone(result.extracted_answer)
        self.assertFalse(result.is_correct)
        self.assertAlmostEqual(reward_response("I cannot parse this.", "42"), -0.1)

    def test_equivalent_negative_and_percent_values(self):
        self.assertEqual(normalize_answer("-2.0"), "-2")
        self.assertEqual(normalize_answer("25%"), "1/4")


if __name__ == "__main__":
    unittest.main()
