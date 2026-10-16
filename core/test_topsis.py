"""
Unit Tests for Pure Python TOPSIS Engine
Task: T9-01 — TOPSIS V1: ALGORITHM, BACKEND, FRONTEND & ACCEPTANCE
Author: Nguyen Ho Truong Tam
"""

import math
from django.test import SimpleTestCase
from core.topsis_service import TopsisEngine


class TopsisEngineTests(SimpleTestCase):
    def test_01_hand_calculated_textbook_example(self):
        """
        Validates exact mathematical calculations against hand-computed values:
        3 Alternatives (A1, A2, A3), 2 Criteria:
        - C1: Benefit, w1=0.6, col=[4.0, 3.0, 5.0]
        - C2: Cost, w2=0.4, col=[10.0, 20.0, 15.0]

        Expected:
        - Col norm 1: sqrt(50) = 7.071068
        - Col norm 2: sqrt(725) = 26.925824
        - Ideal Positive A+: [0.424264, 0.148556]
        - Ideal Negative A-: [0.254558, 0.297113]
        - Closeness C1 ~ 0.668458, C2 = 0.0, C3 ~ 0.713797
        - Rank: #1 A3, #2 A1, #3 A2
        """
        matrix = [
            [4.0, 10.0],
            [3.0, 20.0],
            [5.0, 15.0],
        ]
        weights = [0.6, 0.4]
        types = ["benefit", "cost"]
        alt_ids = ["A1", "A2", "A3"]
        crit_names = ["Kha_Nang", "Chi_Phi"]

        result = TopsisEngine.evaluate(
            matrix=matrix,
            weights=weights,
            criteria_types=types,
            alternative_ids=alt_ids,
            criteria_names=crit_names,
        )

        # Check normalization
        col1_norm = math.sqrt(4.0**2 + 3.0**2 + 5.0**2)
        self.assertAlmostEqual(result["normalized_matrix"][0][0], 4.0 / col1_norm, places=4)
        self.assertAlmostEqual(result["normalized_matrix"][1][0], 3.0 / col1_norm, places=4)
        self.assertAlmostEqual(result["normalized_matrix"][2][0], 5.0 / col1_norm, places=4)

        # Check Ideal Solutions
        self.assertAlmostEqual(result["ideal_positive"][0], 0.424264, places=4)
        self.assertAlmostEqual(result["ideal_positive"][1], 0.148556, places=4)
        self.assertAlmostEqual(result["ideal_negative"][0], 0.254558, places=4)
        self.assertAlmostEqual(result["ideal_negative"][1], 0.297113, places=4)

        # Check Closeness Scores
        # A1: C1 ~ 0.668458
        self.assertAlmostEqual(result["closeness_scores"][0], 0.668458, places=4)
        # A2: C2 = 0.0
        self.assertAlmostEqual(result["closeness_scores"][1], 0.000000, places=4)
        # A3: C3 ~ 0.713797
        self.assertAlmostEqual(result["closeness_scores"][2], 0.713797, places=4)

        # Check Rankings
        rankings = result["rankings"]
        self.assertEqual(len(rankings), 3)
        self.assertEqual(rankings[0]["alternative_id"], "A3")
        self.assertEqual(rankings[0]["rank"], 1)
        self.assertEqual(rankings[1]["alternative_id"], "A1")
        self.assertEqual(rankings[1]["rank"], 2)
        self.assertEqual(rankings[2]["alternative_id"], "A2")
        self.assertEqual(rankings[2]["rank"], 3)

    def test_01b_dialysis_five_criteria_verification(self):
        """
        Validates 3 alternatives across 5 dialysis clinical criteria:
        A1 = [6.0, 8.0, 3.0, 3.0, 5.0]
        A2 = [4.0, 12.0, 1.0, 1.0, 15.0]
        A3 = [5.5, 9.0, 4.0, 2.0, 10.0]
        Weights = [0.30, 0.20, 0.25, 0.15, 0.10]
        Directions = ['benefit', 'cost', 'benefit', 'benefit', 'cost']

        Mathematical verification:
        - A1: D+ = 0.049029, D- = 0.159664, C1 = 0.765066 -> Rank #1
        - A3: D+ = 0.052282, D- = 0.166324, C3 = 0.760840 -> Rank #2
        - A2: D+ = 0.193680, D- = 0.000000, C2 = 0.000000 -> Rank #3
        """
        matrix = [
            [6.0, 8.0, 3.0, 3.0, 5.0],
            [4.0, 12.0, 1.0, 1.0, 15.0],
            [5.5, 9.0, 4.0, 2.0, 10.0],
        ]
        weights = [0.30, 0.20, 0.25, 0.15, 0.10]
        types = ["benefit", "cost", "benefit", "benefit", "cost"]
        alt_ids = ["A1", "A2", "A3"]

        result = TopsisEngine.evaluate(matrix, weights, types, alternative_ids=alt_ids)

        self.assertAlmostEqual(result["distances_positive"][0], 0.049029, places=4)
        self.assertAlmostEqual(result["distances_negative"][0], 0.159664, places=4)
        self.assertAlmostEqual(result["closeness_scores"][0], 0.765066, places=4)

        self.assertAlmostEqual(result["distances_positive"][1], 0.193680, places=4)
        self.assertAlmostEqual(result["distances_negative"][1], 0.000000, places=4)
        self.assertAlmostEqual(result["closeness_scores"][1], 0.000000, places=4)

        self.assertAlmostEqual(result["distances_positive"][2], 0.052282, places=4)
        self.assertAlmostEqual(result["distances_negative"][2], 0.166324, places=4)
        self.assertAlmostEqual(result["closeness_scores"][2], 0.760840, places=4)

        self.assertEqual(result["rankings"][0]["alternative_id"], "A1")
        self.assertEqual(result["rankings"][0]["rank"], 1)
        self.assertEqual(result["rankings"][1]["alternative_id"], "A3")
        self.assertEqual(result["rankings"][1]["rank"], 2)
        self.assertEqual(result["rankings"][2]["alternative_id"], "A2")
        self.assertEqual(result["rankings"][2]["rank"], 3)

    def test_02_all_benefit_criteria(self):
        """Validates evaluation when all criteria are benefit criteria."""
        matrix = [
            [10.0, 50.0],
            [20.0, 30.0],
            [30.0, 10.0],
        ]
        weights = [0.5, 0.5]
        types = ["benefit", "benefit"]

        result = TopsisEngine.evaluate(matrix, weights, types)
        # All C_i must be between 0 and 1
        for c in result["closeness_scores"]:
            self.assertGreaterEqual(c, 0.0)
            self.assertLessEqual(c, 1.0)
        self.assertEqual(len(result["rankings"]), 3)

    def test_03_all_cost_criteria(self):
        """Validates evaluation when all criteria are cost criteria."""
        matrix = [
            [5.0, 10.0],
            [15.0, 20.0],
        ]
        weights = [1.0, 1.0]
        types = ["cost", "cost"]

        result = TopsisEngine.evaluate(matrix, weights, types)
        # A1 has strictly lower cost in both criteria -> should rank #1
        self.assertEqual(result["rankings"][0]["alternative_id"], "A1")
        self.assertAlmostEqual(result["rankings"][0]["closeness_score"], 1.0, places=4)
        self.assertAlmostEqual(result["rankings"][1]["closeness_score"], 0.0, places=4)

    def test_04_single_alternative(self):
        """Validates behavior when m = 1 (single alternative)."""
        matrix = [[5.0, 10.0]]
        weights = [0.5, 0.5]
        types = ["benefit", "cost"]

        result = TopsisEngine.evaluate(matrix, weights, types)
        self.assertEqual(len(result["rankings"]), 1)
        self.assertEqual(result["rankings"][0]["rank"], 1)
        self.assertEqual(result["rankings"][0]["closeness_score"], 1.0)

    def test_05_identical_alternatives_equal_score(self):
        """Validates behavior when all alternatives have identical rows."""
        matrix = [
            [10.0, 20.0],
            [10.0, 20.0],
            [10.0, 20.0],
        ]
        weights = [0.5, 0.5]
        types = ["benefit", "cost"]

        result = TopsisEngine.evaluate(matrix, weights, types)
        for c in result["closeness_scores"]:
            self.assertAlmostEqual(c, 0.5, places=4)
        self.assertEqual(len(result["rankings"]), 3)

    def test_06_zero_column_handled_safely(self):
        """Validates that a criterion column containing only 0.0 does not cause division by zero."""
        matrix = [
            [10.0, 0.0],
            [20.0, 0.0],
            [30.0, 0.0],
        ]
        weights = [0.8, 0.2]
        types = ["benefit", "benefit"]

        result = TopsisEngine.evaluate(matrix, weights, types)
        # Col 2 normalized values should be 0.0
        for row in result["normalized_matrix"]:
            self.assertEqual(row[1], 0.0)
        # Rankings determined cleanly by Col 1
        self.assertEqual(result["rankings"][0]["alternative_id"], "A3")
        self.assertEqual(result["rankings"][1]["alternative_id"], "A2")
        self.assertEqual(result["rankings"][2]["alternative_id"], "A1")

    def test_07_unnormalized_weights_auto_scaled(self):
        """Validates that weights not summing to 1.0 (e.g. [3, 7]) are normalized to [0.3, 0.7]."""
        matrix = [
            [4.0, 10.0],
            [3.0, 20.0],
            [5.0, 15.0],
        ]
        weights_raw = [60.0, 40.0]
        types = ["benefit", "cost"]

        result = TopsisEngine.evaluate(matrix, weights_raw, types)
        self.assertAlmostEqual(result["weights"][0], 0.6, places=5)
        self.assertAlmostEqual(result["weights"][1], 0.4, places=5)

    def test_08_invalid_inputs_raise_value_errors(self):
        """Validates comprehensive edge-case exceptions."""
        # Empty matrix
        with self.assertRaises(ValueError):
            TopsisEngine.evaluate([], [1.0], ["benefit"])

        # Negative weight
        with self.assertRaises(ValueError):
            TopsisEngine.evaluate([[1.0]], [-0.5], ["benefit"])

        # Zero sum of weights
        with self.assertRaises(ValueError):
            TopsisEngine.evaluate([[1.0]], [0.0], ["benefit"])

        # Dimension mismatch
        with self.assertRaises(ValueError):
            TopsisEngine.evaluate([[1.0, 2.0]], [0.5], ["benefit"])

        # Inconsistent row length
        with self.assertRaises(ValueError):
            TopsisEngine.evaluate([[1.0, 2.0], [3.0]], [0.5, 0.5], ["benefit", "cost"])

        # NaN / Inf in matrix
        with self.assertRaises(ValueError):
            TopsisEngine.evaluate([[float("nan"), 2.0]], [0.5, 0.5], ["benefit", "cost"])

        with self.assertRaises(ValueError):
            TopsisEngine.evaluate([[float("inf"), 2.0]], [0.5, 0.5], ["benefit", "cost"])

        # Invalid criterion type string
        with self.assertRaises(ValueError):
            TopsisEngine.evaluate([[1.0]], [1.0], ["invalid_type"])

    def test_09_deterministic_tie_breaking(self):
        """Validates that identical scores preserve deterministic order without crashing."""
        matrix = [
            [10.0, 5.0],
            [20.0, 5.0],
            [20.0, 5.0],
        ]
        weights = [0.7, 0.3]
        types = ["benefit", "cost"]
        alt_ids = ["P1", "P2", "P3"]

        result = TopsisEngine.evaluate(matrix, weights, types, alternative_ids=alt_ids)
        # P2 and P3 have identical rows -> identical closeness
        self.assertAlmostEqual(result["closeness_scores"][1], result["closeness_scores"][2], places=5)
        # Ranks must be 1, 2, 3
        ranks = [r["rank"] for r in result["rankings"]]
        self.assertEqual(ranks, [1, 2, 3])

    def test_10_reproducibility(self):
        """Running the same matrix 10 times produces identical rankings and scores 100%."""
        matrix = [
            [5.8, 10.2, 4.0, 2.5, 3.2],
            [4.2, 12.0, 2.0, 1.1, 8.5],
            [6.5, 8.9, 5.0, 3.8, 12.1],
        ]
        weights = [0.3, 0.2, 0.25, 0.15, 0.1]
        types = ["benefit", "cost", "benefit", "benefit", "cost"]

        r1 = TopsisEngine.evaluate(matrix, weights, types)
        for _ in range(9):
            rn = TopsisEngine.evaluate(matrix, weights, types)
            self.assertEqual(r1["rankings"], rn["rankings"])
            self.assertEqual(r1["closeness_scores"], rn["closeness_scores"])
