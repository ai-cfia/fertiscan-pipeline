import unittest
from pipeline.schemas.inspection import NutrientValue, Value
from scripts.evaluation.evaluate import compare_ingredients, compare_value, compare_weight, normalize_phone_number, normalize_website, preprocess_string

class TestUtilityFunctions(unittest.TestCase):

    def test_preprocess_string(self):
        self.assertEqual(preprocess_string("Hello World!"), "hello world")
        self.assertEqual(preprocess_string("Hello,    World!!!"), "hello world")
        self.assertEqual(preprocess_string(None), None)
        self.assertEqual(preprocess_string("   HEY!!!   "), "hey")
        self.assertEqual(preprocess_string("123"), "123")  # numeric input should stay as is except formatting

    def test_normalize_phone_number(self):
        # Valid Canadian number
        self.assertEqual(normalize_phone_number("416-555-1234"), "+14165551234")
        # Invalid number (returns original)
        self.assertEqual(normalize_phone_number("abc123"), "abc123")
        # Without country code but parseable
        self.assertEqual(normalize_phone_number("555-1234"), "+15551234")

    def test_normalize_website(self):
        # Basic domain
        self.assertEqual(normalize_website("example.com"), "example.com")
        # With scheme and trailing slash
        self.assertEqual(normalize_website("https://www.EXAMPLE.com/"), "example.com")
        # None input
        self.assertIsNone(normalize_website(None))

    def test_compare_value(self):

        val1 = Value(value=10, unit="kg")
        val2 = Value(value=10, unit="kg")
        self.assertEqual(compare_value(val1, val2), 1.0)

        val3 = Value(value=10, unit="lb")
        self.assertEqual(compare_value(val1, val3), 0.5)  # value matches, unit differs

        val4 = Value(value=20, unit="kg")
        self.assertEqual(compare_value(val1, val4), 0.5)  # unit matches, value differs

        val5 = Value(value=20, unit="lb")
        self.assertEqual(compare_value(val1, val5), 0.0)  # no match

        self.assertEqual(compare_value(None, None), 1.0)
        self.assertEqual(compare_value(val1, None), 0.0)
        self.assertEqual(compare_value(None, val2), 0.0)

    def test_compare_weight(self):

        val1 = Value(value=10, unit="kg")
        val2 = Value(value=20, unit="kg")
        val3 = Value(value=30, unit="kg")

        list1 = [val1, val2, val3]
        list2 = [val1, val2, val3]
        self.assertEqual(compare_weight(list1, list2), 1.0)

        # With differences
        val4 = Value(value=10, unit="lb")
        list3 = [val4, val2, val3]

        # First pair matches by value but not unit (0.5)
        # second and third pair matches fully (1.0)
        # average = (0.5 + 1.0 + 1.0) / 3 = approx 0.8333
        score = compare_weight(list1, list3)
        self.assertAlmostEqual(score, (0.5 + 1 + 1)/3)

        # None inputs
        self.assertEqual(compare_weight(None, None), 1.0)
        self.assertEqual(compare_weight(list1, None), 0.0)
        self.assertEqual(compare_weight(None, list1), 0.0)


class TestCompareIngredients(unittest.TestCase):
    def setUp(self):
        # Common test data
        self.exact_match = [
            NutrientValue(nutrient="Protein", value=10, unit="g"),
            NutrientValue(nutrient="Carbohydrate", value=20, unit="g"),
            NutrientValue(nutrient="Fat", value=5, unit="g"),
        ]

        self.out_of_order = [
            NutrientValue(nutrient="Fat", value=5, unit="g"),
            NutrientValue(nutrient="Carbohydrate", value=20, unit="g"),
            NutrientValue(nutrient="Protein", value=10, unit="g"),
        ]

        self.partial_match = [
            NutrientValue(nutrient="Protein", value=10, unit="g"),
            NutrientValue(nutrient="Carbohydrate", value=15, unit="g"),  # Different value
        ]

        self.missing_nutrient = [
            NutrientValue(nutrient="Protein", value=10, unit="g"),
        ]

        self.extra_nutrient = [
            NutrientValue(nutrient="Protein", value=10, unit="g"),
            NutrientValue(nutrient="Carbohydrate", value=20, unit="g"),
            NutrientValue(nutrient="Fat", value=5, unit="g"),
            NutrientValue(nutrient="Fiber", value=8, unit="g"),  # Extra nutrient
        ]

    def test_exact_match(self):
        result = compare_ingredients(self.exact_match, self.exact_match)
        self.assertEqual(result, 1.0, "Exact match should return 1.0")

    def test_out_of_order(self):
        result = compare_ingredients(self.exact_match, self.out_of_order)
        self.assertEqual(result, 1.0, "Out-of-order nutrients should still return 1.0")

    def test_partial_match(self):
        result = compare_ingredients(self.exact_match, self.partial_match)
        self.assertLess(result, 1.0, "Partial match should return a score less than 1.0")
        self.assertGreater(result, 0.0, "Partial match should not result in a score of 0.0")

    def test_missing_nutrient(self):
        result = compare_ingredients(self.exact_match, self.missing_nutrient)
        self.assertLess(result, 1.0, "A few missing nutrients should reduce the score")
        self.assertGreater(result, 0.0, "A few missing nutrients should not result in a score of 0.")

    def test_extra_nutrient(self):
        result = compare_ingredients(self.exact_match, self.extra_nutrient)
        self.assertLess(result, 1.0, "A few extra nutrients should reduce the score")
        self.assertGreater(result, 0.0, "A few extra nutrients should not result in a score of 0.0")

    def test_empty_lists(self):
        result = compare_ingredients([], [])
        self.assertEqual(result, 1.0, "Empty lists should return 1.0")

    def test_one_empty_list(self):
        result = compare_ingredients(self.exact_match, [])
        self.assertEqual(result, 0.0, "One empty list should return 0.0")

        result = compare_ingredients([], self.exact_match)
        self.assertEqual(result, 0.0, "One empty list should return 0.0")

    def test_mismatched_nutrients(self):
        mismatched = [
            NutrientValue(nutrient="Vitamin C", value=10, unit="mg"),
            NutrientValue(nutrient="Calcium", value=20, unit="mg"),
        ]
        result = compare_ingredients(self.exact_match, mismatched)
        self.assertEqual(result, 0.0, "Completely mismatched nutrients should return 0.0")


if __name__ == '__main__':
    unittest.main()