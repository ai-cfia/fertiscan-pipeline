import unittest
from pipeline.inspection import FertilizerInspection, NutrientValue, Value, Specification, npkError
from pydantic import ValidationError


class TestNutrientValue(unittest.TestCase):

    def setUp(self):
        # Valid test case
        self.valid_nutrient_value_data = {
            "nutrient": "Nitrogen",
            "value": "2 mg/L",
            "unit": "mg/L"
        }
        # Expanded invalid test cases for different edge cases
        self.invalid_nutrient_value_data = [
            {"nutrient": "Nitrogen", "value": "abc mg/L", "unit": "mg/L"},  # Non-numeric value
            {"nutrient": "Nitrogen", "value": "", "unit": "mg/L"},  # Empty string
            {"nutrient": "Nitrogen", "value": "30percent", "unit": "mg/L"},  # Mixed number and text
            {"nutrient": "Nitrogen", "value": "@30", "unit": "mg/L"},  # Special characters
            {"nutrient": "Nitrogen", "value": " ", "unit": "mg/L"},  # Whitespace string
            {"nutrient": "Nitrogen", "value": None, "unit": "mg/L"},  # None value
        ]

    def test_valid_nutrient_value(self):
        nutrient_value = NutrientValue(**self.valid_nutrient_value_data)
        self.assertEqual(nutrient_value.nutrient, "Nitrogen")
        self.assertEqual(nutrient_value.value, "2")  # Expect a clean numeric string
        self.assertEqual(nutrient_value.unit, "mg/L")
        
        # Ensure that we can safely convert the value to int and float
        self.assertIsInstance(int(nutrient_value.value), int)
        self.assertIsInstance(float(nutrient_value.value), float)

    def test_invalid_nutrient_value(self):
        for data in self.invalid_nutrient_value_data:
            with self.subTest(data=data):
                nutrient_value = NutrientValue(**data)
                self.assertEqual(nutrient_value.nutrient, "Nitrogen")
                # Invalid cases should result in value being None
                self.assertIsNone(nutrient_value.value, f"Expected None for value with input {data['value']}")
                self.assertEqual(nutrient_value.unit, "mg/L")


class TestValue(unittest.TestCase):

    def setUp(self):
        # Valid test cases for Value
        self.valid_value_data = [
            {"value": "12.5", "unit": "kg"},  # Simple numeric value
            {"value": "12.5 kg", "unit": "kg"},  # Value with unit mixed in
            {"value": "12.5%", "unit": "kg"},  # Value with a percentage sign
        ]
        # Expanded invalid test cases with various edge cases
        self.invalid_value_data = [
            {"value": "abc", "unit": "kg"},  # Non-numeric value
            {"value": "", "unit": "kg"},  # Empty string
            {"value": "approximately 10", "unit": "kg"},  # Mixed text and number
            {"value": "@30", "unit": "kg"},  # Special characters
            {"value": None, "unit": "kg"},  # None value
        ]

    def test_valid_value(self):
        for data in self.valid_value_data:
            with self.subTest(data=data):
                value = Value(**data)
                self.assertEqual(value.value, "12.5")  # Expect a clean numeric string
                self.assertEqual(value.unit, "kg")

    def test_invalid_value(self):
        for data in self.invalid_value_data:
            with self.subTest(data=data):
                value = Value(**data)
                # Invalid cases should result in value being None
                self.assertIsNone(value.value, f"Expected None for value with input {data['value']}")
                self.assertEqual(value.unit, "kg")

class TestSpecification(unittest.TestCase):

    def setUp(self):
        self.valid_specification_data = {
            "humidity": "30",
            "ph": "6.5",
            "solubility": "10"
        }
        self.invalid_specification_data = [
            {"humidity": "forty", "ph": "six", "solubility": "unknown"},  # non-numeric words
            {"humidity": "", "ph": "", "solubility": ""},  # empty strings
            {"humidity": "@30%", "ph": "6.5#", "solubility": "ten$"},  # special characters
            {"humidity": "30percent", "ph": "ph=6.5", "solubility": "approximately 10"},  # mixed numbers and text
            {"humidity": True, "ph": ["6.5"], "solubility": {"value": "10"}},  # non-numeric types
            {"humidity": -10, "ph": -3.0, "solubility": 0},  # negative and edge values
            {"humidity": None, "ph": None, "solubility": None},  # null values
            {"humidity": "9999999999", "ph": "1000000000.1", "solubility": "3.1415926535"},  # large values
            {"humidity": "30.5", "ph": "6.5", "solubility": "10.2"},  # decimals for integers
            {"humidity": "1e2", "ph": "6.5e-1", "solubility": "10"},  # scientific notation
            {"humidity": " ", "ph": " ", "solubility": " "},  # whitespace-only
            {"humidity": "30 and 40", "ph": "6.5/7.0", "solubility": "10, 15"}  # multiple numbers
        ]

    def test_valid_specification(self):
        specification = Specification(**self.valid_specification_data)
        self.assertEqual(specification.humidity, "30")
        self.assertEqual(specification.ph, "6.5")
        self.assertEqual(specification.solubility, "10")
        
        # Ensure valid data can be cast to int or float without issues
        self.assertIsInstance(int(specification.humidity), int)
        self.assertIsInstance(float(specification.ph), float)
        self.assertIsInstance(int(specification.solubility), int)

    def test_invalid_specification(self):
        for data in self.invalid_specification_data:
            with self.subTest(data=data):
                specification = Specification(**data)
                # Invalid cases should result in None for the respective fields
                self.assertIsNone(specification.humidity, f"Expected None for humidity with input {data['humidity']}")
                self.assertIsNone(specification.ph, f"Expected None for ph with input {data['ph']}")
                self.assertIsNone(specification.solubility, f"Expected None for solubility with input {data['solubility']}")


class TestNPKValidation(unittest.TestCase):

    def setUp(self):
        self.valid_npk_data = [
            "10-5-20",   # Simple valid case
            "0-0-0",     # Edge case where all values are 0
            "100-200-300"  # Large valid numbers
        ]
        
        self.invalid_npk_data = [
            "10-abc-20",         # Non-numeric middle value
            "20-10",             # Missing one value (invalid format)
            "10--20",            # Double dash (invalid format)
            "10-5-20-30",        # Too many values (invalid format)
            "10-5",              # Only two values (invalid format)
            "10-5-x",            # Non-numeric last value
            "-10-5-20",          # Negative first value
            "10--5-20",          # Negative middle value formatted incorrectly
            "10-5-2000000",      # Very large third value (out of typical range, if there’s one)
            "abc-def-ghi",       # Completely non-numeric input
            "10- 5 - 20",        # Whitespace around the numbers (invalid format)
            "10:5:20",           # Using colons instead of dashes
            "10,5,20",           # Using commas instead of dashes
            "",                  # Empty string (invalid)
            " ",                 # Whitespace-only string
            "10-5-"              # Missing third value
        ]

    def test_valid_npk(self):
        for npk in self.valid_npk_data:
            with self.subTest(npk=npk):
                inspection = FertilizerInspection(npk=npk)
                self.assertEqual(inspection.npk, npk)

    def test_invalid_npk(self):
        for npk in self.invalid_npk_data:
            with self.subTest(npk=npk):
                with self.assertRaises(npkError, msg=f"Expected npkError for npk input: {npk}"):
                    FertilizerInspection(npk=npk)

if __name__ == '__main__':
    unittest.main()
