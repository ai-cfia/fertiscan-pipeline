import unittest
from pydantic import ValidationError
from pipeline import FertilizerInspection

class TestFertiliserForm(unittest.TestCase):
    def test_valid_fertiliser_form(self):
        data = {
            "company_name": "ABC Company",
            "company_address": "123 Main St",
            "fertiliser_name": "Super Fertiliser",
            "npk": "10-5-5",
            "instructions_en": ["Use as directed"],
            "micronutrients_en": [{"nutrient": "Iron", "value": 2.0, "unit": "%"}],
            "specifications_en": [{"humidity": 23.0, "ph": 7.0, "solubility": 4.0}],
            "guaranteed_analysis": [{"nutrient": "Nitrogen", "value": 10.0, "unit": "%"}]
        }

        try:
            form = FertilizerInspection(**data)
        except ValidationError as e:
            self.fail(f"Validation error: {e}")

        raw_form = form.model_dump()

        # Check if values match
        for key, expected_value in data.items():
            value = raw_form[key]
            self.assertEqual(expected_value, value, f"Value for key '{key}' does not match. Expected '{expected_value}', got '{value}'")

            
if __name__ == '__main__':
    unittest.main()
