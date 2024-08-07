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
            "micronutrients_en": [{"nutrient": "Iron", "value": "2", "unit": "%"}],
            "specifications_en": [{"humidity": "Low", "ph": "7", "solubility": "High"}],
            "guaranteed_analysis": [{"nutrient": "Nitrogen", "value": "10", "unit": "%"}]
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

    def test_invalid_npk_format(self):
        with self.assertRaises(ValidationError):
            FertilizerInspection(npk="invalid-format")

    def test_valid_npk_format(self):
        try:
            FertilizerInspection(npk="10.5-20-30")
            FertilizerInspection(npk="10.5-20.5-30")
            FertilizerInspection(npk="10.5-0.5-30.1")
            FertilizerInspection(npk="0-20.5-30.1")
            FertilizerInspection(npk="0-20.5-1")
            FertilizerInspection(npk="20.5-1-30.1")
        except ValidationError as e:
            self.fail(f"Validation error: {e}")
            
if __name__ == '__main__':
    unittest.main()
