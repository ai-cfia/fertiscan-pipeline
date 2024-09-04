import unittest
from pydantic import ValidationError
from pipeline import FertilizerInspection

class TestFertiliserInspection(unittest.TestCase):
    def setUp(self):
        self.data = {
            "company_name": "ABC Company",
            "company_address": "123 Main St",
            "fertiliser_name": "Super Fertiliser",
            "instructions_en": ["Use as directed"],
            "guaranteed_analysis_en": {
                "title": "Guaranteed Analysis",
                "nutrients": [{"nutrient": "Nitrogen", "value": "10", "unit": "%"}]
            }
        }

    def test_valid_fertiliser_form(self):
        # print(FertilizerInspection.model_json_schema())

        try:
            form = FertilizerInspection(**self.data)
        except ValidationError as e:
            self.fail(f"Validation error: {e}")

        raw_form = form.model_dump()

        # Check if values match
        for key, expected_value in self.data.items():
            value = raw_form[key]
            self.assertEqual(expected_value, value, f"Value for key '{key}' does not match. Expected '{expected_value}', got '{value}'")

    def test_invalid_npk_format(self):
        with self.assertRaises(ValidationError):
            FertilizerInspection(npk="invalid-format")

    def test_valid_npk_format(self):
        try:
            FertilizerInspection(**self.data, npk="10.5-20-30")
            FertilizerInspection(**self.data, npk="10.5-20.5-30")
            FertilizerInspection(**self.data, npk="10.5-0.5-30.1")
            FertilizerInspection(**self.data, npk="0-20.5-30.1")
            FertilizerInspection(**self.data, npk="0-20.5-1")
            FertilizerInspection(**self.data, npk="20.5-1-30.1")
        except ValidationError as e:
            self.fail(f"Validation error: {e}")
            
if __name__ == '__main__':
    unittest.main()
