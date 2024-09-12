import unittest
from pipeline.inspection import FertilizerInspection, NutrientValue, Value, Specification


class TestNutrientValue(unittest.TestCase):

    def setUp(self):
        self.valid_nutrient_value_data = [
            {"nutrient": "Nitrogen", "value": "2", "unit": "mg/L"},
            {"nutrient": "Nitrogen", "value": "2 mg/L", "unit": "mg/L"},
            {"nutrient": "Nitrogen", "value": "2mgl", "unit": "mg/L"},
            {"nutrient": "Nitrogen", "value": "~2", "unit": "mg/L"},
            {"nutrient": "Nitrogen", "value": "approximately 2", "unit": "mg/L"},
            {"nutrient": "Nitrogen", "value": "2 or 3", "unit": "mg/L"}  # assuming that in case of multiple values, we are ok with keeping the first one
        ]

        self.invalid_nutrient_value_data = [
            {"nutrient": "Nitrogen", "value": "mg/L", "unit": "mg/L"},
            {"nutrient": "Nitrogen", "value": "", "unit": "mg/L"},
            {"nutrient": "Nitrogen", "value": " ", "unit": "mg/L"},
            {"nutrient": "Nitrogen", "value": None, "unit": "mg/L"},
            {"nutrient": "Nitrogen", "value": True, "unit": "mg/L"},
            {"nutrient": "Nitrogen", "value": ["2"], "unit": "mg/L"},
            {"nutrient": "Nitrogen", "value": {"value": "2"}, "unit": "mg/L"},
        ]

    def test_valid_nutrient_value(self):
        for data in self.valid_nutrient_value_data:
            with self.subTest(data=data):
                nutrient_value = NutrientValue(**data)
                self.assertEqual(nutrient_value.nutrient, "Nitrogen")
                self.assertEqual(nutrient_value.value, 2.0)
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
        self.valid_value_data = [
            {"value": "12.5", "unit": "kg"},
            {"value": "12.5 kg", "unit": "kg"},
            {"value": "12.5kg", "unit": "kg"},
            {"value": "~12.5", "unit": "kg"},
            {"value": "approximately 12.5", "unit": "kg"},
            {"value": "12.5 and 15", "unit": "kg"}  # assuming that in case of multiple values, we are ok with keeping the first one
        ]

        self.invalid_value_data = [
            {"value": "abc", "unit": "kg"},
            {"value": "", "unit": "kg"},
            {"value": " ", "unit": "kg"},
            {"value": None, "unit": "kg"}, 
            {"value": True, "unit": "kg"}, 
            {"value": ["12.5"], "unit": "kg"}, 
            {"value": {"value": "12.5"}, "unit": "kg"}, 
        ]

    def test_valid_value(self):
        for data in self.valid_value_data:
            with self.subTest(data=data):
                value = Value(**data)
                self.assertEqual(value.value, 12.5)
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
        self.valid_specification_data = [
            {"humidity": "30", "ph": "6.5", "solubility": "10"},
            {"humidity": "30 percent", "ph": "6.5", "solubility": "10 mol/L"},
            {"humidity": "30percent", "ph": "6.5", "solubility": "10molL"},
            {"humidity": "~30%", "ph": "~6.5", "solubility": "~10"},
            {"humidity": "approximately 30%", "ph": "approximately 6.5", "solubility": "approximately 10 mol/L"},
            {"humidity": "~30%", "ph": "~6.5%", "solubility": "~10%"},
            {"humidity": "30 and 40", "ph": "6.5/7.0", "solubility": "10, 15"}  # assuming that in case of multiple values, we are ok with keeping the first one

        ]
        self.invalid_specification_data = [
            {"humidity": "forty", "ph": "six", "solubility": "unknown"},
            {"humidity": "", "ph": "", "solubility": ""},
            {"humidity": " ", "ph": " ", "solubility": " "},
            {"humidity": None, "ph": None, "solubility": None},
            {"humidity": True, "ph": True, "solubility": True},
            {"humidity": ["30"], "ph": ["6.5"], "solubility": ["10"]},
            {"humidity": {"value": "30"}, "ph": {"value": "6.5"}, "solubility": {"value": "10"}},
        ]

    def test_valid_specification(self):
        for data in self.valid_specification_data:
            with self.subTest(data=data):
                specification = Specification(**data)
                self.assertEqual(specification.humidity, 30.0)
                self.assertEqual(specification.ph, 6.5)
                self.assertEqual(specification.solubility, 10.0)
                
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
            "10-5-20",
            "0-0-0",
            "100-200-300",
            "10.2-5.5-20.3"
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
                # Ensure that the values can be cast to int or float without issues
                n, p, k = inspection.npk.split("-")
                self.assertIsInstance(float(n), float)
                self.assertIsInstance(float(p), float)
                self.assertIsInstance(float(k), float)

    def test_invalid_npk(self):
        for npk in self.invalid_npk_data:
            with self.subTest(npk=npk):
                inspection = FertilizerInspection(npk=npk)
                self.assertIsNone(inspection.npk, f"Expected None for npk with input {npk}")


if __name__ == '__main__':
    unittest.main()
