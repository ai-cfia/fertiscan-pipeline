import unittest

from pipeline.schemas.inspection import (
    FertilizerInspection,
    GuaranteedAnalysis,
    NutrientValue,
    Specification,
    Value,
)


class TestNutrientValue(unittest.TestCase):
    def setUp(self):
        self.valid_nutrient_value_data = [
            {"nutrient": "Nitrogen", "value": "2", "unit": "mg/L"},
            {"nutrient": "Nitrogen", "value": "2 mg/L", "unit": "mg/L"},
            {"nutrient": "Nitrogen", "value": "2mgl", "unit": "mg/L"},
            {"nutrient": "Nitrogen", "value": "~2", "unit": "mg/L"},
            {"nutrient": "Nitrogen", "value": "approximately 2", "unit": "mg/L"},
            {"nutrient": "Nitrogen", "value": "2 or 3", "unit": "mg/L"},
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

                self.assertIsInstance(int(nutrient_value.value), int)
                self.assertIsInstance(float(nutrient_value.value), float)

    def test_invalid_nutrient_value(self):
        for data in self.invalid_nutrient_value_data:
            with self.subTest(data=data):
                nutrient_value = NutrientValue(**data)
                self.assertEqual(nutrient_value.nutrient, "Nitrogen")
                self.assertIsNone(nutrient_value.value)
                self.assertEqual(nutrient_value.unit, "mg/L")


class TestValue(unittest.TestCase):
    def setUp(self):
        self.valid_value_data = [
            {"value": "12.5", "unit": "kg"},
            {"value": "12.5 kg", "unit": "kg"},
            {"value": "12.5kg", "unit": "kg"},
            {"value": "~12.5", "unit": "kg"},
            {"value": "approximately 12.5", "unit": "kg"},
            {"value": "12.5 and 15", "unit": "kg"},
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

                self.assertIsInstance(int(value.value), int)
                self.assertIsInstance(float(value.value), float)

    def test_invalid_value(self):
        for data in self.invalid_value_data:
            with self.subTest(data=data):
                value = Value(**data)
                self.assertIsNone(value.value)
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
            {"humidity": "30 and 40", "ph": "6.5/7.0", "solubility": "10, 15"},
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

                self.assertIsInstance(int(specification.humidity), int)
                self.assertIsInstance(float(specification.ph), float)
                self.assertIsInstance(int(specification.solubility), int)

    def test_invalid_specification(self):
        for data in self.invalid_specification_data:
            with self.subTest(data=data):
                specification = Specification(**data)
                self.assertIsNone(specification.humidity)
                self.assertIsNone(specification.ph)
                self.assertIsNone(specification.solubility)


class TestNPKValidation(unittest.TestCase):
    def setUp(self):
        self.valid_npk_data = ["10-5-20", "0-0-0", "100-200-300", "10.2-5.5-20.3"]

        self.invalid_npk_data = [
            "10-abc-20",  # Non-numeric middle value
            "20-10",  # Missing one value (invalid format)
            "10--20",  # Double dash (invalid format)
            "10-5-20-30",  # Too many values (invalid format)
            "10-5",  # Only two values (invalid format)
            "10-5-x",  # Non-numeric last value
            "-10-5-20",  # Negative first value
            "10--5-20",  # Negative middle value formatted incorrectly
            "abc-def-ghi",  # Completely non-numeric input
            "10- 5 - 20",  # Whitespace around the numbers (invalid format)
            "10:5:20",  # Using colons instead of dashes
            "10,5,20",  # Using commas instead of dashes
            "",  # Empty string (invalid)
            " ",  # Whitespace-only string
            "10-5-",  # Missing third value
        ]

    def test_valid_npk(self):
        for npk in self.valid_npk_data:
            with self.subTest(npk=npk):
                inspection = FertilizerInspection(npk=npk)
                self.assertEqual(inspection.npk, npk)
                n, p, k = inspection.npk.split("-")
                self.assertIsInstance(float(n), float)
                self.assertIsInstance(float(p), float)
                self.assertIsInstance(float(k), float)

    def test_invalid_npk(self):
        for npk in self.invalid_npk_data:
            with self.subTest(npk=npk):
                inspection = FertilizerInspection(npk=npk)
                self.assertIsNone(inspection.npk)


class TestGuaranteedAnalysis(unittest.TestCase):
    def setUp(self):
        self.nutrient_1 = NutrientValue(nutrient="Nitrogen", value="2", unit="mg/L")
        self.nutrient_2 = NutrientValue(nutrient="Organic matter", value="15", unit="mg/L")

    def test_set_is_minimal(self):
        guaranteed_analysis = GuaranteedAnalysis(
            title="Guaranteed minimum analysis",
            nutrients=[self.nutrient_1, self.nutrient_2],
        )
        self.assertTrue(guaranteed_analysis.is_minimal)

    def test_set_is_not_minimal(self):
        guaranteed_analysis = GuaranteedAnalysis(
            title="Guaranteed analysis", nutrients=[self.nutrient_1, self.nutrient_2]
        )
        self.assertFalse(guaranteed_analysis.is_minimal)

    def test_is_minimal_in_none(self):
        guaranteed_analysis = GuaranteedAnalysis(
            nutrients=[self.nutrient_1, self.nutrient_2]
        )
        self.assertIsNone(guaranteed_analysis.is_minimal)


class TestFertilizerInspectionListFields(unittest.TestCase):
    def setUp(self):
        self.default_data = {
            "organizations": [],
            "fertiliser_name": "Test Fertilizer",
            "registration_number": [],
            "lot_number": "LOT987",
            "npk": "10-5-20",
            "guaranteed_analysis_en": None,
            "guaranteed_analysis_fr": None,
            "cautions_en": None,
            "cautions_fr": None,
            "instructions_en": None,
            "instructions_fr": None,
            "ingredients_en": None,
            "ingredients_fr": None,
            "weight": None,
        }

    def test_replace_none_with_empty_list(self):
        inspection = FertilizerInspection(**self.default_data)
        self.assertEqual(inspection.cautions_en, [])
        self.assertEqual(inspection.cautions_fr, [])
        self.assertEqual(inspection.instructions_en, [])
        self.assertEqual(inspection.instructions_fr, [])
        self.assertEqual(inspection.ingredients_en, [])
        self.assertEqual(inspection.ingredients_fr, [])
        self.assertEqual(inspection.weight, [])


class TestFertilizerInspectionRegistrationNumber(unittest.TestCase):
    def test_registration_number_with_less_digits(self):
        instance = FertilizerInspection(registration_number=[{"identifier": "1234"}])
        self.assertIsNone(instance.registration_number[0].identifier)

    def test_registration_number_less_than_seven_digits(self):
        instance = FertilizerInspection(registration_number=[{"identifier": "12345A"}])
        self.assertIsNone(instance.registration_number[0].identifier)

    def test_registration_number_seven_digits_no_letter(self):
        instance = FertilizerInspection(registration_number=[{"identifier": "1234567"}])
        self.assertIsNone(instance.registration_number[0].identifier)

    def test_registration_number_seven_digits_with_lowercase_letter(self):
        instance = FertilizerInspection(registration_number=[{"identifier": "1234567a"}])
        self.assertIsNone(instance.registration_number[0].identifier)

    def test_registration_number_correct_format(self):
        instance = FertilizerInspection(registration_number=[{"identifier": "1234567A"}])
        self.assertEqual(instance.registration_number[0].identifier, "1234567A")

    def test_registration_number_extra_characters(self):
        instance = FertilizerInspection(registration_number=[{"identifier": "12345678B"}])
        self.assertIsNone(instance.registration_number[0].identifier)

    def test_registration_number_mixed_format(self):
        instance = FertilizerInspection(registration_number=[{"identifier": "12A34567B"}])
        self.assertIsNone(instance.registration_number[0].identifier)


class TestFertilizerInspectionPhoneNumberFormat(unittest.TestCase):
    def test_valid_phone_number_with_country_code(self):
        instance = FertilizerInspection(organizations=[{"phone_number": "+1 800 640 9605"}])
        self.assertEqual(instance.organizations[0].phone_number, "+18006409605")

    def test_valid_phone_number_without_country_code(self):
        instance = FertilizerInspection(organizations=[{"phone_number": "800 640 9605"}])
        self.assertEqual(instance.organizations[0].phone_number, "+18006409605")

    def test_phone_number_with_parentheses(self):
        instance = FertilizerInspection(organizations=[{"phone_number": "(757) 321-4567"}])
        self.assertEqual(instance.organizations[0].phone_number, "+17573214567")

    def test_phone_number_with_extra_characters(self):
        instance = FertilizerInspection(organizations=[{"phone_number": "+1 800 321-9605 FAX"}])
        self.assertIsNone(instance.organizations[0].phone_number)

    def test_phone_number_with_multiple_numbers(self):
        instance = FertilizerInspection(organizations=[{"phone_number": "(757) 123-4567 (800) 456-7890, 1234567890"}])
        self.assertIsNone(instance.organizations[0].phone_number)

    def test_phone_number_from_other_country(self):
        instance = FertilizerInspection(organizations=[{"phone_number": "+44 20 7946 0958"}])
        self.assertEqual(instance.organizations[0].phone_number, "+442079460958")

    def test_invalid_phone_number(self):
        instance = FertilizerInspection(organizations=[{"phone_number": "invalid phone"}])
        self.assertIsNone(instance.organizations[0].phone_number)

    def test_phone_number_with_invalid_format(self):
        instance = FertilizerInspection(organizations=[{"phone_number": "12345"}])
        self.assertIsNone(instance.organizations[0].phone_number)


if __name__ == "__main__":
    unittest.main()
