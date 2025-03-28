import unittest

from pipeline.inspection import (
    FertilizerInspection,
    GuaranteedAnalysis,
    RegistrationNumber,
    NutrientValue,
    Specification,
    Organization,
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
            {
                "nutrient": "Nitrogen",
                "value": "2 or 3",
                "unit": "mg/L",
            },  # assuming that in case of multiple values, we are ok with keeping the first one
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
                self.assertIsNone(
                    nutrient_value.value,
                    f"Expected None for value with input {data['value']}",
                )
                self.assertEqual(nutrient_value.unit, "mg/L")


class TestValue(unittest.TestCase):
    def setUp(self):
        self.valid_value_data = [
            {"value": "12.5", "unit": "kg"},
            {"value": "12.5 kg", "unit": "kg"},
            {"value": "12.5kg", "unit": "kg"},
            {"value": "~12.5", "unit": "kg"},
            {"value": "approximately 12.5", "unit": "kg"},
            {
                "value": "12.5 and 15",
                "unit": "kg",
            },  # assuming that in case of multiple values, we are ok with keeping the first one
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
                self.assertIsNone(
                    value.value, f"Expected None for value with input {data['value']}"
                )
                self.assertEqual(value.unit, "kg")


class TestSpecification(unittest.TestCase):
    def setUp(self):
        self.valid_specification_data = [
            {"humidity": "30", "ph": "6.5", "solubility": "10"},
            {"humidity": "30 percent", "ph": "6.5", "solubility": "10 mol/L"},
            {"humidity": "30percent", "ph": "6.5", "solubility": "10molL"},
            {"humidity": "~30%", "ph": "~6.5", "solubility": "~10"},
            {
                "humidity": "approximately 30%",
                "ph": "approximately 6.5",
                "solubility": "approximately 10 mol/L",
            },
            {"humidity": "~30%", "ph": "~6.5%", "solubility": "~10%"},
            {
                "humidity": "30 and 40",
                "ph": "6.5/7.0",
                "solubility": "10, 15",
            },  # assuming that in case of multiple values, we are ok with keeping the first one
        ]
        self.invalid_specification_data = [
            {"humidity": "forty", "ph": "six", "solubility": "unknown"},
            {"humidity": "", "ph": "", "solubility": ""},
            {"humidity": " ", "ph": " ", "solubility": " "},
            {"humidity": None, "ph": None, "solubility": None},
            {"humidity": True, "ph": True, "solubility": True},
            {"humidity": ["30"], "ph": ["6.5"], "solubility": ["10"]},
            {
                "humidity": {"value": "30"},
                "ph": {"value": "6.5"},
                "solubility": {"value": "10"},
            },
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
                self.assertIsNone(
                    specification.humidity,
                    f"Expected None for humidity with input {data['humidity']}",
                )
                self.assertIsNone(
                    specification.ph, f"Expected None for ph with input {data['ph']}"
                )
                self.assertIsNone(
                    specification.solubility,
                    f"Expected None for solubility with input {data['solubility']}",
                )

class TestOrganizations(unittest.TestCase):
    def setUp(self):
        self.valid_organization_data = [
            {"name": "Test Company", "address": "123 Test St", "website": "https://test.com", "phone_number": "800 640 9605"},
            {"name": "Test Manufacturer", "address": "456 Test Blvd", "website": "https://manufacturer.com", "phone_number": "800-765-4321"},
        ]

        self.invalid_organization_data = [
            {"name": "Test Company", "address": "123 Test St", "website": "https://test.com", "phone_number": "123-456-7890"},
            {"name": "Test Manufacturer", "address": "456 Test Blvd", "website": "https://manufacturer.com", "phone_number": "098-765-4321"},
        ]

    def test_valid_organization(self):
        for data in self.valid_organization_data:
            with self.subTest(data=data):
                organization = Organization(**data)
                self.assertIsNotNone(organization.name, data["name"])
                self.assertIsNotNone(organization.address, data["address"])
                self.assertIsNotNone(organization.website, data["website"])
                self.assertIsNotNone(organization.phone_number, data["phone_number"])

    def test_invalid_organization(self):
        for data in self.invalid_organization_data:
            with self.subTest(data=data):
                organization = Organization(**data)
                self.assertIsNone(
                    organization.phone_number, f"Expected None for phone_number with input {data['phone_number']}"
                )
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
                # Ensure that the values can be cast to int or float without issues
                n, p, k = inspection.npk.split("-")
                self.assertIsInstance(float(n), float)
                self.assertIsInstance(float(p), float)
                self.assertIsInstance(float(k), float)

    def test_invalid_npk(self):
        for npk in self.invalid_npk_data:
            with self.subTest(npk=npk):
                inspection = FertilizerInspection(npk=npk)
                self.assertIsNone(
                    inspection.npk, f"Expected None for npk with input {npk}"
                )

class TestRegistrationNumber(unittest.TestCase):
    def setUp(self):
        self.valid_registration_number_data = [
            "1234567A",
            "1234567B",
            "1234567C",
            "1234567D",
            "1234567E",
            "1234567F",
            "1234567G",
            "1234567H",
            "1234567I",
            "1234567J",
            "1234567K",
            "1234567L",
            "1234567M",
            "1234567N",
            "1234567O",
            "1234567P",
            "1234567Q",
            "1234567R",
            "1234567S",
            "1234567T",
            "1234567U",
            "1234567V",
            "1234567W",
            "1234567X",
            "1234567Y",
            "1234567Z",
        ]

        self.invalid_registration_number_data = [
            "1234567",
            "1234567AA",
            "1234567A1",
            "1234567A ",
            "1234567 A",
            "1234567A-",
            "1234567A.",
            "1234567A,",
        ]

        self.invalid_registration_type_data = [
            "FERTILIZER",
            "INGREDIENT",
            "PRODUCT",
            "COMPONENT",
        ]

    def test_valid_registration_number(self):
        for registration_number in self.valid_registration_number_data:
            with self.subTest(registration_number=registration_number):
                registration = RegistrationNumber(identifier=registration_number)
                self.assertEqual(registration.identifier, registration_number)
            
    def test_invalid_registration_number(self):
        for registration_number in self.invalid_registration_number_data:
            with self.subTest(registration_number=registration_number):
                self.assertIsNone(RegistrationNumber(identifier=registration_number, type="fertilizer_product").identifier)

    def test_invalid_registration_type(self):
        for registration_type in self.invalid_registration_type_data:
            with self.subTest(registration_type=registration_type):
                self.assertIsNone(RegistrationNumber(identifier="1234567A", type=registration_type).type)
    

class TestGuaranteedAnalysis(unittest.TestCase):
    def setUp(self):
        self.nutrient_1 = NutrientValue(nutrient="Nitrogen", value="2", unit="mg/L")
        self.nutrient_2 = NutrientValue(
            nutrient="Organic matter", value="15", unit="mg/L"
        )

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
            "fertiliser_name": "Test Fertilizer",
            "registration_number": [
                {
                    "identifier": "1234567A",
                    "type": "fertilizer_product",
                }
            ],
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
        inspection = FertilizerInspection.model_validate(self.default_data)
        self.assertEqual(inspection.cautions_en, [])
        self.assertEqual(inspection.cautions_fr, [])
        self.assertEqual(inspection.instructions_en, [])
        self.assertEqual(inspection.instructions_fr, [])
        self.assertEqual(inspection.ingredients_en, [])
        self.assertEqual(inspection.ingredients_fr, [])
        self.assertEqual(inspection.weight, [])


class TestFertilizerInspectionRegistrationNumber(unittest.TestCase):
    def test_registration_number_with_less_digits(self):
        instance = RegistrationNumber(identifier="1234")
        self.assertIsNone(instance.identifier)

    def test_registration_number_less_than_seven_digits(self):
        instance = RegistrationNumber(identifier="12345A")
        self.assertIsNone(instance.identifier)

    def test_registration_number_seven_digits_no_letter(self):
        instance = RegistrationNumber(identifier="1234567")
        self.assertIsNone(instance.identifier)

    def test_registration_number_seven_digits_with_lowercase_letter(self):
        instance = RegistrationNumber(identifier="1234567a")
        self.assertIsNone(instance.identifier)

    def test_registration_number_correct_format(self):
        instance = RegistrationNumber(identifier="1234567A")
        self.assertEqual(instance.identifier, "1234567A")

    def test_registration_number_extra_characters(self):
        instance = RegistrationNumber(identifier="12345678B")
        self.assertIsNone(instance.identifier)

    def test_registration_number_mixed_format(self):
        instance = RegistrationNumber(identifier="12A34567B")
        self.assertIsNone(instance.identifier)


class TestOrganizationPhoneNumberFormat(unittest.TestCase):
    def test_valid_phone_number_with_country_code(self):
        instance = Organization(phone_number="+1 800 640 9605")
        self.assertEqual(instance.phone_number, "+18006409605")

    def test_valid_phone_number_without_country_code(self):
        instance = Organization(phone_number="800 640 9605")
        self.assertEqual(instance.phone_number, "+18006409605")

    def test_phone_number_with_parentheses(self):
        instance = Organization(phone_number="(757) 321-4567")
        self.assertEqual(instance.phone_number, "+17573214567")

    def test_phone_number_with_extra_characters(self):
        instance = Organization(phone_number="+1 800 321-9605 FAX")
        self.assertIsNone(instance.phone_number)

    def test_phone_number_with_multiple_numbers(self):
        instance = Organization(
            phone_number="(757) 123-4567 (800) 456-7890, 1234567890"
        )
        self.assertIsNone(instance.phone_number)

    def test_phone_number_from_other_country(self):
        instance = Organization(phone_number="+44 20 7946 0958")
        self.assertEqual(instance.phone_number, "+442079460958")

    def test_invalid_phone_number(self):
        instance = Organization(phone_number="invalid phone")
        self.assertIsNone(instance.phone_number)

    def test_phone_number_with_invalid_format(self):
        instance = Organization(phone_number="12345")
        self.assertIsNone(instance.phone_number)


if __name__ == "__main__":
    unittest.main()

if __name__ == "__main__":
    unittest.main()
