# test_script.py

import unittest
import os
import json
from unittest.mock import patch, mock_open
import pydantic
from typing import Optional
from pydantic import BaseModel, field_validator

from performance_assessment import (
    validate_environment_variables,
    classify_test_result,
    load_and_validate_json_inspection_file,
    extract_leaf_fields,
    TestCase,
    TestRunner,
    find_test_cases
)

from pipeline.inspection import FertilizerInspection, extract_first_number

class TestValidateEnvironmentVariables(unittest.TestCase):
    @patch.dict(os.environ, {
        "AZURE_API_ENDPOINT": "endpoint",
        "AZURE_API_KEY": "key",
        "AZURE_OPENAI_ENDPOINT": "endpoint",
        "AZURE_OPENAI_KEY": "key",
        "AZURE_OPENAI_DEPLOYMENT": "deployment"
    })
    def test_validate_environment_variables_all_set(self):
        try:
            validate_environment_variables()
        except EnvironmentError:
            self.fail("validate_environment_variables() raised EnvironmentError unexpectedly!")

    @patch.dict(os.environ, {}, clear=True)
    def test_validate_environment_variables_missing(self):
        with self.assertRaises(EnvironmentError) as context:
            validate_environment_variables()
        self.assertIn("Missing required environment variables", str(context.exception))

    @patch.dict(os.environ, {
        "AZURE_API_ENDPOINT": "endpoint",
        "AZURE_API_KEY": "key",
        "AZURE_OPENAI_ENDPOINT": "endpoint"
    }, clear=True)
    def test_validate_some_environment_variables_missing(self):
        with self.assertRaises(EnvironmentError) as context:
            validate_environment_variables()
        self.assertIn("Missing required environment variables", str(context.exception))
        self.assertIn("AZURE_OPENAI_KEY", str(context.exception))
        self.assertIn("AZURE_OPENAI_DEPLOYMENT", str(context.exception))


class TestClassifyTestResult(unittest.TestCase):
    @patch('performance_assessment.ACCURACY_THRESHOLD', 80.0)
    def test_classify_test_result_pass(self):
        self.assertEqual(classify_test_result(80.0), "Pass")
        self.assertEqual(classify_test_result(90.0), "Pass")
        self.assertEqual(classify_test_result(100.0), "Pass")

    @patch('performance_assessment.ACCURACY_THRESHOLD', 80.0)
    def test_classify_test_result_fail(self):
        self.assertEqual(classify_test_result(79.9), "Fail")
        self.assertEqual(classify_test_result(50.0), "Fail")
        self.assertEqual(classify_test_result(0.0), "Fail")
        self.assertEqual(classify_test_result(-10.0), "Fail")


"""
class MockNutrientValue(BaseModel):
    nutrient: str
    value: Optional[float] = None
    unit: Optional[str] = None
    
    @field_validator('value', mode='before', check_fields=False)
    def convert_value(cls, v):
        if isinstance(v, bool):
            return None
        elif isinstance(v, (int, float)):
            return str(v)
        elif isinstance(v, (str)):
            return extract_first_number(v)
        return None

class MockGuaranteedAnalysis(BaseModel):
    title: Optional[str] = None
    nutrients: list[MockNutrientValue] = []

    @field_validator(
        "nutrients",
        mode="before",
    )
    def replace_none_with_empty_list(cls, v):
        if v is None:
            v = []
        return v
    
class MockFertilizerInspection(FertilizerInspection):
    company_name: Optional[str] = None
    guaranteed_analysis: Optional[MockGuaranteedAnalysis] = None


class TestLoadAndValidateJsonInspectionFile(unittest.TestCase):    

    def test_load_json_inspection_file_valid(self):
        test_data = {'key': 'value'}
        json_content = json.dumps(test_data)
        
        # Use mock_open to simulate file operations
        with patch('builtins.open', mock_open(read_data=json_content)) as mocked_file:
            result = load_and_validate_json_inspection_file('dummy_path.json')
            self.assertEqual(result, test_data)
            mocked_file.assert_called_once_with('dummy_path.json', 'r')

    def test_load_json_inspection_file_invalid_json(self):
        invalid_json_content = '{"key": "value"'  # Missing closing brace
        
        with patch('builtins.open', mock_open(read_data=invalid_json_content)):
            with self.assertRaises(json.JSONDecodeError):
                load_and_validate_json_inspection_file('dummy_path.json')

    def test_load_json_inspection_file_file_not_found(self):
        # Simulate FileNotFoundError when attempting to open the file
        with patch('builtins.open', side_effect=FileNotFoundError):
            with self.assertRaises(FileNotFoundError):
                load_and_validate_json_inspection_file('nonexistent_file.json')

    def test_load_json_inspection_file_permission_error(self):
        # Simulate PermissionError when attempting to open the file
        with patch('builtins.open', side_effect=PermissionError):
            with self.assertRaises(PermissionError):
                load_and_validate_json_inspection_file('protected_file.json')

    @patch('pipeline.inspection.FertilizerInspection', MockFertilizerInspection)
    def test_load_json_inspection_file_validation_is_valid(self):
        mock_data = {
            "company_name": "Nature's aid",
            "guaranteed_analysis": {
                "title": "Analyse Garantie",
                "nutrients": [
                    {
                        "nutrient": "extraits d'algues (ascophylle noueuse)",
                        "value": 8.5,
                        "unit": "%"
                    },
                    {
                        "nutrient": "acide humique",
                        "value": 0.6,
                        "unit": "%"
                    }
                ]
            }
        }
        json_content = json.dumps(mock_data)

        with patch('builtins.open', mock_open(read_data=json_content)) as mocked_file:
            result = load_and_validate_json_inspection_file('dummy_path.json')
            self.assertEqual(result.company_name, "Nature's aid")
            self.assertEqual(result.guaranteed_analysis.title, "Analyse Garantie")
            self.assertEqual(result.guaranteed_analysis.nutrients[0].nutrient, "extraits d'algues (ascophylle noueuse)")
            self.assertEqual(result.guaranteed_analysis.nutrients[0].value, 8.5)
            self.assertEqual(result.guaranteed_analysis.nutrients[0].unit, "%")
            self.assertEqual(result.guaranteed_analysis.nutrients[1].nutrient, "acide humique")
            self.assertEqual(result.guaranteed_analysis.nutrients[1].value, 0.6)
            self.assertEqual(result.guaranteed_analysis.nutrients[1].unit, "%")

            mocked_file.assert_called_once_with('dummy_path.json', 'r')

    @patch('pipeline.inspection.FertilizerInspection', MockFertilizerInspection)
    def test_load_json_inspection_file_validation_is_invalid(self):
        mock_data = {
            "company_name": "Nature's aid",
            "guaranteed_analysis": [
                {
                    "nutrients": "extraits d'algues (ascophylle noueuse)",
                    "value": 8.5,
                    "unit": "%"
                },
                {
                    "nutrient": "acide humique",
                    "value": 0.6,
                    "unit": "%"
                }
            ]
        }

        json_content = json.dumps(mock_data)

        with patch('builtins.open', mock_open(read_data=json_content)):
            self.assertRaises(pydantic.ValidationError, load_and_validate_json_inspection_file('dummy_path.json'))
"""

class TestExtractLeafFields(unittest.TestCase):
    def test_extract_leaf_fields_simple_root_dict(self):
        mock_input = {
            "company_name": "Nature's aid",
            "company_address": None,
            "company_website": "http://www.SOIL-AID.com",
            "company_phone_number": None,
            "manufacturer_name": "Diamond Fertilizers Inc.",
            "manufacturer_address": "PO Box 5508 stn Main Hight River, AB CANADA T1V 1M6",
            "manufacturer_website": None,
        }
        
        expected_output = {
            "company_name": "Nature's aid",
            "company_address": None,
            "company_website": "http://www.SOIL-AID.com",
            "company_phone_number": None,
            "manufacturer_name": "Diamond Fertilizers Inc.",
            "manufacturer_address": "PO Box 5508 stn Main Hight River, AB CANADA T1V 1M6",
            "manufacturer_website": None,
        }  

        actual_output = extract_leaf_fields(mock_input)
        self.assertEqual(actual_output, expected_output)

    def test_extract_leaf_fields_simple_child_dict(self):
        mock_input = {
            "density": {
                "value": None,
                "unit": None
            },
            "volume": {
                "value": 10,
                "unit": "liter"
            },
        }

        expected_output = {
            "density.value": None,
            "density.unit": None,
            "volume.value": 10,
            "volume.unit": "liter"
        }

        actual_output = extract_leaf_fields(mock_input)
        self.assertEqual(actual_output, expected_output)

    def test_extract_leaf_fields_simple_root_list(self):
        mock_input = [
            "step 1: prepare the soil",
            "step 2: plant the seeds",
            "step 3: water the plants",
        ]

        expected_output = {
            "[0]": "step 1: prepare the soil",
            "[1]": "step 2: plant the seeds",
            "[2]": "step 3: water the plants",
        }

        actual_output = extract_leaf_fields(mock_input)
        self.assertEqual(actual_output, expected_output)

    def test_extract_leaf_fields_simple_child_list(self):
        mock_input = {
            "steps": [
                "step 1: prepare the soil",
                "step 2: plant the seeds",
                "step 3: water the plants",
            ],
            "materials": [
                "soil",
                "seeds",
                "water",
            ]
        }

        expected_output = {
            "steps[0]": "step 1: prepare the soil",
            "steps[1]": "step 2: plant the seeds",
            "steps[2]": "step 3: water the plants",
            "materials[0]": "soil",
            "materials[1]": "seeds",
            "materials[2]": "water",
        }

        actual_output = extract_leaf_fields(mock_input)
        self.assertEqual(actual_output, expected_output)

    def test_extract_leaf_fields_mixed_structure(self):
        mock_input = {
            "guaranteed_analysis_fr": {
                "title": "Analyse Garantie",
                "nutrients": [
                    {
                    "nutrient": "extraits d'algues (ascophylle noueuse)",
                    "value": 8.5,
                    "unit": "%"
                    },
                    {
                    "nutrient": "acide humique",
                    "value": 0.6,
                    "unit": "%"
                    }
                ]
            },
            "cautions_en": None,
            "cautions_fr": None,
            "instructions_en": [
                "step 1: prepare the soil",
                "step 2: plant the seeds",
                "step 3: water the plants",
            ],
        }

        expected_output = {
            "guaranteed_analysis_fr.title": "Analyse Garantie",
            "guaranteed_analysis_fr.nutrients[0].nutrient": "extraits d'algues (ascophylle noueuse)",
            "guaranteed_analysis_fr.nutrients[0].value": 8.5,
            "guaranteed_analysis_fr.nutrients[0].unit": "%",
            "guaranteed_analysis_fr.nutrients[1].nutrient": "acide humique",
            "guaranteed_analysis_fr.nutrients[1].value": 0.6,
            "guaranteed_analysis_fr.nutrients[1].unit": "%",
            "cautions_en": None,
            "cautions_fr": None,
            "instructions_en[0]": "step 1: prepare the soil",
            "instructions_en[1]": "step 2: plant the seeds",
            "instructions_en[2]": "step 3: water the plants",
        }

        actual_output = extract_leaf_fields(mock_input)
        self.assertEqual(actual_output, expected_output)

if __name__ == '__main__':
    unittest.main()



