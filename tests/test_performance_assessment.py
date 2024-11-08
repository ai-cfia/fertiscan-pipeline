import json
import unittest
import os
import shutil
import csv
import tempfile
from unittest.mock import patch, MagicMock
from scripts.run_performance_assessment_data_collection import (
    extract_leaf_fields,
    find_test_cases,
    calculate_accuracy,
    run_test_case,
    generate_csv_report,
)

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

class TestFindTestCases(unittest.TestCase):

    def setUp(self):
        self.test_dir = "test_labels_folder"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def create_test_structure(self, structure):
        for path, files in structure.items():
            dir_path = os.path.join(self.test_dir, path)
            os.makedirs(dir_path, exist_ok=True)
            for file_name, content in files.items():
                with open(os.path.join(dir_path, file_name), 'w') as f:
                    f.write(content)

    def test_valid_structure(self):
        structure = {
            "label_001": {
                "image001.png": "",
                "image002.jpg": "",
                "expected_output.json": "{}"
            },
            "label_002": {
                "image001.png": "",
                "expected_output.json": "{}"
            },
            "label_003": {
                "image001.png": "",
                "image002.jpg": "",
                "expected_output.json": "{}"
            }
        }
        self.create_test_structure(structure)
        expected = [
            (
                [
                    os.path.join(self.test_dir, "label_001", "image001.png"),
                    os.path.join(self.test_dir, "label_001", "image002.jpg")
                ],
                os.path.join(self.test_dir, "label_001", "expected_output.json")
            ),
            (
                [
                    os.path.join(self.test_dir, "label_002", "image001.png")
                ],
                os.path.join(self.test_dir, "label_002", "expected_output.json")
            ),
            (
                [
                    os.path.join(self.test_dir, "label_003", "image001.png"),
                    os.path.join(self.test_dir, "label_003", "image002.jpg")
                ],
                os.path.join(self.test_dir, "label_003", "expected_output.json")
            )
        ]
        result = find_test_cases(self.test_dir)
        self.assertEqual(result, expected)

    def test_missing_expected_output(self):
        structure = {
            "label_001": {
                "image001.png": "",
                "image002.jpg": ""
            }
        }
        self.create_test_structure(structure)
        with self.assertRaises(FileNotFoundError):
            find_test_cases(self.test_dir)

    def test_no_image_files(self):
        structure = {
            "label_001": {
                "expected_output.json": "{}"
            }
        }
        self.create_test_structure(structure)
        with self.assertRaises(FileNotFoundError):
            find_test_cases(self.test_dir)

    def test_empty_labels_folder(self):
        with self.assertRaises(FileNotFoundError):
            find_test_cases(self.test_dir)


class TestCalculateAccuracy(unittest.TestCase):
    def test_calculate_accuracy_perfect_score(self):
        expected = {
            "field_1": "value_1",
            "field_2": "value_2"
        }
        actual = {
            "field_1": "value_1",
            "field_2": "value_2"
        }
        result = calculate_accuracy(expected, actual)
        for field in result.values():
            self.assertEqual(field['pass_fail'], "Pass")
            self.assertEqual(field['score'], 100.0)

    def test_calculate_accuracy_all_fail(self):
        expected = {
            "field_1": "value_1",
            "field_2": "value_2"
        }
        actual = {
            "field_1": "wrong_value",
            "field_2": "another_wrong_value"
        }
        result = calculate_accuracy(expected, actual)
        self.assertEqual(int(result['field_1']['score']), 27)
        self.assertEqual(int(result['field_2']['score']), 15)


    def test_calculate_accuracy_with_missing_field(self):
        expected = {
            "field_1": "value_1",
            "field_2": "value_2"
        }
        actual = {
            "field_1": "value_1",
        }
        result = calculate_accuracy(expected, actual)
        self.assertEqual(int(result['field_1']['score']), 100)
        self.assertEqual(int(result['field_2']['score']), 0)

class TestRunTestCase(unittest.TestCase):
    @patch("scripts.run_performance_assessment_data_collection.LabelStorage")
    @patch("scripts.run_performance_assessment_data_collection.OCR")
    @patch("scripts.run_performance_assessment_data_collection.GPT")
    @patch("scripts.run_performance_assessment_data_collection.analyze")
    def test_run_test_case(self, mock_analyze, MockGPT, MockOCR, MockLabelStorage):
        # Setting up mock returns
        mock_analyze.return_value.model_dump_json.return_value = json.dumps({
            "field_1": "value_1",
            "field_2": "value_2"
        })
        MockOCR.return_value = MagicMock()
        MockGPT.return_value = MagicMock()
        MockLabelStorage.return_value = MagicMock()
        
        # Create temporary image and expected JSON files
        temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_json = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        with open(temp_json.name, 'w') as f:
            json.dump({"field_1": "value_1", "field_2": "value_2"}, f)
        
        result = run_test_case(1, [temp_image.name], temp_json.name)
        
        self.assertEqual(result['test_case_number'], 1)
        self.assertIn('accuracy_results', result)
        self.assertIn('performance', result)
        for field in result['accuracy_results'].values():
            self.assertEqual(field['pass_fail'], "Pass")

        # Clean up temporary files
        os.unlink(temp_image.name)
        os.unlink(temp_json.name)


class TestGenerateCSVReport(unittest.TestCase):
    def setUp(self):
        self.results = [
            {
                'test_case_number': 1,
                'performance': 5.44,
                'accuracy_results': {
                    'field_1': {
                        'score': 100.0,
                        'expected_value': 'value_1',
                        'actual_value': 'value_1',
                        'pass_fail': 'Pass'
                    },
                    'field_2': {
                        'score': 100.0,
                        'expected_value': 'value_2',
                        'actual_value': 'value_2',
                        'pass_fail': 'Pass'
                    }
                }
            }
        ]
        self.report_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.report_dir)

    @patch("os.makedirs")
    @patch("os.path.join", return_value=os.path.join(tempfile.gettempdir(), "test_results.csv"))
    def test_generate_csv_report(self, mock_join, mock_makedirs):
        # Generate the CSV report
        generate_csv_report(self.results) # <- since we patched make_dirs, we don't need to clean up the directory
        report_path = mock_join.return_value

        # Check if the file was created
        self.assertTrue(os.path.exists(report_path))

        # Verify the contents of the CSV
        with open(report_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Check header
        self.assertEqual(rows[0], [
            "Test Case",
            "Field Name",
            "Pass/Fail",
            "Accuracy Score",
            "Pipeline Speed (seconds)",
            "Expected Value",
            "Actual Value"
        ])

        # Check data row
        self.assertEqual(rows[1], [
            '1',
            'field_1',
            'Pass',
            '100.00',
            '5.4400',
            'value_1',
            'value_1'
        ])


if __name__ == "__main__":
    unittest.main()
