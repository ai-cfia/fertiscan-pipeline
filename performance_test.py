import os
import time
import json
import shutil
import datetime
import csv
from typing import Dict, List, Union

from dotenv import load_dotenv
from pipeline import analyze, LabelStorage, OCR, GPT
from tests import levenshtein_similarity

ACCURACY_THRESHOLD = 80.0

def validate_environment_variables() -> None:
    """Ensure all required environment variables are set."""
    required_vars = [
        "AZURE_API_ENDPOINT",
        "AZURE_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_KEY",
        "AZURE_OPENAI_DEPLOYMENT",
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

def classify_test_result(score: float) -> str:
    """Classify test results as Pass or Fail based on accuracy score."""
    return "Pass" if score >= ACCURACY_THRESHOLD else "Fail"

def load_json_file(file_path: str) -> Dict:
    """Load and return JSON content from a file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def extract_leaf_fields(data: Union[dict, list], parent_key: str = '') -> Dict[str, Union[str, int, float, bool, None]]:
    """Extract all leaf fields from nested dictionaries and lists."""
    leaves = {}

    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, (dict, list)):
                leaves.update(extract_leaf_fields(value, new_key))
            else:
                leaves[new_key] = value
    elif isinstance(data, list):
        for index, item in enumerate(data):
            list_key = f"{parent_key}[{index}]"
            leaves.update(extract_leaf_fields(item, list_key))

    return leaves

class TestCase:
    def __init__(self, image_paths: List[str], expected_json_path: str):
        """Initialize a test case with image paths and the expected JSON output path."""
        self.original_image_paths = image_paths
        self.image_paths = self._create_image_copies(image_paths)
        self.expected_json_path = expected_json_path
        self.actual_json_path = self._generate_output_path()
        self.results: Dict[str, float] = {}
        self.label_storage = self._initialize_label_storage()
        self.ocr = self._initialize_ocr()
        self.gpt = self._initialize_gpt()

    def _create_image_copies(self, image_paths: List[str]) -> List[str]:
        """Create copies of the input images to prevent deletion of original files."""
        return [self._copy_image(path) for path in image_paths]

    def _copy_image(self, image_path: str) -> str:
        """Create a copy of a single image file."""
        base, ext = os.path.splitext(image_path)
        copy_path = f"{base}_copy{ext}"
        shutil.copy2(image_path, copy_path)
        return copy_path

    def _generate_output_path(self) -> str:
        """Generate a timestamped path for the actual output JSON."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        os.makedirs("test_outputs", exist_ok=True)
        return os.path.join("test_outputs", f'actual_output_{timestamp}.json')

    def _initialize_label_storage(self) -> LabelStorage:
        """Initialize and populate LabelStorage with image paths."""
        storage = LabelStorage()
        for image_path in self.image_paths:
            storage.add_image(image_path)
        return storage

    def _initialize_ocr(self) -> OCR:
        """Initialize OCR with API credentials."""
        return OCR(os.getenv("AZURE_API_ENDPOINT"), os.getenv("AZURE_API_KEY"))

    def _initialize_gpt(self) -> GPT:
        """Initialize GPT with API credentials."""
        return GPT(os.getenv("AZURE_OPENAI_ENDPOINT"), os.getenv("AZURE_OPENAI_KEY"), os.getenv("AZURE_OPENAI_DEPLOYMENT"))

    def save_json_output(self, output: str) -> None:
        """Save the actual output JSON to a file."""
        with open(self.actual_json_path, 'w') as file:
            file.write(output)

    def run_tests(self) -> None:
        """Run performance and accuracy tests for the pipeline."""
        self._run_performance_test()
        self._run_accuracy_test()

    def _run_performance_test(self) -> None:
        """Measure the time taken to run the pipeline analysis."""
        start_time = time.time()
        actual_output = analyze(self.label_storage, self.ocr, self.gpt)
        end_time = time.time()
        self.results['performance'] = end_time - start_time
        self.save_json_output(actual_output.model_dump_json(indent=2))

    def _run_accuracy_test(self) -> None:
        """Calculate and store the accuracy of the pipeline's output."""
        self.results['accuracy'] = self._calculate_levenshtein_accuracy()

    def _calculate_levenshtein_accuracy(self) -> Dict[str, float]:
        """Calculate Levenshtein accuracy per field between expected and actual output."""
        expected_fields = extract_leaf_fields(load_json_file(self.expected_json_path))
        actual_fields = extract_leaf_fields(load_json_file(self.actual_json_path))

        return {
            field_name: levenshtein_similarity(str(field_value), str(actual_fields.get(field_name)))
            for field_name, field_value in expected_fields.items()
        }

class TestRunner:
    def __init__(self, test_cases: List[TestCase]):
        """Initialize the test runner with a list of test cases."""
        self.test_cases = test_cases

    def run_tests(self) -> None:
        """Run all test cases."""
        for test_case in self.test_cases:
            test_case.run_tests()

    def generate_csv_report(self) -> None:
        """Generate a CSV report of the test results and save it as a timestamped .csv file."""
        report_path = self._generate_report_path()
        with open(report_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self._get_csv_header())
            self._write_test_results(writer)
        print(f"CSV report generated and saved to: {report_path}")

    def _generate_report_path(self) -> str:
        """Generate a timestamped path for the CSV report."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
        os.makedirs("reports", exist_ok=True)
        return os.path.join("reports", f"test_results_{timestamp}.csv")

    def _get_csv_header(self) -> List[str]:
        """Return the CSV header row."""
        return ["Test Case", "Field Name", "Accuracy Score", "Expected Value", "Actual Value", "Pass/Fail", "Pipeline Speed (seconds)"]

    def _write_test_results(self, writer: csv.writer) -> None:
        """Write the test results for each test case to the CSV file."""
        for i, test_case in enumerate(self.test_cases, 1):
            expected_fields = extract_leaf_fields(load_json_file(test_case.expected_json_path))
            actual_fields = extract_leaf_fields(load_json_file(test_case.actual_json_path))
            performance = test_case.results['performance']

            for field_name, score in test_case.results['accuracy'].items():
                writer.writerow([
                    f"{i}",
                    field_name,
                    f"{score:.2f}",
                    expected_fields.get(field_name, ""),
                    actual_fields.get(field_name, ""),
                    classify_test_result(score),
                    f"{performance:.4f}"
                ])

def find_test_cases(labels_folder: str) -> List[TestCase]:
    """Find and create test cases from the labels folder."""
    test_cases = []
    for root, _, files in os.walk(labels_folder):
        image_paths = [os.path.join(root, f) for f in files if f.lower().endswith((".png", ".jpg"))]
        expected_json_path = os.path.join(root, "expected_output.json")
        if image_paths and os.path.exists(expected_json_path):
            test_cases.append(TestCase(image_paths, expected_json_path))
    return test_cases

def main():
    """Main function to run the performance tests."""
    load_dotenv()
    validate_environment_variables()

    test_cases = find_test_cases("test_data/labels")
    runner = TestRunner(test_cases)
    runner.run_tests()
    runner.generate_csv_report()

if __name__ == "__main__":
    main()
