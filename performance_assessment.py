import os
import time
import json
import shutil
import datetime
import csv
import pydantic

from dotenv import load_dotenv
from pipeline import analyze, LabelStorage, OCR, GPT, FertilizerInspection
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
    
def load_and_validate_json_inspection_file(file_path: str) -> dict:
    """Load JSON content from a file and validate it against the schema."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    try:
        # Validate against the current schema
        FertilizerInspection.model_validate(data, strict=True)
    except pydantic.ValidationError as e:
        print(f"Warning: Validation error in {file_path}.: This inspection JSON does not conform to the current inspection schema.\n")
    return data

def extract_leaf_fields(data: dict| list, parent_key: str = '') -> dict[str, str | int | float | bool | None]:
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
            list_key = f"{parent_key}[{index}]" if parent_key else f"[{index}]"
            if isinstance(item, (dict, list)):
                leaves.update(extract_leaf_fields(item, list_key))
            else:
                leaves[list_key] = item

    return leaves

class TestCase:
    def __init__(self, image_paths: list[str], expected_json_path: str):
        """Initialize a test case with image paths and the expected JSON output path."""
        self.original_image_paths : list[str] = image_paths
        self.image_paths: list[str] = self._create_image_copies(image_paths) # because the pipeline automatically deletes images when processing them
        self.expected_json_path: str = expected_json_path
        self.expected_fields: dict[str, str | int | float | bool | None] = {}
        self.actual_json_path : str = self._generate_output_path()
        self.actual_fields: dict[str, str | int | float | bool | None] = {}
        self.results: dict[str, float] = {}
        self.label_storage: LabelStorage = self._initialize_label_storage()
        self.ocr: OCR = self._initialize_ocr()
        self.gpt: GPT = self._initialize_gpt()

    def _create_image_copies(self, image_paths: list[str]) -> list[str]:
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
        return self.expected_json_path.replace("expected", "actual")

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
        
        # clean up the actual output .json files
        if os.path.exists(self.actual_json_path):
            os.remove(self.actual_json_path)

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

    def _calculate_levenshtein_accuracy(self) -> dict[str, float]:
        """Calculate Levenshtein accuracy per field between expected and actual output."""
        self.expected_fields = extract_leaf_fields(load_and_validate_json_inspection_file(self.expected_json_path))
        self.actual_fields = extract_leaf_fields(load_and_validate_json_inspection_file(self.actual_json_path))

        return {
            field_name: levenshtein_similarity(str(field_value), str(self.actual_fields.get(field_name)))
            for field_name, field_value in self.expected_fields.items()
        }

class TestRunner:
    def __init__(self, test_cases: list[TestCase]):
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

    def _get_csv_header(self) -> list[str]:
        """Return the CSV header row."""
        return ["Test Case", "Field Name", "Accuracy Score", "Expected Value", "Actual Value", "Pass/Fail", "Pipeline Speed (seconds)"]

    def _write_test_results(self, writer: csv.writer) -> None:
        """Write the test results for each test case to the CSV file."""
        for i, test_case in enumerate(self.test_cases, 1):
            performance = test_case.results['performance']

            for field_name, score in test_case.results['accuracy'].items():
                writer.writerow([
                    f"{i}",
                    field_name,
                    f"{score:.2f}",
                    test_case.expected_fields.get(field_name, ""),
                    test_case.actual_fields.get(field_name, ""),
                    classify_test_result(score),
                    f"{performance:.4f}"
                ])

def find_test_cases(labels_folder: str) -> list[TestCase]:
    """Find and create test cases from the labels folder in an ordered manner."""
    test_cases = []
    # List all entries in the labels_folder
    label_entries = os.listdir(labels_folder)
    # Filter out directories that start with 'label_'
    label_dirs = [
        os.path.join(labels_folder, d)
        for d in label_entries
        if os.path.isdir(os.path.join(labels_folder, d)) and d.startswith("label_")
    ]
    # Sort the label directories
    label_dirs.sort()
    # Process each label directory
    for label_dir in label_dirs:
        files = os.listdir(label_dir)
        image_paths = [
            os.path.join(label_dir, f)
            for f in files
            if f.lower().endswith((".png", ".jpg"))
        ]
        expected_json_path = os.path.join(label_dir, "expected_output.json")
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
