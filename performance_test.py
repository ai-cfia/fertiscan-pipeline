import os
import time
import json
import shutil
import datetime
from typing import Dict, List

from dotenv import load_dotenv
from pipeline import analyze, LabelStorage, OCR, GPT
from tests import levenshtein_similarity


def validate_env_vars() -> None:
    """Ensure all required environment variables are set.

    This function checks for the presence of the environment variables needed for 
    the pipeline to function correctly. If any required variables are missing, 
    it raises an EnvironmentError.
    """
    required_vars = [
        "AZURE_API_ENDPOINT",
        "AZURE_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_KEY",
        "AZURE_OPENAI_DEPLOYMENT",
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {
                               ', '.join(missing_vars)}")


# Load environment variables
load_dotenv()

DOCUMENT_INTELLIGENCE_API_ENDPOINT = os.getenv("AZURE_API_ENDPOINT")
DOCUMENT_INTELLIGENCE_API_KEY = os.getenv("AZURE_API_KEY")
OPENAI_API_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_KEY = os.getenv("AZURE_OPENAI_KEY")
OPENAI_DEPLOYMENT_ID = os.getenv("AZURE_OPENAI_DEPLOYMENT")


class TestCase:
    def __init__(self, image_paths: List[str], expected_json_path: str):
        """
        Initialize a test case with image paths and the expected JSON output path.
        """
        self.original_image_paths = image_paths

        # Note: The pipeline's `LabelStorage.clear()` method removes added images.
        # To preserve the original test data for future runs, copies are created and used instead.
        # This is clearly a workaround that will need to be addressed properly eventually.
        self.image_paths = self._create_image_copies(image_paths)

        self.expected_json_path = expected_json_path

        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        os.makedirs("test_outputs", exist_ok=True)
        self.actual_json_path = os.path.join(
            "test_outputs", f'actual_output_{timestamp}.json')

        self.results: Dict[str, float] = {}
        self.label_storage = LabelStorage()
        for image_path in self.image_paths:
            self.label_storage.add_image(image_path)
        self.ocr = OCR(DOCUMENT_INTELLIGENCE_API_ENDPOINT,
                       DOCUMENT_INTELLIGENCE_API_KEY)
        self.gpt = GPT(OPENAI_API_ENDPOINT, OPENAI_API_KEY,
                       OPENAI_DEPLOYMENT_ID)

    def _create_image_copies(self, image_paths: List[str]) -> List[str]:
        """
        Create copies of the input images to prevent deletion of original files 
        done by the clear method of LabelStorage.
        """
        copied_paths = []
        for image_path in image_paths:
            # Create a copy of the image with a '_copy' suffix
            base, ext = os.path.splitext(image_path)
            copy_path = f"{base}_copy{ext}"
            shutil.copy2(image_path, copy_path)
            copied_paths.append(copy_path)
        return copied_paths

    def run_tests(self) -> None:
        """
        Run performance and accuracy tests for the pipeline.
        """
        self.run_performance_test()
        self.run_accuracy_test()

    def run_performance_test(self) -> None:
        """Measure the time taken to run the pipeline analysis."""
        start_time = time.time()
        actual_output = analyze(self.label_storage, self.ocr, self.gpt)
        end_time = time.time()
        self.results['performance'] = end_time - start_time

        # Save actual output to JSON file
        self.save_json_output(actual_output.model_dump_json(indent=2))

    def run_accuracy_test(self) -> None:
        """Calculate and store the accuracy of the pipeline's output."""
        self.results['accuracy'] = self.calculate_levenshtein_accuracy()

    def calculate_levenshtein_accuracy(self) -> float:
        """Calculate Levenshtein accuracy between expected and actual output."""
        expected_json = self.load_json(self.expected_json_path)
        actual_json = self.load_json(self.actual_json_path)

        expected_str = json.dumps(expected_json, sort_keys=True)
        actual_str = json.dumps(actual_json, sort_keys=True)

        return levenshtein_similarity(expected_str, actual_str)

    def save_json_output(self, output: str) -> None:
        """Save the actual output JSON to a file."""
        with open(self.actual_json_path, 'w') as f:
            f.write(output)

    @staticmethod
    def load_json(path: str) -> Dict:
        """Load and return JSON content from a file."""
        with open(path, 'r') as f:
            return json.load(f)


class TestRunner:
    def __init__(self, test_cases: List[TestCase]):
        """
        Initialize the test runner with a list of test cases.
        """
        self.test_cases = test_cases

    def run_tests(self) -> None:
        """Run all test cases."""
        for current_test_case in self.test_cases:
            current_test_case.run_tests()

    def generate_report(self) -> None:
        """Generate a report of the test results and save it as a timestamped .md file."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        os.makedirs("reports", exist_ok=True)
        report_path = os.path.join(
            "reports", f"performance_report_{timestamp}.md")

        with open(report_path, 'w') as f:
            f.write("# Performance Test Report\n\n")
            f.write(f"Generated on: {
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            for i, test_case in enumerate(self.test_cases):
                f.write(f"## Test Case {i+1}\n\n")
                f.write(
                    f"- Performance: {test_case.results['performance']:.4f} seconds\n")
                f.write(f"- Accuracy: {test_case.results['accuracy']:.2f}\n")
                f.write(
                    f"- Actual output saved to: {test_case.actual_json_path}\n")
                f.write(
                    f"- Expected output path: {test_case.expected_json_path}\n")
                f.write(
                    f"- Original image paths: {', '.join(test_case.original_image_paths)}\n\n")

        print(f"Report generated and saved to: {report_path}")


if __name__ == "__main__":
    validate_env_vars()

    test_cases = []
    labels_folder = "test_data/labels"

    # Recursively analyze the labels folder
    for root, dirs, files in os.walk(labels_folder):
        for dir_name in dirs:
            label_folder = os.path.join(root, dir_name)
            image_paths = []
            expected_json_path = ""

            # Find image paths and expected output for each label folder
            for file_name in os.listdir(label_folder):
                file_path = os.path.join(label_folder, file_name)
                if file_name.endswith((".png", ".jpg")):
                    image_paths.append(file_path)
                elif file_name == "expected_output.json":
                    expected_json_path = file_path

            # Create a test case if image paths and expected output are found
            if image_paths and expected_json_path:
                test_cases.append(TestCase(image_paths, expected_json_path))

    runner = TestRunner(test_cases)
    runner.run_tests()
    runner.generate_report()
