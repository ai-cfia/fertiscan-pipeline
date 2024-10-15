import os
import time
import json
import shutil
import datetime
import csv
import tempfile
from dotenv import load_dotenv
from pipeline import analyze, LabelStorage, OCR, GPT
from tests import levenshtein_similarity

ACCURACY_THRESHOLD = 80.0


def extract_leaf_fields(
    data: dict | list, parent_key: str = ''
) -> dict[str, str | int | float | bool | None]:
    leaves: dict[str, str | int | float | bool | None] = {}

    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, (dict, list)):
                leaves.update(extract_leaf_fields(value, new_key))
            else:
                leaves[new_key] = value
    elif isinstance(data, list):
        for index, item in enumerate(data):
            new_key = f"{parent_key}[{index}]" if parent_key else f"[{index}]"
            if isinstance(item, (dict, list)):
                leaves.update(extract_leaf_fields(item, new_key))
            else:
                leaves[new_key] = item

    return leaves


def find_test_cases(labels_folder: str) -> list[tuple[list[str], str]]:
    test_cases = []
    label_directories = sorted(
        os.path.join(labels_folder, directory)
        for directory in os.listdir(labels_folder)
        if os.path.isdir(os.path.join(labels_folder, directory)) and directory.startswith("label_")
    )
    if len(label_directories) == 0:
        raise FileNotFoundError(f"No label directories found in {labels_folder}")

    for label_directory in label_directories:
        files = os.listdir(label_directory)
        image_paths = [
            os.path.join(label_directory, file)
            for file in files
            if file.lower().endswith((".png", ".jpg"))
        ]
        expected_json_path = os.path.join(label_directory, "expected_output.json")

        if not image_paths:
            raise FileNotFoundError(f"No image files found in {label_directory}")
        if not os.path.exists(expected_json_path):
            raise FileNotFoundError(f"Expected output JSON not found in {label_directory}")
        test_cases.append((image_paths, expected_json_path))

    return test_cases


def calculate_accuracy(
    expected_fields: dict[str, str],
    actual_fields: dict[str, str]
) -> dict[str, dict[str, str | float]]:
    accuracy_results = {}
    for field_name, expected_value in expected_fields.items():
        actual_value = actual_fields.get(field_name, "FIELD_NOT_FOUND")
        if actual_value == "FIELD_NOT_FOUND":
            score = 0.0
        else:
            score = levenshtein_similarity(str(expected_value), str(actual_value))
        pass_fail = "Pass" if score >= ACCURACY_THRESHOLD else "Fail"
        accuracy_results[field_name] = {
            'score': score,
            'expected_value': expected_value,
            'actual_value': actual_value,
            'pass_fail': pass_fail,
        }
    return accuracy_results


def run_test_case(
    test_case_number: int, image_paths: list[str], expected_json_path: str
) -> dict[str, any]:
    # Copy images to temporary files to prevent deletion due to LabelStorage behavior
    copied_image_paths = []
    for image_path in image_paths:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image_path)[1])
        shutil.copy2(image_path, temp_file.name)
        copied_image_paths.append(temp_file.name)

    # Initialize LabelStorage, OCR, GPT
    storage = LabelStorage()
    for image_path in copied_image_paths:
        storage.add_image(image_path)

    ocr = OCR(os.getenv("AZURE_API_ENDPOINT"), os.getenv("AZURE_API_KEY"))
    gpt = GPT(
        os.getenv("AZURE_OPENAI_ENDPOINT"),
        os.getenv("AZURE_OPENAI_KEY"),
        os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    )

    # Run performance test
    start_time = time.time()
    actual_output = analyze(storage, ocr, gpt) # <-- the `analyse` function deletes the images it processes so we don't need to clean up our image copies
    performance = time.time() - start_time

    # Process actual output
    actual_fields = extract_leaf_fields(json.loads(actual_output.model_dump_json()))

    # Load expected output
    with open(expected_json_path, 'r') as file:
        expected_fields = extract_leaf_fields(json.load(file))

    # Calculate accuracy
    accuracy_results = calculate_accuracy(expected_fields, actual_fields)

    # Return results
    return {
        'test_case_number': test_case_number,
        'performance': performance,
        'accuracy_results': accuracy_results,
    }


def generate_csv_report(results: list[dict[str, any]]) -> None:
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", f"test_results_{timestamp}.csv")

    with open(report_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Test Case",
            "Field Name",
            "Pass/Fail",
            "Accuracy Score",
            "Pipeline Speed (seconds)",
            "Expected Value",
            "Actual Value",
        ])

        for result in results:
            test_case_number = result['test_case_number']
            performance = result['performance']
            for field_name, data in result['accuracy_results'].items():
                writer.writerow([
                    test_case_number,
                    field_name,
                    data['pass_fail'],
                    f"{data['score']:.2f}",
                    f"{performance:.4f}",
                    data['expected_value'],
                    data['actual_value'],
                ])
    print(f"CSV report generated and saved to: {report_path}")


def main() -> None:
    load_dotenv()

    # Validate required environment variables
    required_vars = [
        "AZURE_API_ENDPOINT",
        "AZURE_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_KEY",
        "AZURE_OPENAI_DEPLOYMENT",
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing_vars)}")

    test_cases = find_test_cases("test_data/labels")
    results = []
    for idx, (image_paths, expected_json_path) in enumerate(test_cases, 1):
        result = run_test_case(idx, image_paths, expected_json_path)
        results.append(result)
    generate_csv_report(results)


if __name__ == "__main__":
    main()