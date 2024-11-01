import os
import time
import json
import shutil
import datetime
import csv
import tempfile
import pandas as pd
from dotenv import load_dotenv
from pipeline import analyze, LabelStorage, OCR, GPT
from tests import levenshtein_similarity

ACCURACY_THRESHOLD = 80.0
CSV_FOLDER = "reports"

def extract_leaf_fields(data, parent_key=''):
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
            new_key = f"{parent_key}[{index}]" if parent_key else f"[{index}]"
            if isinstance(item, (dict, list)):
                leaves.update(extract_leaf_fields(item, new_key))
            else:
                leaves[new_key] = item
    return leaves

def find_test_cases(labels_folder):
    test_cases = []
    label_directories = sorted(
        os.path.join(labels_folder, directory)
        for directory in os.listdir(labels_folder)
        if os.path.isdir(os.path.join(labels_folder, directory)) and directory.startswith("label_")
    )
    if len(label_directories) == 0:
        print(f"No label directories found in {labels_folder}")
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

def calculate_accuracy(expected_fields, actual_fields):
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

def run_test_case(test_case_number, image_paths, expected_json_path):
    copied_image_paths = []
    for image_path in image_paths:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image_path)[1])
        shutil.copy2(image_path, temp_file.name)
        copied_image_paths.append(temp_file.name)

    storage = LabelStorage()
    for image_path in copied_image_paths:
        storage.add_image(image_path)

    ocr = OCR(os.getenv("AZURE_API_ENDPOINT"), os.getenv("AZURE_API_KEY"))
    gpt = GPT(
        os.getenv("AZURE_OPENAI_ENDPOINT"),
        os.getenv("AZURE_OPENAI_KEY"),
        os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    )

    print("\tRunning analysis for test case...")
    start_time = time.time()
    actual_output = analyze(storage, ocr, gpt)
    performance = time.time() - start_time
    print(f"\tAnalysis completed in {performance:.2f} seconds.")

    actual_fields = extract_leaf_fields(json.loads(actual_output.model_dump_json()))

    with open(expected_json_path, 'r') as file:
        expected_fields = extract_leaf_fields(json.load(file))

    print("\tCalculating accuracy of results...")
    accuracy_results = calculate_accuracy(expected_fields, actual_fields)

    return {
        'test_case_number': test_case_number,
        'performance': performance,
        'accuracy_results': accuracy_results,
    }

def generate_csv_report(results):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
    os.makedirs(CSV_FOLDER, exist_ok=True)
    report_path = os.path.join(CSV_FOLDER, f"test_results_{timestamp}.csv")

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
    return report_path

def load_csv(csv_file):
    csv_file_path = os.path.join(CSV_FOLDER, csv_file)
    try:
        df = pd.read_csv(csv_file_path)
        return df
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
        exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file is empty at {csv_file_path}")
        exit(1)
    except pd.errors.ParserError:
        print(f"Error: Unable to parse CSV file at {csv_file_path}")
        exit(1)

def calculate_field_stats(df, original_field_order):
    field_stats = df.groupby('Field Name').agg(
        Accuracy_Mean=('Accuracy Score', 'mean'),
        Pass_Rate=('Pass', 'mean'),
        Missing_Rate=('Missing', 'mean'),
        Occurrences=('Field Name', 'count')
    ).reset_index()
    
    field_order_mapping = {field: index for index, field in enumerate(original_field_order)}
    field_stats['Field Order'] = field_stats['Field Name'].map(field_order_mapping)
    
    field_stats = field_stats.sort_values('Field Order').drop('Field Order', axis=1)
    
    return field_stats

def calculate_test_case_stats(df):
    test_case_stats = df.groupby('Test Case').agg(
        Overall_Pass_Rate=('Pass', 'mean'),
        Average_Accuracy_Score=('Accuracy Score', 'mean')
    ).reset_index()
    
    return test_case_stats

def calculate_overall_metrics(df):
    total_pass_rate = df['Pass'].mean()
    if 'Pipeline Speed (seconds)' in df.columns:
        average_pipeline_speed = df['Pipeline Speed (seconds)'].mean()
    else:
        average_pipeline_speed = None
    return total_pass_rate, average_pipeline_speed

def generate_markdown_report(field_stats, test_case_stats, total_pass_rate, average_pipeline_speed, total_test_cases):
    markdown_report = "# Field Analysis Report\n\n"
    
    markdown_report += "## Field-Level Aggregation\n\n"
    markdown_report += "The following table provides an overview of the accuracy, pass rate, and missing rate for each field across all test cases.\n\n"
    markdown_report += "`Field name`: correspond to a unique field in the Inspection.\n\n"
    markdown_report += "`Average Accuracy Score`: the average accuracy score for the given field across all test cases.\n\n"
    markdown_report += "`Pass Rate`: the percentage of test cases where the given field passed the accuracy threshold(default is 80%).\n\n"
    markdown_report += "`Missing Rate`: the percentage of test cases where the given field was not found in the Inspection returned by the pipeline.\n\n"
    markdown_report += "`Occurrences`: the number of times the given field was found across all test cases.\n\n"
    markdown_report += "| Field Name | Average Accuracy Score | Pass Rate | Missing Rate | Occurrences |\n"
    markdown_report += "|------------|------------------------|-----------|--------------|-------------|\n"
    
    for _, row in field_stats.iterrows():
        field_name = row['Field Name']
        accuracy_mean = f"{row['Accuracy_Mean']:.2f}"
        pass_rate = f"{row['Pass_Rate']:.2f}"
        missing_rate = f"{row['Missing_Rate']:.2f}"
        occurrences = f"{int(row['Occurrences'])}/{total_test_cases}"
        
        markdown_report += f"| {field_name} | {accuracy_mean} | {pass_rate} | {missing_rate} | {occurrences} |\n"
    
    markdown_report += "\n## Test Case-Level Aggregation\n\n"
    markdown_report += "The following table provides an overview of the overall pass rate and average accuracy score for each test case.\n\n"
    markdown_report += "`Overall Pass Rate`: the percentage of fields that passed the accuracy threshold(default is 80%) in the given test case.\n\n"
    markdown_report += "`Average Accuracy Score`: the average accuracy score for all fields in the given test case.\n\n"
    markdown_report += "| Test Case | Overall Pass Rate | Average Accuracy Score |\n"
    markdown_report += "|-----------|-------------------|------------------------|\n"
    
    for _, row in test_case_stats.iterrows():
        test_case = row['Test Case']
        overall_pass_rate = f"{row['Overall_Pass_Rate']:.2f}"
        avg_accuracy = f"{row['Average_Accuracy_Score']:.2f}"
        markdown_report += f"| {test_case} | {overall_pass_rate} | {avg_accuracy} |\n"
    
    markdown_report += "\n## Overall Pipeline Metrics\n\n"
    markdown_report += f"- Total Pass Rate: {total_pass_rate:.2f}\n"
    if average_pipeline_speed is not None:
        markdown_report += f"- Average Pipeline Speed: {average_pipeline_speed:.2f} seconds\n"
    else:
        markdown_report += "- Average Pipeline Speed: Not available\n"
    
    markdown_report += f"\n- Total Number of Test Cases: {total_test_cases}\n"
    
    return markdown_report

def save_report(markdown_content, report_file):
    with open(report_file, "w") as f:
        f.write(markdown_content)

def main():
    print("Script execution started.")

    load_dotenv()
    
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
    print(f"Found {len(test_cases)} test case(s) to process.")

    results = []
    for idx, (image_paths, expected_json_path) in enumerate(test_cases, 1):
        print(f"Processing test case {idx}...")
        try:
            result = run_test_case(idx, image_paths, expected_json_path)
            results.append(result)
        except Exception as e:
            print(f"Error processing test case {idx}: {e}")
            continue
    
    csv_report_path = generate_csv_report(results)
    
    df = load_csv(os.path.basename(csv_report_path))
    
    df['Pass'] = df['Pass/Fail'] == 'Pass'
    df['Missing'] = df['Actual Value'] == 'FIELD_NOT_FOUND'
    
    total_test_cases = df['Test Case'].nunique()
    original_field_order = df['Field Name'].unique()
    
    field_stats = calculate_field_stats(df, original_field_order)
    test_case_stats = calculate_test_case_stats(df)
    total_pass_rate, average_pipeline_speed = calculate_overall_metrics(df)
    
    markdown_report = generate_markdown_report(
        field_stats,
        test_case_stats,
        total_pass_rate,
        average_pipeline_speed,
        total_test_cases
    )

    report_name = os.path.splitext(os.path.basename(csv_report_path))[0] + ".md"
    report_path = os.path.join(CSV_FOLDER, report_name)

    save_report(markdown_report, report_path)
    
    print(f"Markdown report generated and saved to: {report_path}")
    print("Script execution completed.")

if __name__ == "__main__":
    main()
