import os
from dotenv import load_dotenv


from scripts.evaluation.run_performance_assessment_data_collection import find_test_cases, run_test_case, generate_csv_report
from scripts.visualization.run_performance_assessment_data_visualization import load_csv, calculate_field_stats, calculate_test_case_stats, calculate_overall_metrics, generate_markdown_report, save_report

CSV_FOLDER = "reports"

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
    
    # csv_report_path = os.path.join(CSV_FOLDER, os.path.basename(generate_csv_report(results)))
    csv_report_path = generate_csv_report(results)
    print(csv_report_path)

    df = load_csv(CSV_FOLDER, os.path.basename(csv_report_path))
    
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
