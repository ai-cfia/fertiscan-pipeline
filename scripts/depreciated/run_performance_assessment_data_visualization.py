import os
import pandas as pd

CSV_FILE = "test_results_202410221641.csv"
CSV_FOLDER = "reports"


def load_csv(csv_folder, csv_file):
    csv_file_path = os.path.join(csv_folder, csv_file)
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
    # Group by 'Field Name' and aggregate
    field_stats = df.groupby('Field Name').agg(
        Accuracy_Mean=('Accuracy Score', 'mean'),
        Pass_Rate=('Pass', 'mean'),
        Missing_Rate=('Missing', 'mean'),
        Occurrences=('Field Name', 'count')
    ).reset_index()
    
    # Map original field order (the order we are used to see these fields in the UI)
    field_order_mapping = {field: index for index, field in enumerate(original_field_order)}
    field_stats['Field Order'] = field_stats['Field Name'].map(field_order_mapping)
    
    # Sort by original field order
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
    
    # Field-Level Aggregation
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
    
    # Test Case-Level Aggregation
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
    
    # Overall Pipeline Metrics
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
    # Load the CSV report
    df = load_csv(CSV_FOLDER, CSV_FILE)
    
    # Preprocess the data
    # Create 'Pass' boolean column
    df['Pass'] = df['Pass/Fail'] == 'Pass'
    
    # Create 'Missing' boolean column
    df['Missing'] = df['Actual Value'] == 'FIELD_NOT_FOUND'
    
    # Get total number of test cases
    total_test_cases = df['Test Case'].nunique()
    
    # Get the original field order
    original_field_order = df['Field Name'].unique()
    
    # Calculate statistics
    field_stats = calculate_field_stats(df, original_field_order)
    test_case_stats = calculate_test_case_stats(df)
    total_pass_rate, average_pipeline_speed = calculate_overall_metrics(df)
    
    # Generate markdown report
    markdown_report = generate_markdown_report(
        field_stats,
        test_case_stats,
        total_pass_rate,
        average_pipeline_speed,
        total_test_cases
    )

    # name the report after the csv but with a markdown extension
    report_name = os.path.splitext(CSV_FILE)[0] + ".md"
    report_path = os.path.join(CSV_FOLDER, report_name)


    # Save the report
    save_report(markdown_report, report_path)
    
    print(f"Aggregation complete. Check '{report_path}' for the results.")

if __name__ == "__main__":
    main()
