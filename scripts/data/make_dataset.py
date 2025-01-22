import os
import json
import csv

def create_data_list(base_dir):
    rows = []
    for instance_dir in os.listdir(base_dir):
        instance_path = os.path.join(base_dir, instance_dir)
        if os.path.isdir(instance_path):
            image_files = [os.path.join(instance_path, file) for file in os.listdir(instance_path) if file.endswith(('.png', '.jpg', '.jpeg'))]
            json_path = os.path.join(instance_path, "expected_output.json")
            if os.path.exists(json_path):
                with open(json_path, "r") as json_file:
                    expected_output = json.load(json_file)
            else:
                expected_output = None
            rows.append([image_files, json.dumps(expected_output, ensure_ascii=False)])
    return rows

def create_output_csv(output_csv, base_dir):
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_paths", "inspection"])
        rows = create_data_list(base_dir)
        for row in rows:
            writer.writerow(row)
    print(f"CSV file '{output_csv}' created successfully.")

if __name__ == "__main__":
    base_dir = os.path.join(os.getcwd(), "test_data", "labels")
    output_csv = os.path.join(os.getcwd(), "data", "processed", "dataset.csv")
    create_output_csv(output_csv, base_dir)