import os
import json
import csv

# Set the directory path
base_dir = "test_data/labels"
output_csv = "test_data.csv"

# Initialize a list to hold the rows for the CSV file
rows = []

# Iterate over each instance directory in `all_labels`
for instance_dir in os.listdir(base_dir):
    instance_path = os.path.join(base_dir, instance_dir)
    
    # Ensure we're looking at a directory
    if os.path.isdir(instance_path):
        # Get the list of image files in the directory
        image_files = [file for file in os.listdir(instance_path) if file.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Read the JSON object from expected_output.json
        json_path = os.path.join(instance_path, "expected_output.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as json_file:
                expected_output = json.load(json_file)
        else:
            expected_output = None  # If no expected_output.json file is found, set to None or handle as needed

        # Append the row to the list with the image files list and JSON object
        rows.append([image_files, expected_output])

# Write to the CSV file
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Write the header
    writer.writerow(["image_files", "expected_output"])
    # Write each row
    for row in rows:
        writer.writerow(row)

print(f"CSV file '{output_csv}' created successfully.")
