import os
import json
import csv
import base64

# Set the directory path
base_dir = "test_data/labels"
output_csv = "byte_obj_test_data.csv"

# Initialize a list to hold the rows for the CSV file
rows = []

# Iterate over each instance directory in `all_labels`
for instance_dir in os.listdir(base_dir):
    instance_path = os.path.join(base_dir, instance_dir)
    
    # Ensure we're looking at a directory
    if os.path.isdir(instance_path):
        # Collect base64-encoded byte objects for each image in the directory
        image_bytes = []
        for file in os.listdir(instance_path):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(instance_path, file)
                with open(file_path, "rb") as image_file:
                    # Read and encode the image to base64
                    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
                    image_bytes.append(encoded_image)

        # Read the JSON object from expected_output.json
        json_path = os.path.join(instance_path, "expected_output.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as json_file:
                expected_output = json.load(json_file)
        else:
            expected_output = None  # Handle missing JSON file

        # Append the row to the list with the base64-encoded images and JSON object
        rows.append([image_bytes, expected_output])

# Write to the CSV file
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Write the header
    writer.writerow(["image_bytes", "expected_output"])
    # Write each row
    for row in rows:
        writer.writerow(row)

print(f"CSV file '{output_csv}' created successfully.")
