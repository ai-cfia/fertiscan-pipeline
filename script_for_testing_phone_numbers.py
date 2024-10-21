import os
import pickle

from dotenv import load_dotenv

from pipeline import GPT, OCR, LabelStorage, analyze
from pipeline.inspection import FertilizerInspection

# Load environment variables
load_dotenv()

# Define the label folder numbers
label_folders = [8, 11, 19, 22, 24, 25, 27, 28, 30, 34]
# label_folders = [24, 25]

# Define possible image filenames and extensions
image_filenames = ["img_001", "img_002"]  # Basenames without extension
image_extensions = [".jpg", ".png"]  # Possible extensions

# Mock environment setup for OCR and GPT
api_endpoint_ocr = os.getenv("AZURE_API_ENDPOINT")
api_key_ocr = os.getenv("AZURE_API_KEY")
api_endpoint_gpt = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key_gpt = os.getenv("AZURE_OPENAI_KEY")
api_deployment_gpt = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Initialize OCR and GPT objects (reusable)
ocr = OCR(api_endpoint=api_endpoint_ocr, api_key=api_key_ocr)
gpt = GPT(api_endpoint=api_endpoint_gpt, api_key=api_key_gpt, deployment_id=api_deployment_gpt)

# Dictionary to store inspection results for all labels
all_inspections = {}

# Loop through each label folder
for label_num in label_folders:
    label_folder = f"test_data/labels/label_{label_num:03d}"  # Format as label_008, label_011, etc.
    label_storage = LabelStorage()  # Initialize a new LabelStorage for each label folder

    # Add relevant images to the label storage
    for image_filename in image_filenames:
        for ext in image_extensions:
            image_path = os.path.join(label_folder, f"{image_filename}{ext}")
            if os.path.exists(image_path):
                print("Adding image:", image_path)
                label_storage.add_image(image_path)

    # Run the analyze function
    inspection = analyze(label_storage, ocr, gpt)

    # Store the result in the dictionary with the label number as the key
    all_inspections[f"label_{label_num:03d}"] = inspection

# Pickle all the results in a single file
pickle.dump(all_inspections, open("all_inspections.pkl", "wb"))

print("All inspections have been processed and saved to all_inspections.pkl")


# Load the pickled data
with open("all_inspections.pkl", "rb") as f:
    all_inspections: dict[str, FertilizerInspection] = pickle.load(f)

for label, inspection in all_inspections.items():
    print(f"Label: {label}")
    print(f"  Company Phone Number: {inspection.company_phone_number}")
    print(f"  Manufacturer Phone Number: {inspection.manufacturer_phone_number}")
    print()
