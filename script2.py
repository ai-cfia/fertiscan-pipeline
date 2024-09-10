import os
from pprint import pprint
from tests import curl_file
from dotenv import load_dotenv
# from pipeline.inspection import FertilizerInspection
from pipeline import LabelStorage, OCR, GPT, analyze

# Load environment variables
load_dotenv()

# Set up the required objects
log_dir_path = 'test_logs'
image_path = 'granulaine.png'  # Path to your test image

# Ensure the log directory exists
if not os.path.exists(log_dir_path):
    os.mkdir(log_dir_path)

# Download the test image
# curl_file(url='https://tlhort.com/cdn/shop/products/10-52-0MAP.jpg', path=image_path)

# Mock environment setup for OCR and GPT
api_endpoint_ocr = os.getenv('AZURE_API_ENDPOINT')
api_key_ocr = os.getenv('AZURE_API_KEY')
api_endpoint_gpt = os.getenv('AZURE_OPENAI_ENDPOINT')
api_key_gpt = os.getenv('AZURE_OPENAI_KEY')
api_deployment_gpt = os.getenv('AZURE_OPENAI_DEPLOYMENT')

# Initialize the objects
label_storage = LabelStorage()
label_storage.add_image(image_path)
ocr = OCR(api_endpoint=api_endpoint_ocr, api_key=api_key_ocr)
gpt = GPT(api_endpoint=api_endpoint_gpt, api_key=api_key_gpt, deployment_id=api_deployment_gpt)

# Run the analyze function
form = analyze(label_storage, ocr, gpt, log_dir_path=log_dir_path)

# Pretty print the form
print(form.model_dump_json())
