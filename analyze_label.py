import os
from dotenv import load_dotenv
from pipeline import analyze, LabelStorage, GPT, OCR
import glob
import json

load_dotenv()

def analyze_label(file_path: list[str]):
    label = LabelStorage()
    for image_path in file_path:
        label.add_image(image_path)

    ocr = OCR(api_endpoint=os.getenv('AZURE_API_ENDPOINT'), api_key=os.getenv('AZURE_API_KEY'))
    gpt = GPT(api_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'), api_key=os.getenv('AZURE_OPENAI_KEY'), deployment_id=os.getenv('AZURE_OPENAI_DEPLOYMENT'))

    return analyze(label, ocr, gpt)

label_files = glob.glob(os.path.expanduser('./labels/*'))

print(label_files)

for folder in label_files:
    label_files = glob.glob(os.path.expanduser(f'{folder}/*'))
    # Remove from the list files that are not .png or .jpg
    label_files = [f for f in label_files if f.endswith('.png') or f.endswith('.jpg')]
    if len(label_files) == 0:
        print(f"No images found in {folder}")
        continue
    inspection = analyze_label(label_files)
    print(f"Results for {folder}:")
    # print(inspection.model_dump_json(indent=2))
    result_file = os.path.join(folder, 'inspection.json')
    with open(result_file, 'w') as f:
        json.dump(inspection.model_dump(), f, indent=2)
    print(f"Results written to {result_file}")