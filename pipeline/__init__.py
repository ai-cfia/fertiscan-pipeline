from .label import LabelStorage  # noqa: F401
from .ocr import OCR  # noqa: F401
from .form import FertiliserForm  # noqa: F401
from .gpt import GPT  # noqa: F401

import os
import datetime
import requests
from dspy import Prediction


def curl_file(url:str, path: str):
    img_data = requests.get(url).content
    with open(path, 'wb') as handler:
        handler.write(img_data)  

def save_text_to_file(text: str, output_path: str):
    with open(output_path, 'w') as output_file:
        output_file.write(text)

def save_image_to_file(image_bytes: bytes, output_path: str):
    with open(output_path, 'wb') as output_file:
        output_file.write(image_bytes)

def analyze(files: list[str]) -> Prediction:
    ocr = OCR()
    gpt = GPT()
    label_storage = LabelStorage()

    for file_path in files:
        label_storage.add_image(file_path)

    document = label_storage.get_document()
    result = ocr.extract_text(document=document)

    # Logs the results from document intelligence
    now = datetime.now()
    save_text_to_file(result.content, f"./logs/{now}.md")

    # Generate form from extracted text
    prediction = gpt.generate_form(result.content)

    return prediction
