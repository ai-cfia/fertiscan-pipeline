from .label import LabelStorage  # noqa: F401
from .ocr import OCR  # noqa: F401
from .form import FertiliserForm  # noqa: F401
from .gpt import GPT  # noqa: F401

import os
import json
import requests
from datetime import datetime

def curl_file(url:str, path: str): # pragma: no cover
    """
    Pull a file from an URL and save its content.
    """
    data = requests.get(url).content
    with open(path, 'wb') as handler:
        handler.write(data)  

def save_text_to_file(text: str, output_path: str): # pragma: no cover
    """
    Save text to a file. 
    """
    with open(output_path, 'w') as output_file:
        output_file.write(text)

def save_image_to_file(image_bytes: bytes, output_path: str): # pragma: no cover
    """
    Save the raw byte data of an image to a file. 
    """
    with open(output_path, 'wb') as output_file:
        output_file.write(image_bytes)

def analyze(label_storage: LabelStorage, ocr: OCR, gpt: GPT, log_dir_path: str = './logs') -> FertiliserForm:
    """
    Analyze a fertiliser label using an OCR and an LLM.
    It returns the data extracted from the label in a FertiliserForm.
    """
    if not os.path.exists(log_dir_path):
        print('create path')
        os.mkdir(path=log_dir_path)

    document = label_storage.get_document()
    result = ocr.extract_text(document=document)

    # Logs the results from document intelligence
    now = datetime.now()
    save_text_to_file(result.content, f"{log_dir_path}/{now}.md")

    # Generate form from extracted text
    prediction = gpt.generate_form(result.content)

    # Logs the results from GPT
    save_text_to_file(prediction.form, f"{log_dir_path}/{now}.json")
    save_text_to_file(prediction.rationale, f"{log_dir_path}/{now}.txt")

    # Check the conformity of the JSON
    form = FertiliserForm(**json.loads(prediction.form))

    # Clear the label cache
    label_storage.clear()

    # Delete the logs if there's no error
    os.remove(f"{log_dir_path}/{now}.md")   
    os.remove(f"{log_dir_path}/{now}.txt")     
    os.remove(f"{log_dir_path}/{now}.json")

    return form
