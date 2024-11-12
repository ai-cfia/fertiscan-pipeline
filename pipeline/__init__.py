from .label import LabelStorage  # noqa: F401
from .ocr import OCR  # noqa: F401
from .inspection import FertilizerInspection  # noqa: F401
from .gpt import GPT  # noqa: F401

import openai
from IPython.display import Image, display, Audio, Markdown
import base64

import os
from datetime import datetime

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

def analyze(label_storage: LabelStorage, ocr: OCR, gpt: GPT, log_dir_path: str = './logs') -> FertilizerInspection:
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
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_text_to_file(result.content, f"{log_dir_path}/{now}.md")

    # Generate inspection from extracted text
    prediction = gpt.create_inspection(result.content)

    # Check the coninspectionity of the JSON
    inspection = prediction.inspection

    # Logs the results from GPT
    save_text_to_file(prediction.reasoning, f"{log_dir_path}/{now}.txt")
    save_text_to_file(inspection.model_dump_json(indent=2), f"{log_dir_path}/{now}.json")

    # Clear the label cache
    label_storage.clear()

    # Delete the logs if there's no error
    os.remove(f"{log_dir_path}/{now}.md")   
    os.remove(f"{log_dir_path}/{now}.txt")     
    os.remove(f"{log_dir_path}/{now}.json")

    return inspection

def analyze_with_ocr_enhancment(label_storage: LabelStorage, ocr: OCR, gpt: GPT, log_dir_path: str = './logs') -> FertilizerInspection:
    """
    Analyze a fertiliser label using an OCR and an LLM.
    It returns the data extracted from the label in a FertiliserForm.
    """
    if not os.path.exists(log_dir_path):
        print('create path')
        os.mkdir(path=log_dir_path)

    document = label_storage.get_document()
    result = ocr.extract_text(document=document)
    
    # Open the image file and encode it as a base64 string
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
        
    base64_image = encode_image(label_storage.images[0])
    
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that responds in Markdown. Help me with reading the infromation in the image!"},
            {"role": "user", "content": [
                {"type": "text", "text": "In the image below, there is a fertilizer label. We extracted the following text from the image: " + result.content + " Using both the image and the extracted text, I want you to improve and enrich the text based on the image in the following ways: -fill out the missing information in the image. - improve the structure of the text to be more readable. - correct any errors in the extracted text."},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"}
                }
            ]}
        ],
        temperature=0.0,
    )

    ocr_enhanced = response.choices[0].message.content
    
    print("--------------------------------------------------------------")
    print(ocr_enhanced)
    print("--------------------------------------------------------------")

    # Logs the results from document intelligence
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_text_to_file(result.content, f"{log_dir_path}/{now}.md")

    # Generate inspection from extracted text
    prediction = gpt.create_inspection(ocr_enhanced)

    # Check the coninspectionity of the JSON
    inspection = prediction.inspection

    # Logs the results from GPT
    save_text_to_file(prediction.reasoning, f"{log_dir_path}/{now}.txt")
    save_text_to_file(inspection.model_dump_json(indent=2), f"{log_dir_path}/{now}.json")

    # Clear the label cache
    label_storage.clear()

    # Delete the logs if there's no error
    os.remove(f"{log_dir_path}/{now}.md")   
    os.remove(f"{log_dir_path}/{now}.txt")     
    os.remove(f"{log_dir_path}/{now}.json")

    return inspection
