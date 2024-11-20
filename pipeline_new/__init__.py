from pipeline_new.modules.LanguageProgram import LanguageProgram
from pipeline_new.components.label import LabelStorage  # noqa: F401
from pipeline_new.components.ocr import OCR  # noqa: F401
from pipeline_new.schemas.inspection import FertilizerInspection  # noqa: F401

import os

def save_text_to_file(text: str, output_path: str):  # pragma: no cover
    """
    Save text to a file. 
    """
    with open(output_path, 'w') as output_file:
        output_file.write(text)


def save_image_to_file(image_bytes: bytes, output_path: str):  # pragma: no cover
    """
    Save the raw byte data of an image to a file. 
    """
    with open(output_path, 'wb') as output_file:
        output_file.write(image_bytes)


def analyze(images_path: list[str], llm_api_key, llm_api_endpoint, llm_api_deployment_id, ocr_api_key, ocr_api_endpoint, log_dir_path: str = './logs') -> FertilizerInspection:
    """
    Analyze a fertiliser label using an OCR and an LLM.
    It returns the data extracted from the label in a FertiliserForm.
    """
    if not os.path.exists(log_dir_path):
        print('create path')
        os.mkdir(path=log_dir_path)

    # Create the label storage
    label_storage = LabelStorage()

    for image_path in images_path:
        label_storage.add_image(image_path)

    # Create the language program
    language_program = LanguageProgram(
        llm_api_key, llm_api_endpoint, llm_api_deployment_id, ocr_api_key, ocr_api_endpoint)

    # Analyse all the images loaded
    predition = language_program.forward(images_path)

    # Clear the label storage to delete any local images
    label_storage.clear()

    return predition.inspection
