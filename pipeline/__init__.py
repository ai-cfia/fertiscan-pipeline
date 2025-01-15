from pipeline.modules.MainModule import MainModule
from pipeline.schemas.inspection import FertilizerInspection  # noqa: F401
from PIL import Image

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


def analyze(images: list[Image.Image], llm_api_key, llm_api_endpoint, llm_api_deployment_id, ocr_api_key, ocr_api_endpoint) -> FertilizerInspection:
    """
    Analyze a fertiliser label using our pipeline module(s).
    It returns the data extracted from the label in a FertiliserForm.
    """
    predictor = MainModule(
        llm_api_key, llm_api_endpoint, llm_api_deployment_id, ocr_api_key, ocr_api_endpoint)

    # Analyse all the images loaded
    predition = predictor.forward(images)

    return predition.inspection