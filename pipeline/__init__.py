from .modules.main_module import MainModule
from .schemas.settings import Settings
from .schemas.inspection import FertilizerInspection  # noqa: F401
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


def analyze(images: list[Image.Image], settings: Settings) -> FertilizerInspection:
    """
    Analyze a fertiliser label using our pipeline module(s).
    It returns the data extracted from the label in a FertiliserForm.
    """
    predictor = MainModule(settings)

    # Analyse all the images loaded
    prediction = predictor.forward(images)

    return prediction.inspection
