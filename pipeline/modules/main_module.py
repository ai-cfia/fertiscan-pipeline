import os
import dspy
from components.label import LabelStorage
from components.ocr import OCR
from schemas.inspection import FertilizerInspection
from schemas.settings import Settings
from PIL import Image

# Constants
SUPPORTED_MODELS = {
    "gpt-3.5-turbo": {
        "max_tokens": 12000,
        "api_version": "2024-02-01",
        "response_format": {"type": "json_object"},
    },
    "gpt-4o": {
        "max_tokens": None,
        "api_version": "2024-02-15-preview",
        "response_format": {"type": "json_object"},
    },
}

REQUIREMENTS = """
The content of keys with the suffix _en must be in English.
The content of keys with the suffix _fr must be in French.
Translation of the text is prohibited.
You are prohibited from generating any text that is not part of the JSON.
The JSON must contain exclusively keys specified in "keys".
"""


# Signatures
class Inspector(dspy.Signature):
    """
    You are a fertilizer label inspector working for the Canadian Food Inspection Agency.
    Your task is to classify all information present in the provided text using the specified keys.
    Your response should be accurate, intelligible, information in JSON, and contain all the text from the provided text.
    """

    text: str = dspy.InputField(
        desc="The text of the fertilizer label extracted using OCR."
    )

    # TODO remove the depency on the pseudo prompt engineering
    requirements: str = dspy.InputField(
        desc="The instructions and guidelines to follow."
    )

    inspection: FertilizerInspection = dspy.OutputField(desc="The inspection results.")


# Modules
class MainModule(dspy.Module):
    def __init__(self, settings: Settings, useCache=True):
        # initialize all the components to be used in the forward method
        lm = dspy.LM(
            model=f"azure/{settings.llm_api_deployment}",
            api_base=settings.llm_api_endpoint,
            api_key=settings.llm_api_key.get_secret_value(),
            max_tokens=SUPPORTED_MODELS[settings.llm_api_deployment]["max_tokens"],
            api_version=SUPPORTED_MODELS[settings.llm_api_deployment]["api_version"],
            cache=useCache
        )
        dspy.configure(lm=lm)
        self.ocr = OCR(settings.document_api_endpoint, settings.document_api_key.get_secret_value())
        self.label_storage = LabelStorage()
        self.inspector = dspy.ChainOfThought(Inspector)

    def forward(self, images) -> dspy.Prediction:
        for image in images:
            self.label_storage.images.append(image)

        document = self.label_storage.get_document()
        ocr_results = self.ocr.extract_text(document=document)

        inspection_prediction = self.inspector(text=ocr_results.content, requirements=REQUIREMENTS)

        self.label_storage.clear()

        return inspection_prediction


if __name__ == "__main__":
    # load_dotenv()
    settings = Settings()

    test_image = Image.open(os.path.join(os.getcwd(), "test_data", "labels", "label_001", "img_001.png"))

    language_program = MainModule(settings=settings)

    prediction = language_program.forward(images=[test_image]).inspection

    print(dspy.inspect_history())
