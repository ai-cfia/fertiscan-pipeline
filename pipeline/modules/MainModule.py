import os
from dotenv import load_dotenv
import dspy
from pipeline.components.label import LabelStorage
from pipeline.components.ocr import OCR
from pipeline.schemas.inspection import FertilizerInspection
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
    def __init__(self, llm_api_key, llm_api_endpoint, llm_deployment_id, orc_api_key, ocr_api_endpoint, useCache=True):
        # initialize all the components to be used in the forward method
        lm = dspy.LM(
            model=f"azure/{llm_deployment_id}",
            api_base=llm_api_endpoint,
            api_key=llm_api_key,
            max_tokens=SUPPORTED_MODELS.get(llm_deployment_id)["max_tokens"],
            api_version=SUPPORTED_MODELS.get(llm_deployment_id)["api_version"],
            cache=useCache
        )
        dspy.configure(lm=lm)
        self.ocr = OCR(ocr_api_endpoint, orc_api_key)
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
    load_dotenv()    

    required_vars = [
        "AZURE_API_ENDPOINT",
        "AZURE_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_KEY",
        "AZURE_OPENAI_DEPLOYMENT",
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing_vars)}")

    AZURE_API_ENDPOINT = os.getenv('AZURE_API_ENDPOINT')
    AZURE_API_KEY = os.getenv('AZURE_API_KEY')
    AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
    AZURE_OPENAI_KEY = os.getenv('AZURE_OPENAI_KEY')
    AZURE_OPENAI_DEPLOYMENT = os.getenv('AZURE_OPENAI_DEPLOYMENT')

    test_image = Image.open(os.path.join(os.getcwd(), "test_data", "labels", "label_001", "img_001.png"))

    language_program = MainModule(AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT,AZURE_API_KEY,AZURE_API_ENDPOINT)

    prediction = language_program.forward(images=[test_image]).inspection

    print(dspy.inspect_history())