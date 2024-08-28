import dspy
from dspy import Prediction
import dspy.adapters
import dspy.utils

from pipeline.inspection import FertilizerInspection

MODELS_WITH_RESPONSE_FORMAT = [
    "ailab-llm",
    "ailab-llm-gpt-4o"
]  # List of models that support the response_format option

REQUIREMENTS = """
The content of keys with the suffix _en must be in English.
The content of keys with the suffix _fr must be in French.
Translation of the text is prohibited.
You are prohibited from generating any text that is not part of the JSON.
The JSON must contain exclusively keys specified in "keys".
"""

class ProduceLabelForm(dspy.Signature):
    """
    You are a fertilizer label inspector working for the Canadian Food Inspection Agency. 
    Your task is to classify all information present in the provided text using the specified keys.
    Your response should be accurate, intelligible, information in JSON, and contain all the text from the provided text.
    """
    
    text = dspy.InputField(desc="The text of the fertilizer label extracted using OCR.")
    json_schema = dspy.InputField(desc="The JSON schema of the object to be returned.")
    requirements = dspy.InputField(desc="The instructions and guidelines to follow.")
    inspection = dspy.OutputField(desc="Only a complete JSON.")

class GPT:
    def __init__(self, api_endpoint, api_key, deployment_id):
        if not api_endpoint or not api_key or not deployment_id:
            raise ValueError("The API endpoint, key and deployment_id are required to instantiate the GPT class.")

        response_format = None
        if deployment_id in MODELS_WITH_RESPONSE_FORMAT:
            response_format = { "type": "json_object" }

        max_token = 12000
        api_version = "2024-02-01"
        if deployment_id == MODELS_WITH_RESPONSE_FORMAT[0]:
            max_token = 3500
        elif deployment_id == MODELS_WITH_RESPONSE_FORMAT[1]:
            max_token = 4096
            api_version="2024-02-15-preview"

        self.dspy_client = dspy.AzureOpenAI(
            user="fertiscan",
            api_base=api_endpoint,
            api_key=api_key,
            deployment_id=deployment_id,
            # model_type='text',
            api_version=api_version,
            max_tokens=max_token,
            response_format=response_format,
        )

    def create_inspection(self, text) -> Prediction:
        with dspy.context(lm=self.dspy_client, experimental=True):
            json_schema = FertilizerInspection.model_json_schema()
            signature = dspy.ChainOfThought(ProduceLabelForm)
            prediction = signature(text=text, json_schema=json_schema, requirements=REQUIREMENTS)

        return prediction
