import dspy
from dspy import Prediction
import dspy.adapters
import dspy.utils

from pipeline.inspection import FertilizerInspection

from phoenix.otel import register
from openinference.instrumentation.dspy import DSPyInstrumentor

tracer_provider = register(
  project_name="gpt-ferti", # Default is 'default'
  endpoint="http://0.0.0.0:4317", # gRPC endpoint given by Phoenix when starting the server (default is "http://localhost:4317")
)

DSPyInstrumentor().instrument(tracer_provider=tracer_provider)

SUPPORTED_MODELS = {
    "gpt-3.5-turbo": {
        "max_tokens": 12000,
        "api_version": "2024-02-01",
        "response_format": { "type": "json_object" },
    },
    "gpt-4o": {
        "max_tokens": None,
        "api_version": "2024-02-15-preview",
        "response_format": { "type": "json_object" },
    }
} 

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
    
    text : str = dspy.InputField(desc="The text of the fertilizer label extracted using OCR.")
    # json_schema : str = dspy.InputField(desc="The JSON schema to follow.")
    requirements : str = dspy.InputField(desc="The instructions and guidelines to follow.")
    inspection : FertilizerInspection = dspy.OutputField(desc="The inspection results.")

class GPT:
    def __init__(self, api_endpoint, api_key, deployment_id):
        if not api_endpoint or not api_key or not deployment_id:
            raise ValueError("The API endpoint, key and deployment_id are required to instantiate the GPT class.")

        config = SUPPORTED_MODELS.get(deployment_id)
        if not config:
            raise ValueError(f"The deployment_id {deployment_id} is not supported.")
        
        self.dspy_client = dspy.AzureOpenAI(
            user="fertiscan",
            api_base=api_endpoint,
            api_key=api_key,
            deployment_id=deployment_id,
            **config
            # api_version=config.get("api_version"),
            # max_tokens=config.get("max_token"),
            # response_format=config.get("response_format"),
        )

    def create_inspection(self, text) -> Prediction:
        with dspy.context(lm=self.dspy_client, experimental=True):
            predictor = dspy.TypedChainOfThought(ProduceLabelForm)
            json_schema = FertilizerInspection.model_json_schema()
            prediction = predictor(text=text, json_schema=json_schema, requirements=REQUIREMENTS)

        return prediction
