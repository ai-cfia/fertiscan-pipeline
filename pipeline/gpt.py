import dspy
from dspy import Prediction
from openai.types.chat.completion_create_params import ResponseFormat

MODELS_WITH_RESPONSE_FORMAT = [
    "ailab-llm",
    "ailab-llm-gpt-4o"
]  # List of models that support the response_format option

SPECIFICATION = """
Keys:
"company_name"
"company_address"
"company_website"
"company_phone_number"
"manufacturer_name"
"manufacturer_address"
"manufacturer_website"
"manufacturer_phone_number"
"fertiliser_name"
"registration_number" (a series of letters and numbers)
"lot_number"
"weight" (array of objects with "value", and "unit")
"density" (an object with "value", and "unit")
"volume" (an object with "value", and "unit")
"npk" (format: "number-number-number") **important
"guaranteed_analysis" (array of objects with "nutrient", "value", and "unit") **important
"warranty"
"cautions_en"  (array of strings)
"instructions_en" (array of strings)
"micronutrients_en" (array of objects with "nutrient", "value", and "unit")
"ingredients_en" (array of objects with "nutrient", "value", and "unit")
"specifications_en" (array of objects with "humidity", "ph", and "solubility")
"first_aid_en"  (array of strings)
"cautions_fr" (array of strings)
"instructions_fr" (array of strings)
"micronutrients_fr" (array of objects with "nutrient", "value", and "unit")
"ingredients_fr" (array of objects with "nutrient", "value", and "unit")
"specifications_fr" (array of objects with "humidity", "ph", and "solubility")
"first_aid_fr" (array of strings)

Requirements:
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
    Your response should be accurate, intelligible, formatted in JSON, and contain all the text from the provided text.
    """
    
    text = dspy.InputField(desc="The text of the fertilizer label extracted using OCR.")
    specification = dspy.InputField(desc="The specification containing the fields to highlight and their requirements.")
    form = dspy.OutputField(desc="Only a complete JSON.")

class GPT:
    def __init__(self, api_endpoint, api_key, deployment):
        if not api_endpoint or not api_key:
            raise ValueError("API endpoint and key are required to instantiate the GPT class.")

        response_format = None
        if deployment in MODELS_WITH_RESPONSE_FORMAT:
            response_format = ResponseFormat(type='json_object')

        max_token = 12000
        api_version = "2024-02-01"
        if deployment == MODELS_WITH_RESPONSE_FORMAT[0]:
            max_token = 3500
        elif deployment == MODELS_WITH_RESPONSE_FORMAT[1]:
            max_token = 4096
            api_version="2024-02-15-preview"

        self.dspy_client = dspy.AzureOpenAI(
            api_base=api_endpoint,
            api_key=api_key,
            deployment_id=deployment,
            # model_type='text',
            api_version=api_version,
            max_tokens=max_token,
            response_format=response_format,
        )

    def generate_form(self, prompt) -> Prediction:
        with dspy.context(lm=self.dspy_client, experimental=True):
            signature = dspy.ChainOfThought(ProduceLabelForm)
            prediction = signature(specification=SPECIFICATION, text=prompt)

        return prediction
