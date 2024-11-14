import dspy

from pipeline_new.schemas.inspection import FertilizerInspection


REQUIREMENTS = """
The content of keys with the suffix _en must be in English.
The content of keys with the suffix _fr must be in French.
Translation of the text is prohibited.
You are prohibited from generating any text that is not part of the JSON.
The JSON must contain exclusively keys specified in "keys".
"""

class Inspector(dspy.Signature):
    """
    You are a fertilizer label inspector working for the Canadian Food Inspection Agency.
    Your task is to classify all information present in the provided text using the specified keys.
    Your response should be accurate, intelligible, information in JSON, and contain all the text from the provided text.
    """

    text: str = dspy.InputField(
        desc="The text of the fertilizer label extracted using OCR."
    )
    requirements: str = dspy.InputField(
        desc="The instructions and guidelines to follow."
    )
    inspection: FertilizerInspection = dspy.OutputField(desc="The inspection results.")