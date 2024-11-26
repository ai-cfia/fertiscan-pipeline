from pprint import pprint
from dotenv import load_dotenv
import dspy
from dspy.datasets import DataLoader
import os
import ast  # Add this import
import json  # Add this import

import pandas as pd

from pipeline_new.modules.LanguageProgram import LanguageProgram


def validate_inspection(example: dspy.Example, pred: dspy.Prediction, trace=None):
    """
    placeholder validation that simply checks that the keys in the example are in the prediction
    """
    total_keys = len(example.inspection)
    print(f"Total keys in example inspection: {total_keys}")
    
    example_keys = list(example.inspection.keys())
    print(f"Example inspection keys: {example_keys}")
    
    pred_inspection_dict = pred.inspection.model_dump() 
    pred_keys = list(pred_inspection_dict.keys())
    print(f"Prediction inspection keys: {pred_keys}")
    
    matching_keys = sum(1 for key in example_keys if key in pred_keys)
    print(f"Matching keys: {matching_keys}")
    
    score = matching_keys / total_keys if total_keys > 0 else 0
    print(f"Score: {score}")
    
    return score


def init_language_program():
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

    return LanguageProgram(AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT,AZURE_API_KEY,AZURE_API_ENDPOINT)


if __name__ == "__main__":

    dataset_path = 'data/processed/dataset.csv'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    df = pd.read_csv(dataset_path)
    print(df.shape)

    dataset = []

    for image_paths, inspection in df.values:
        image_paths = ast.literal_eval(image_paths)
        inspection = json.loads(inspection)
        dataset.append(dspy.Example(image_paths=image_paths, inspection=inspection).with_inputs("image_paths"))

    evaluate = dspy.Evaluate(devset=dataset, metric=validate_inspection, display_progress=True, display_table=True)

    language_program = init_language_program()
    evaluate(language_program)