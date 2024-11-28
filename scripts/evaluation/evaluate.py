import re
from dotenv import load_dotenv
import dspy
import os
import ast
import json
import jellyfish
from urllib.parse import urlparse
import phonenumbers

import pandas as pd

from pipeline_new.modules.LanguageProgram import LanguageProgram


def preprocess_string(input_string):
    """
    Preprocesses a string by:
    1. Converting to lowercase.
    2. Removing extra whitespace, punctuation, and special characters.

    Args:
        input_string (str): The input string to preprocess.

    Returns:
        str: The preprocessed string.
    """
    if input_string is None:
        return None
    lowercased = input_string.lower()
    cleaned = re.sub(r'[^a-z0-9\s]', '', lowercased)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def normalize_phone_number(phone, country='CA'):
    try:
        parsed = phonenumbers.parse(phone, country)
        return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    except phonenumbers.NumberParseException:
        return phone


def normalize_website(url):
    if url is not None:
        try:
            url = url.lower().strip()
            if not urlparse(url).scheme:
                url = 'http://' + url
            parsed = urlparse(url)
            netloc = parsed.netloc.lstrip('www.')
            return netloc.rstrip('/')
        except Exception as e:
            print(f"Error normalizing URL '{url}': {e}")
            return url
    else:
        return url


def validate_inspection(example: dspy.Example, pred: dspy.Prediction, trace=None):
    scores = []

    pred_inspection_dict = pred.inspection.model_dump()
    pred_keys = list(pred_inspection_dict.keys())

    for example_key, example_value in example.inspection.items():
        if example_key in pred_keys:
            pred_value = pred_inspection_dict.get(example_key)

            if example_key in ["company_name", "company_address", "manufacturer_name", "manufacturer_address", "fertiliser_name"]:
                if example_value is not None and pred_value is not None:
                    score = jellyfish.jaro_winkler_similarity(
                        preprocess_string(example_value), preprocess_string(pred_value))
                    scores.append(score)
                else:
                    score = 1.0 if example_value == pred_value else 0.0
                    scores.append(score)
                # if score == 0:
                #     print(f"for {example_key}\tgot : {pred_value}\n\texpected: {example_value}")

            elif example_key in ["company_phone_number", "manufacturer_phone_number"]:
                score = 1.0 if normalize_phone_number(
                    example_value) == normalize_phone_number(pred_value) else 0.0
                scores.append(score)
                # if score == 0:
                #     print(f"for {example_key}\tgot : {pred_value}\n\texpected: {example_value}")

            elif example_key in ["company_website", "manufacturer_wekbsite"]:
                score = 1.0 if normalize_website(
                    example_value) == normalize_website(pred_value) else 0.0
                scores.append(score)
                # if score == 0:
                #     print(f"for {example_key}\tgot : {pred_value}\n\texpected: {example_value}")

            elif example_key in ["registration_number", "lot_number", "npk"]:
                score = 1.0 if example_value == pred_value else 0.0
                scores.append(score)
                # if score == 0:
                #     print(f"for {example_key}\tgot : {pred_value}\n\texpected: {example_value}")

    # print(scores)

    return sum(scores) / len(scores) if scores else 0.0


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
        raise RuntimeError(f"Missing required environment variables: {
                           ', '.join(missing_vars)}")

    AZURE_API_ENDPOINT = os.getenv('AZURE_API_ENDPOINT')
    AZURE_API_KEY = os.getenv('AZURE_API_KEY')
    AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
    AZURE_OPENAI_KEY = os.getenv('AZURE_OPENAI_KEY')
    AZURE_OPENAI_DEPLOYMENT = os.getenv('AZURE_OPENAI_DEPLOYMENT')

    return LanguageProgram(AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT, AZURE_API_KEY, AZURE_API_ENDPOINT)


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
        dataset.append(dspy.Example(image_paths=image_paths,
                       inspection=inspection).with_inputs("image_paths"))

    evaluate = dspy.Evaluate(
        devset=dataset[:5], metric=validate_inspection, display_progress=True, display_table=True)

    language_program = init_language_program()
    evaluate(language_program)
