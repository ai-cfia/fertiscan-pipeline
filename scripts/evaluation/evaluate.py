import re
from dotenv import load_dotenv
import dspy
import os
import ast
import json
import jellyfish
from urllib.parse import urlparse
import numpy as np
import phonenumbers
import pandas as pd
from openai import AzureOpenAI
from sklearn.metrics.pairwise import cosine_similarity

import sys 

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from pipeline_new.modules.LanguageProgram import LanguageProgram
from pipeline_new.schemas.inspection import FertilizerInspection, GuaranteedAnalysis, NutrientValue, Organization, RegistrationNumber, Value

# HELPER FUNCTION TO LOAD .ENV WITH CHECKS
def load_env_variables():
    load_dotenv()

    required_vars = [
        "AZURE_API_ENDPOINT",
        "AZURE_API_KEY",
        "AZURE_OPENAI_KEY",
        "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_EMBEDDING_ENDPOINT",
        "AZURE_OPENAI_EMBEDDING_KEY",
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing_vars)}")

    AZURE_API_ENDPOINT = os.getenv('AZURE_API_ENDPOINT')
    AZURE_API_KEY = os.getenv('AZURE_API_KEY')
    AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
    AZURE_OPENAI_KEY = os.getenv('AZURE_OPENAI_KEY')
    AZURE_OPENAI_DEPLOYMENT = os.getenv('AZURE_OPENAI_DEPLOYMENT')
    AZURE_OPENAI_EMBEDDING_ENDPOINT = os.getenv(
        'AZURE_OPENAI_EMBEDDING_ENDPOINT')
    AZURE_OPENAI_EMBEDDING_KEY = os.getenv('AZURE_OPENAI_EMBEDDING_KEY')

    return AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT, AZURE_API_KEY, AZURE_API_ENDPOINT, AZURE_OPENAI_EMBEDDING_KEY, AZURE_OPENAI_EMBEDDING_ENDPOINT


# GLOBAL VARIABLES
AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT, AZURE_API_KEY, AZURE_API_ENDPOINT, AZURE_OPENAI_EMBEDDING_KEY, AZURE_OPENAI_EMBEDDING_ENDPOINT = load_env_variables()


EMBEDDING_MODEL = AzureOpenAI(
    api_key=AZURE_OPENAI_EMBEDDING_KEY,
    api_version="2023-05-15",
    azure_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT
)

SCORES_BY_FIELD = {
    "organizations": [],
    "fertiliser_name": [],
    "registration_number": [],
    "lot_number": [],
    "npk": [],
    "weight": [],
    "density": [],
    "volume": [],
    "guaranteed_analysis_en": [],
    "guaranteed_analysis_fr": [],
    "ingredients_en": [],
    "ingredients_fr": [],
    "cautions_en": [],
    "cautions_fr": [],
    "instructions_en": [],
    "instructions_fr": [],
}

# UTILITY FUNCTIONS
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
    if not isinstance(input_string, str):
        input_string = str(input_string)
    lowercased = input_string.lower()
    cleaned = re.sub(r'[^\w\s]', '', lowercased, flags=re.UNICODE)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def display_scores_by_field(scores_by_field=SCORES_BY_FIELD):
    print("\n--Scores by field--")
    for field, score_list in scores_by_field.items():
        average = sum(score_list)/len(score_list)
        print(f"{field:<30} | {average}")

def if_array_is_empty(array):
    if array is None:
        return []
    for item in array:
        if isinstance(item, dict):
            values = item.values()
            if any(value is not None for value in values):
                return array
    return []

def normalize_ingredients_array(array):
    return sorted(
            [
                {
                    'ingredient': item.ingredient.lower() if item.ingredient is not None else None,
                    'value': item.value,
                    'unit': item.unit.lower().replace('s', '') if item.unit is not None else None
                }
                for item in array
            ],
            key=lambda x: x['ingredient']  # Sort by 'nutrient'
        )

def is_gma_none(gma):
    if gma is None:
        return True
    if ((gma.title is None) and ((gma.nutrients is None) or ((isinstance(gma.nutrients, list) and len(gma.nutrients) == 0))) and (gma.is_minimal is None)):
        return True
    return False

def normalize_gma_array(array):
    return sorted(
            [
                {
                    'nutrient': item.nutrient.lower() if item.nutrient is not None else None, 
                    'value': item.value, 
                    'unit': item.unit.lower().replace('s', '') if item.unit is not None else None
                }
                for item in array
            ],
            key=lambda x: x['nutrient']  # Sort by 'nutrient'
        )

def if_dictionary_values_are_none(dictionary):
    return all(value is None for value in dictionary.values())

def sort_and_preprocess_organization_array(array: list[Organization]):
    return sorted(
            [
                {
                    'name': preprocess_string(item.name) if item.name is not None else None,
                    'address': preprocess_string(item.address) if item.address is not None else None,
                    'website': normalize_website(item.website) if item.website is not None else None,
                    'phone_number': normalize_phone_number(item.phone_number) if item.phone_number is not None else None
                }
                for item in array
            ],
            key=lambda x: x['name']  # Sort by 'name'
        )

def sort_and_preprocess_registration_numbers_array(array: list[RegistrationNumber]):
    return sorted(
            [
                {
                    'identifier': item.identifier,
                    'type': item.type,
                }
                for item in array
            ],
            key=lambda x: x['type']  # Sort by 'type'
        )

# COMPARATORS
def compare_value(ex_value: Value, pred_value: Value):

    if not ex_value or not pred_value:
        return 1.0 if ex_value == pred_value else 0.0

    value_score = 1 if ex_value.value == pred_value.value else 0
    # TODO change this to a more robust metric as exact match is not suitable due to the various ways unit can be expressed
    unit_score = 1 if ex_value.unit == pred_value.unit else 0

    return (value_score + unit_score)/2

# PHONE NUMBER
def normalize_phone_number(phone, country='CA'):
    try:
        parsed = phonenumbers.parse(phone, country)
        if phonenumbers.is_possible_number(parsed):
            return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
        else:
            return phone
    except phonenumbers.NumberParseException:
        return phone

# WEBSITE
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

def compare_density_or_volume(dspy_output, expected_output):
    if (((expected_output is None) or if_dictionary_values_are_none(expected_output)) and ((dspy_output is None) or if_dictionary_values_are_none(dspy_output))):
        return 1
    if (((expected_output is None) or if_dictionary_values_are_none(expected_output)) or ((dspy_output is None) or if_dictionary_values_are_none(dspy_output))):
        return 0
    if (dspy_output['value'] == expected_output['value']) and (dspy_output['unit'].lower() == expected_output['unit'].lower()):
        return 1
    return 0

# WEIGHT
def clean_weights(weights):
    # Filter out weights with None values and clean units
    cleaned_weights = [
        {'value': weight.value, 'unit': weight.unit.lower().replace('s', '')}
        for weight in weights
        if weight.value is not None and weight.unit is not None
    ]

    # Sort weights by unit
    cleaned_weights.sort(key=lambda x: x['unit'])

    return cleaned_weights

def compare_weights(dspy_output, expected_output):
    dspy_output_cleaned = clean_weights(dspy_output)
    expected_output_cleaned = clean_weights(expected_output)
    if (len(dspy_output_cleaned) == 0) and (len(expected_output_cleaned) == 0):
        return 1
    if (len(dspy_output_cleaned) == 0) or (len(expected_output_cleaned) == 0):
        return 0
    correct_num = 0
    for dspy_weight, expected_weight in zip(dspy_output_cleaned, expected_output_cleaned):
        if (dspy_weight['value'] == expected_weight['value']) and (dspy_weight['unit'] == expected_weight['unit']):
            correct_num += 1
    return correct_num / len(expected_output_cleaned)

# NUTRIENT VALUE
def compare_nutrient_value(ex_nutrient_value, pred_nutrient_value):
    if not ex_nutrient_value or not pred_nutrient_value:
        return 1.0 if ex_nutrient_value == pred_nutrient_value else 0.0

    # Compare nutrient names using Jaro-Winkler similarity
    nutrient_score = jellyfish.jaro_winkler_similarity(
        preprocess_string(ex_nutrient_value.get('nutrient')),
        preprocess_string(pred_nutrient_value.get('nutrient'))
    )

    # Compare value and unit using the existing compare_value function
    value_score = compare_value(
        Value(value=ex_nutrient_value.get('value'), unit=ex_nutrient_value.get('unit')),
        Value(value=pred_nutrient_value.get('value'), unit=pred_nutrient_value.get('unit'))
    )

    # Average the scores
    return (nutrient_score + value_score) / 2


def compare_ingredients(ex_ingredients: list[NutrientValue], pred_ingredients: list[NutrientValue]) -> float:
    cleaned_expected_output = if_array_is_empty(ex_ingredients)
    cleaned_dspy_output = if_array_is_empty(pred_ingredients)
    if (len(cleaned_expected_output) == 0) and (len(cleaned_dspy_output) == 0):
        return 1.0
    if (len(cleaned_expected_output) == 0) or (len(cleaned_dspy_output) == 0):
        return 0.0

    #TODO: should change the labelled data instead of this step
    for nutrient in ex_ingredients:
        if 'ingredient' in nutrient:
            nutrient['nutrient'] = nutrient.pop('ingredient')

    sorted_pred_ingredients = normalize_ingredients_array(pred_ingredients)
    sorted_ex_ingredients = normalize_ingredients_array(ex_ingredients)

    # Initialize scores
    scores = []

    # Compare all expected nutrients
    for ex_ingredient, pred_ingredient in zip(sorted_ex_ingredients, sorted_pred_ingredients):
        score = compare_nutrient_value(ex_ingredient, pred_ingredient)
        scores.append(score)

    # Penalize for extraneous nutrients
    extra_nutrients = len(sorted_pred_ingredients) - set(sorted_ex_ingredients)
    scores.extend([0.0] * len(extra_nutrients))  # Add one penalty per extra nutrient

    # Compute average score
    return sum(scores) / len(scores) if scores else 0.0

# GUARANTEED ANALYSIS
def compare_guaranteed_analysis(ex_ga: GuaranteedAnalysis, pred_ga: GuaranteedAnalysis):
    if is_gma_none(ex_ga) and is_gma_none(pred_ga):
        return 1
    if is_gma_none(ex_ga) or is_gma_none(pred_ga):
        return 0

    scores = []

    # Compare title using Jaro-Winkler similarity
    if ex_ga.title and pred_ga.title:
        title_score = jellyfish.jaro_winkler_similarity(
            preprocess_string(ex_ga.title),
            preprocess_string(pred_ga.title)
        )
    else:
        title_score = 1.0 if ex_ga.title == pred_ga.title else 0.0
    scores.append(title_score)

    if (pred_ga.is_minimal == ex_ga.is_minimal):
        scores.append(1.0)
    else:
        scores.append(0.0)

    sorted_ex_ga_nutrients = normalize_gma_array(ex_ga.nutrients)
    sorted_pred_ga_nutrients = normalize_gma_array(pred_ga.nutrients)

    for ex_ingredient, pred_ingredient in zip(sorted_ex_ga_nutrients, sorted_pred_ga_nutrients):
        score = compare_nutrient_value(ex_ingredient, pred_ingredient)
        scores.append(score)

    # Penalize for extraneous nutrients
    extra_nutrients = len(sorted_pred_ga_nutrients) - len(sorted_ex_ga_nutrients)
    scores.extend([0.0] * (extra_nutrients))  # Add one penalty per extra nutrient

    # Compute average score
    return sum(scores) / len(scores) if scores else 0.0


def generate_embeddings(text, model="3-large"):
    return EMBEDDING_MODEL.embeddings.create(input=[text], model=model).data[0].embedding
    # return client.embeddings.create(model="text-embedding-ada-002",
    # input=[text]).data[0].embedding


def compare_list_text(ex_value: list[str], pred_value: list[str]):
    if not ex_value or not pred_value:
        return 1.0 if ex_value == pred_value else 0.0

    ex_value_string = "".join(ex_value)
    pred_value_string = "".join(pred_value)

    preprocessed_ex_value_string = preprocess_string(ex_value_string)
    preprocessed_pred_value_string = preprocess_string(pred_value_string)

    ex_value_embeddings = np.array(
        generate_embeddings(preprocessed_ex_value_string)).reshape(1, -1)
    pred_value_embeddings = np.array(
        generate_embeddings(preprocessed_pred_value_string)).reshape(1, -1)

    return cosine_similarity(ex_value_embeddings, pred_value_embeddings)[0][0]

def compare_organizations(ex_organization: list[Organization], pred_organization: list[Organization]):
    cleaned_ex_organization = if_array_is_empty(ex_organization)
    cleaned_pred_organization = if_array_is_empty(pred_organization)
    if (len(cleaned_ex_organization) == 0) and (len(cleaned_pred_organization) == 0):
        return 1.0
    if (len(cleaned_ex_organization) == 0) or (len(cleaned_pred_organization) == 0):
        return 0.0

    sorted_ex_organization = sort_and_preprocess_organization_array(cleaned_ex_organization)
    sorted_pred_organization = sort_and_preprocess_organization_array(cleaned_pred_organization)

    scores = []

    for ex_organization, pred_organization in zip(sorted_ex_organization, sorted_pred_organization):
        # Compare title using Jaro-Winkler similarity
        if ex_organization.name and pred_organization.name:
            name_score = jellyfish.jaro_winkler_similarity(ex_organization.name, pred_organization.name)
        else:
            name_score = 1.0 if ex_organization.name == pred_organization.name else 0.0        

        if ex_organization.address and pred_organization.address:
            address_score = jellyfish.jaro_winkler_similarity(ex_organization.address, pred_organization.address)
        else:
            address_score = 1.0 if ex_organization.address == pred_organization.address else 0.0

        website_score = 1.0 if ex_organization.website == pred_organization.website else 0.0

        phone_score = 1.0 if ex_organization.phone_number == pred_organization.phone_number else 0.0

        scores.append((name_score + address_score + website_score + phone_score) / 4)

    return sum(scores) / len(scores) if scores else 0.0

def compare_registration_number(ex_registration_number: list[RegistrationNumber], pred_registration_number: list[RegistrationNumber]):
    cleaned_ex_registration_number = if_array_is_empty(ex_registration_number)
    cleaned_pred_registration_number = if_array_is_empty(pred_registration_number)
    if (len(cleaned_ex_registration_number) == 0) and (len(cleaned_pred_registration_number) == 0):
        return 1.0
    if (len(cleaned_ex_registration_number) == 0) or (len(cleaned_pred_registration_number) == 0):
        return 0.0

        # sort_and_preprocess_registration_numbers_array
    sorted_ex_registration_number = sort_and_preprocess_registration_numbers_array(cleaned_ex_registration_number)
    sorted_pred_registration_number = sort_and_preprocess_registration_numbers_array(cleaned_pred_registration_number)

    scores = []

    for ex_registration_number, pred_registration_number in zip(sorted_ex_registration_number, sorted_pred_registration_number):
        if ex_registration_number.identifier == pred_registration_number.identifier:
            scores.append(1.0)
        if ex_registration_number.type == pred_registration_number.type:
            scores.append(1.0)

    return sum(scores) / len(scores) if scores else 0.0

# METRIC FUNCTION USED TO RUN EVALS
def validate_inspection(example: dspy.Example, pred: dspy.Prediction, trace=None):
    scores = []

    # because for now example.inspection returns a dict not a FertilizerInspection
    example_inspection = FertilizerInspection.model_validate(
        example.inspection)

    for field_name, _ in FertilizerInspection.model_fields.items():
        example_value = getattr(example_inspection, field_name, None)
        pred_value = getattr(pred.inspection, field_name, None)

        # Evaluate the fields that need to use Jaro Winkler similarity
        if field_name in ["fertiliser_name"]:
            if example_value is not None and pred_value is not None:
                score = jellyfish.jaro_winkler_similarity(
                    preprocess_string(example_value), preprocess_string(pred_value))
                scores.append(score)
            else:
                # Unless both fields are None, score is 0
                score = 1.0 if example_value == pred_value else 0.0
                scores.append(score)
            SCORES_BY_FIELD[field_name].append(score)

        elif field_name in ["organizations"]:
            score = compare_organizations(example_value, pred_value)
            scores.append(score)
            SCORES_BY_FIELD[field_name].append(score)

        elif field_name in ["registration_number"]:
            score = compare_registration_number(example_value, pred_value)
            scores.append(score)
            SCORES_BY_FIELD[field_name].append(score)

        elif field_name in ["lot_number", "npk"]:
            score = 1.0 if example_value == pred_value else 0.0
            scores.append(score)
            SCORES_BY_FIELD[field_name].append(score)

        elif field_name in ["weight"]:
            score = compare_weights(example_value, pred_value)
            scores.append(score)
            SCORES_BY_FIELD[field_name].append(score)

        elif field_name in ["density", "volume"]:
            score = compare_value(example_value, pred_value)
            scores.append(score)
            SCORES_BY_FIELD[field_name].append(score)

        elif field_name in ["guaranteed_analysis_en", "guaranteed_analysis_fr"]:
            score = compare_guaranteed_analysis(example_value, pred_value)
            scores.append(score)
            SCORES_BY_FIELD[field_name].append(score)

        elif field_name in ["ingredients_en", "ingredients_fr"]:
            score = compare_ingredients(example_value, pred_value)
            scores.append(score)
            SCORES_BY_FIELD[field_name].append(score)

        elif field_name in ["cautions_en", "cautions_fr", "instructions_en", "instructions_fr"]:
            score = compare_list_text(example_value, pred_value)
            scores.append(score)
            SCORES_BY_FIELD[field_name].append(score)

            # print(f"\nfor {field_name}\n\tgot : {pred_value}\n\texpected: {example_value}")
            # print(f"score: {score}")

    return sum(scores) / len(scores) if scores else 0.0


if __name__ == "__main__":
    # dataset_path = '/../data/output.csv'
    image_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'test_data', 'labels')

    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'output.csv')

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    df = pd.read_csv(dataset_path)

    dataset = []

    # for image_paths, inspection in df.values:
    for index, row in df.iterrows():
        image_paths = ast.literal_eval(row['image_files'])
        full_image_paths = os.path.join(image_dir, row['instance_path'])
        full_image_paths = [os.path.join(full_image_paths, img_path) for img_path in image_paths]
        inspection = json.loads(row['new_dspy_output'])
        dataset.append(dspy.Example(image_paths=full_image_paths,
                       inspection=inspection).with_inputs("image_paths"))

    evaluate = dspy.Evaluate(
        devset=dataset[:35], metric=validate_inspection, display_progress=True, display_table=True, provide_traceback=True)

    language_program = LanguageProgram(
        AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT, AZURE_API_KEY, AZURE_API_ENDPOINT, AZURE_OPENAI_EMBEDDING_ENDPOINT,
        AZURE_OPENAI_EMBEDDING_KEY)
    evaluate(language_program)

    display_scores_by_field()
