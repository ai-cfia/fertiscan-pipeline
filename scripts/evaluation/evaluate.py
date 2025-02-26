import re
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
from PIL import Image

from pipeline.modules.main_module import MainModule
from pipeline.schemas.inspection import FertilizerInspection, GuaranteedAnalysis, NutrientValue, Organization, RegistrationNumber, Value
from pipeline.schemas.settings import Settings

SETTINGS = Settings()

EMBEDDING_MODEL = AzureOpenAI(
    api_version="2023-05-15",
    azure_endpoint=SETTINGS.llm_embedding_api_endpoint,
    api_key=SETTINGS.llm_embedding_api_key.get_secret_value(),
)

SCORES_BY_FIELD = {
    "organizations": [],
    "fertiliser_name": [],
    "registration_number": [],
    "lot_number": [],
    "weight": [],
    "density": [],
    "volume": [],
    "npk": [],
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
    lowercased = input_string.lower()
    cleaned = re.sub(r'[^a-z0-9\s]', '', lowercased)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def normalize_phone_number(phone, country='CA'):
    """
    Normalize phone numbers to E.164 format.

    Args:
        phone (str): Phone number to normalize.
        country (str): Country code for localization. Defaults to 'CA'.

    Returns:
        str: Normalized phone number or original phone number if invalid.
    """
    try:
        parsed = phonenumbers.parse(phone, country)
        if phonenumbers.is_possible_number(parsed):
            return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
        else:
            return phone
    except phonenumbers.NumberParseException:
        return phone


def normalize_website(url):
    """
    Normalize and simplify a website URL.

    Args:
        url (str): The website URL to normalize.

    Returns:
        str: Normalized URL, or original input if normalization fails.
    """
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
    return url


def display_scores_by_field(scores_by_field=SCORES_BY_FIELD):
    """
    Display average scores for each field.

    Args:
        scores_by_field (dict): Dictionary containing the scores for each field.
        Defaults to the global variable SCORES_BY_FIELD
    """
    print("\n--Scores by field--")
    for field, score_list in scores_by_field.items():
        average = sum(score_list)/len(score_list)
        print(f"{field:<30} | {average}")

# COMPARATORS
def compare_value(ex_value: Value, pred_value: Value):

    if not ex_value or not pred_value:
        return 1.0 if ex_value == pred_value else 0.0

    value_score = 1 if ex_value.value == pred_value.value else 0
    # TODO change this to a more robust metric as exact match is not suitable due to the various ways unit can be expressed
    unit_score = 1 if ex_value.unit == pred_value.unit else 0

    return (value_score + unit_score)/2

# TODO there is a flaw here since this assumes the list of weights is ordered consistently on the prediction and the example
def compare_weight(example_weight: list[Value], pred_weight: list[Value]):
    if not example_weight or not pred_weight:
        return 1 if example_weight == pred_weight else 0

    example_weight.sort(key=lambda x: x.value)
    pred_weight.sort(key=lambda x: x.value)

    scores = []

    for ex_value, pred_value in zip(example_weight, pred_weight):
        score = compare_value(ex_value, pred_value)
        scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0

def compare_organization(ex_org: Organization, pred_org: Organization):
    scores = []

    # Compare name
    if ex_org.name and pred_org.name:
        name_score = jellyfish.jaro_winkler_similarity(
            preprocess_string(ex_org.name),
            preprocess_string(pred_org.name)
        )
        scores.append(name_score)

    # Compare address
    if ex_org.address and pred_org.address:
        address_score = jellyfish.jaro_winkler_similarity(
            preprocess_string(ex_org.address),
            preprocess_string(pred_org.address)
        )
        scores.append(address_score)

    # Compare website
    website_score = 1.0 if ex_org.website == pred_org.website else 0.0
    scores.append(website_score)

    # Compare phone number
    phone_score = 1.0 if ex_org.phone_number == pred_org.phone_number else 0.0
    scores.append(phone_score)

    return sum(scores) / len(scores) if scores else 0.0

# TODO there is a flaw here since this assumes the list of weights is ordered consistently on the prediction and the example
def compare_organizations(ex_orgs: list[Organization], pred_orgs: list[Organization]):
    if not ex_orgs and not pred_orgs:
        return 1.0
    if not ex_orgs or not pred_orgs:
        return 0.0

    ex_orgs.sort(key=lambda x: x.name)
    pred_orgs.sort(key=lambda x: x.name)

    scores = []
    for ex_org in ex_orgs:
        best_score = max(
            compare_organization(ex_org, pred_org)
            for pred_org in pred_orgs
        )
        scores.append(best_score)

    return sum(scores) / len(scores)

def compare_registration_numbers(ex_regs: list[RegistrationNumber], pred_regs: list[RegistrationNumber]):
    if not ex_regs and not pred_regs:
        return 1.0
    if not ex_regs or not pred_regs:
        return 0.0

    ex_regs.sort(key=lambda x: x.identifier)
    pred_regs.sort(key=lambda x: x.identifier)

    scores = []
    for ex_reg in ex_regs:
        best_score = 0.0
        for pred_reg in pred_regs:
            id_score = 1.0 if ex_reg.identifier == pred_reg.identifier else 0.0
            type_score = 1.0 if ex_reg.type == pred_reg.type else 0.0
            score = (id_score + type_score) / 2
            best_score = max(best_score, score)
        scores.append(best_score)

    return sum(scores) / len(scores)

def compare_nutrient_value(ex_nutrient_value: NutrientValue, pred_nutrient_value: NutrientValue):
    if not ex_nutrient_value or not pred_nutrient_value:
        return 1.0 if ex_nutrient_value == pred_nutrient_value else 0.0

    nutrient_score = jellyfish.jaro_winkler_similarity(
        preprocess_string(ex_nutrient_value.nutrient),
        preprocess_string(pred_nutrient_value.nutrient)
    )

    value_score = compare_value(
        Value(value=ex_nutrient_value.value, unit=ex_nutrient_value.unit),
        Value(value=pred_nutrient_value.value, unit=pred_nutrient_value.unit)
    )

    return (nutrient_score + value_score) / 2


def compare_ingredients(ex_ingredients: list[NutrientValue], pred_ingredients: list[NutrientValue]) -> float:
    if not ex_ingredients and not pred_ingredients:
        return 1.0
    if not ex_ingredients or not pred_ingredients:
        return 0.0

    # Create lookup dictionaries for fast comparisons
    ex_dict = {nv.nutrient: nv for nv in ex_ingredients}
    pred_dict = {nv.nutrient: nv for nv in pred_ingredients}

    scores = []

    for nutrient, ex_nv in ex_dict.items():
        pred_nv = pred_dict.get(nutrient)
        if pred_nv:
            score = compare_nutrient_value(ex_nv, pred_nv)
        else:
            score = 0.0  # Missing nutrient in prediction
        scores.append(score)

    # Penalize for extraneous nutrients
    extra_nutrients = set(pred_dict.keys()) - set(ex_dict.keys())
    scores.extend([0.0] * len(extra_nutrients))  # Add one penalty per extra nutrient

    return sum(scores) / len(scores) if scores else 0.0


# TODO there is a flaw that produces inconsistent result with the way this checks for matches
def compare_guaranteed_analysis(ex_ga: GuaranteedAnalysis, pred_ga: GuaranteedAnalysis):
    if not ex_ga or not pred_ga:
        return 1.0 if ex_ga == pred_ga else 0.0

    scores = []

    if ex_ga.title and pred_ga.title:
        title_score = jellyfish.jaro_winkler_similarity(
            preprocess_string(ex_ga.title),
            preprocess_string(pred_ga.title)
        )
    else:
        title_score = 1.0 if ex_ga.title == pred_ga.title else 0.0
    scores.append(title_score)

    nutrients_score = compare_ingredients(ex_ga.nutrients, pred_ga.nutrients)
    scores.append(nutrients_score)

    return sum(scores) / len(scores) if scores else 0.0


def generate_embeddings(text, model="ada"):
    return EMBEDDING_MODEL.embeddings.create(input=[text], model=model).data[0].embedding


def compare_list_text(ex_value: list[str], pred_value: list[str]):
    if not ex_value or not pred_value:
        return 1.0 if ex_value == pred_value else 0.0

    ex_value_string = "".join(ex_value)
    pred_value_string = "".join(pred_value)

    ex_value_embeddings = np.array(
        generate_embeddings(ex_value_string)).reshape(1, -1)
    pred_value_embeddings = np.array(
        generate_embeddings(pred_value_string)).reshape(1, -1)

    return cosine_similarity(ex_value_embeddings, pred_value_embeddings)[0][0]


# METRIC FUNCTION USED TO RUN EVALS
def validate_inspection(example: dspy.Example, pred: dspy.Prediction, trace=None):
    scores = []
    example_inspection = FertilizerInspection.model_validate(example.inspection)

    for field_name, _ in FertilizerInspection.model_fields.items():
        example_value = getattr(example_inspection, field_name, None)
        pred_value = getattr(pred.inspection, field_name, None)

        if field_name == "organizations":
            score = compare_organizations(example_value, pred_value)
            scores.append(score)
            SCORES_BY_FIELD[field_name].append(score)

        elif field_name == "registration_number":
            score = compare_registration_numbers(example_value, pred_value)
            scores.append(score)
            SCORES_BY_FIELD[field_name].append(score)

        elif field_name == "fertiliser_name":
            if example_value is not None and pred_value is not None:
                score = jellyfish.jaro_winkler_similarity(
                    preprocess_string(example_value),
                    preprocess_string(pred_value)
                )
            else:
                score = 1.0 if example_value == pred_value else 0.0
            scores.append(score)
            SCORES_BY_FIELD[field_name].append(score)

        # Continue with rest of field comparisons...

    return sum(scores) / len(scores) if scores else 0.0


if __name__ == "__main__":
    dataset_path = 'data/processed/dataset.csv'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    df = pd.read_csv(dataset_path)

    dataset = []

    for image_paths, inspection in df.values:
        image_paths = ast.literal_eval(image_paths)
        image_list = []
        for image_path in image_paths:
           image_list.append(Image.open(image_path))

        inspection = json.loads(inspection)
        dataset.append(dspy.Example(images=image_list,
                       inspection=inspection).with_inputs("images"))

    evaluate = dspy.Evaluate(
        devset=dataset[:35], metric=validate_inspection, display_progress=True, display_table=True)

    main_module = MainModule(settings=SETTINGS)
    evaluate(main_module)

    display_scores_by_field()
