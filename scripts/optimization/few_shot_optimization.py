# %%
import os
import pandas as pd
import dspy
import json
import ast
import random
from typing import List, Optional
from dspy.teleprompt import LabeledFewShot
from pipeline.schemas.inspection import FertilizerInspection
from pipeline.modules.main_module import MainModule
from pipeline.schemas.settings import Settings
from PIL import Image
from datetime import datetime

# Set a constant seed for reproducibility
SEED = 42 # DO NOT CHANGE THIS UNLESS ABSOLUTELY CONFIDENT IT IS WARENTED
random.seed(SEED)

# Load dataset from CSV and parse `inspection` column with Pydantic
def load_csv_dataset(file_path: str) -> List[dict]:
    """
    Loads a CSV file into a list of dictionaries, ensuring correct parsing of structured fields.
    
    Args:
        file_path (str): Path to the CSV file.

    Returns:
        List[dict]: Each dictionary represents a row of the dataset.
    """
    df = pd.read_csv(file_path)

    # Convert `inspection` column from string to FertilizerInspection objects
    def parse_inspection(inspection_str: str) -> Optional[FertilizerInspection]:
        inspection_data = json.loads(inspection_str)
        return FertilizerInspection.model_validate(inspection_data)

    df["inspection"] = df["inspection"].apply(parse_inspection)

    # Convert `image_paths` from string representation of list to actual list of PIL images
    def load_images(image_list_str: str):
        try:
            image_paths = ast.literal_eval(image_list_str) if isinstance(image_list_str, str) else image_list_str
            images = [Image.open(img_path) for img_path in image_paths if os.path.exists(img_path)]
            return images
        except Exception as e:
            print(f"Warning: Failed to load images {image_list_str}. Error: {e}")
            return []

    df["image_paths"] = df["image_paths"].apply(load_images)

    # Filter out rows where inspection parsing failed
    df = df[df["inspection"].notnull()]
    df = df[df["image_paths"].notnull()]

    return df.to_dict(orient="records")

# Convert each row into a DSPy Example
def prepare_dataset(dataset: List[dict]) -> List[dspy.Example]:
    """
    Converts dataset rows into DSPy Example objects.

    Args:
        dataset (List[dict]): The dataset as a list of dictionaries.

    Returns:
        List[dspy.Example]: DSPy Examples with inputs and expected outputs.
    """
    examples = []
    for row in dataset:
        example = dspy.Example(
            images=row["image_paths"],
            inspection=row["inspection"]
        ).with_inputs("images")
        examples.append(example)
    return examples

# Split dataset into training and evaluation sets
def split_dataset(dataset: List[dspy.Example], train_ratio: float = 0.5) -> tuple[List[dspy.Example], List[dspy.Example]]:
    """
    Splits the dataset into training and evaluation sets.

    Args:
        dataset (List[dspy.Example]): The full dataset.
        train_ratio (float): Ratio of data to be used for training.

    Returns:
        Tuple[List[dspy.Example], List[dspy.Example]]: Train and test datasets.
    """
    random.shuffle(dataset)
    split_index = int(len(dataset) * train_ratio)
    return dataset[:split_index], dataset[split_index:]


# Main execution
if __name__ == "__main__":

    # Load and prepare dataset
    print("Loading the dataset...")
    file_path = os.path.join(os.getcwd(), "data", "processed", "dataset.csv")
    dataset = load_csv_dataset(file_path)
    dspy_examples = prepare_dataset(dataset)

    # Split into training and evaluation sets
    train_set, eval_set = split_dataset(dspy_examples)

    # initialize the module
    print("Preparing DSPy Module...")
    settings = Settings()
    predictor = MainModule(settings=settings)
    
    # Define and compile the teleprompter optimizer
    print("Starting the optimization...")
    teleprompter = LabeledFewShot(k=3)
    compiled_model = teleprompter.compile(student=predictor, trainset=train_set)

    # Save the modele
    print("Saving the optimized module...")
    checkpoint_folder = os.path.join(os.getcwd(), "checkpoints")
    os.makedirs(os.path.dirname(checkpoint_folder), exist_ok=True)
    timestamp = datetime.now().isoformat()
    checkpoint_file = f"checkpoints-{timestamp}.json"
    compiled_model.save(os.path.join(checkpoint_folder, checkpoint_file), save_program=False)

