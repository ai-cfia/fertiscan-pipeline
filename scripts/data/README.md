# FertiScan Data Pipeline

This directory contains scripts for fetching, processing, and managing the dataset
used for the FertiScan pipeline. The dataset consists of fertilizer label images
and their corresponding inspection data.

## Dataset Structure

The dataset is organized as follows:

- `test_data/labels/` - Raw data organized by picture set ID
  - `{picture_set_id}/` - Directory for each picture set
    - `{file_id}.jpg` - Image files of fertilizer labels
    - `expected_output.json` - Labeled inspection data for evaluation
- `data/processed/` - Processed data ready for model training and evaluation
  - `dataset.csv` - CSV file containing image paths and inspection data

## Purpose

The dataset is created to:

1. Train and optimize DSPy modules for fertilizer label extraction
2. Evaluate the performance of the pipeline against real-world examples
3. Support a systematic approach to model optimization and validation
4. Create a data flywheel that grows with user interactions

## Dataset Creation

### Fetching Data

The `fetch_data.py` script is designed to load data from our continuously growing
database and preprocess it for the FertiScan pipeline. This is a key component in
creating our data flywheel, as it:

1. Connects to the API endpoint specified in the environment variables
2. Fetches inspection data along with associated images from our growing user database
3. Organizes the data into a structured format in the `test_data/labels` directory
4. Scales automatically as more users interact with the system, enabling increasingly
   sophisticated model training

### Processing Data

The `make_dataset.py` script converts the raw data into a format suitable for
training and evaluation:

1. Scans the `test_data/labels` directory for all picture sets
2. For each picture set, collects the image paths and expected output
3. Creates a CSV file with columns for image paths and inspection data

## Usage Guidelines

### Prerequisites

1. Set up the environment variables by creating a `.env` file with:

```bash
API_ENDPOINT=<your_api_endpoint>
```

### Data Fetching

To fetch data from the API and build a training dataset from our growing user database:

```bash
python scripts/data/fetch_data.py
```

### Data Processing

To process the fetched data into a training-ready CSV:

```bash
python scripts/data/make_dataset.py
```

## Training and Optimization

This dataset is designed to work with DSPy optimizers. As our database grows with
user interactions, we can employ increasingly sophisticated optimization techniques:

1. Split the dataset with an 20/80 train/evaluation split
2. Start with simpler optimizers like `BootstrapFewShot` for example injection
3. Progress to more complex optimizers as the dataset grows

### Optimization Steps

1. Load the dataset from the processed CSV
2. Split the data into training and evaluation sets
3. Use the appropriate DSPy optimizer based on dataset size:
   - For ~10 examples: `BootstrapFewShot`
   - For 50+ examples: `BootstrapFewShotWithRandomSearch`
   - For 200+ examples: `MIPROv2`
4. Evaluate the optimized module using the evaluation set

## Data Flywheel Strategy

The data pipeline is designed as a flywheel that:

1. Grows automatically as more users interact with the system
2. Allows for periodic retraining of models with fresh data
3. Improves model accuracy over time as the dataset diversity increases
4. Enables transitioning to more sophisticated optimization techniques as data
   volume increases

## Limitations and Future Work

1. **Initial Dataset Size**: While the current dataset is relatively small (~35
   examples), our database is continuously growing, which will soon enable more
   advanced optimization techniques.
2. **Validation**: As the dataset grows, cross-validation should be implemented to
   ensure generalizability.
3. **Automated Retraining**: Future work should include automated pipelines to
   retrain models as the database grows past certain thresholds.
4. **Backend API**: The application's backend API currently does not support
   complete search functionality. It currently only supports basic search queries
   like by user ID and inspection ID. However, future enhancements will include
   advanced search capabilities. The groundwork for this feature is already laid
   out in the project's architecture.

## Project Context

This dataset is a crucial component of the FertiScan pipeline, enabling systematic
optimization of DSPy modules for extracting structured information from fertilizer
labels. The trained models will be used to automatically extract relevant information
from fertilizer product labels to provide accurate information to users.

Our database-driven approach ensures that as more users interact with the system,
the dataset will naturally scale up, enabling more sophisticated optimization
techniques, potentially including fine-tuning of open models like Deepseek.
