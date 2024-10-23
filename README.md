# fertiscan-pipeline

This repository contains the core analysis pipeline for FertiScan. It is
designed to be used as a standalone Python package that can be integrated with
other projects, such as the
[fertiscan-backend](https://github.com/ai-cfia/fertiscan-backend).

## Setup for Development

### Prerequisites

- Python 3.8+
- [pip](https://pip.pypa.io/en/stable/installation/)
- [virtualenv](https://virtualenv.pypa.io/en/latest/installation.html)
- Azure Document Intelligence and OpenAI API keys

### Installation

To install the package directly from GitHub:

#### **Direct Installation using pip**

Run the following command in your terminal:

```sh
pip install git+https://github.com/ai-cfia/fertiscan-pipeline.git@main
```

#### **Installation via requirements.txt**

   Add the following line to your `requirements.txt` file:

   ```sh
   git+https://github.com/ai-cfia/fertiscan-pipeline.git@main
   ```

   Then, install the dependencies with:

   ```sh
   pip install -r requirements.txt
   ```

### Environment Variables

Create a `.env` file and set the necessary environment variables:

```ini
AZURE_API_ENDPOINT=your_azure_form_recognizer_endpoint
AZURE_API_KEY=your_azure_form_recognizer_key
AZURE_OPENAI_API_ENDPOINT=your_azure_openai_endpoint
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_DEPLOYMENT=your_azure_openai_deployment
```

## Packaging and release workflow

The pipeline triggers on PRs to check code quality, markdown, repository
standards, and ensures that the version in `pyproject.toml` is bumped. When a PR
is merged, the workflow automatically creates a release based on the version in
`pyproject.toml`. The latest releases and changelogs are available
[here](https://github.com/ai-cfia/fertiscan-pipeline/releases).

To use this package in other projects, add it to your `requirements.txt` (e.g.,
in the [fertiscan-backend](https://github.com/ai-cfia/fertiscan-backend)):

```sh
git+https://github.com/ai-cfia/fertiscan-pipeline.git@vX.X.X
```

Where `vX.X.X` is the version from the [release
page](https://github.com/ai-cfia/fertiscan-pipeline/releases).
