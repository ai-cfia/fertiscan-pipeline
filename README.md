# fertiscan-pipeline

([*Le français est disponible au bas de la page*](#Français))

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

---

## Français

Ce dépôt contient le pipeline d'analyse principal pour FertiScan. Il est conçu
pour être utilisé comme un package Python autonome qui peut être intégré à
d'autres projets, tels que
[fertiscan-backend](https://github.com/ai-cfia/fertiscan-backend).

## Configuration pour le développement

### Prérequis

- Python 3.8+
- [pip](https://pip.pypa.io/en/stable/installation/)
- [virtualenv](https://virtualenv.pypa.io/en/latest/installation.html)
- Clés API pour Azure Document Intelligence et OpenAI

### Installation via GitHub

Pour installer le package directement via GitHub :

#### **Installation directe avec pip**

Exécutez la commande suivante dans votre terminal :

```sh
pip install git+https://github.com/ai-cfia/fertiscan-pipeline.git@main
```

#### **Installation avec requirements.txt**

   Ajoutez la ligne suivante à votre fichier `requirements.txt` :

   ```sh
   git+https://github.com/ai-cfia/fertiscan-pipeline.git@main
   ```

   Ensuite, installez les dépendances avec :

   ```sh
   pip install -r requirements.txt
   ```

### Variables d'environnement

Créez un fichier `.env` et configurez les variables d'environnement
nécessaires :

```ini
AZURE_API_ENDPOINT=your_azure_form_recognizer_endpoint
AZURE_API_KEY=your_azure_form_recognizer_key
AZURE_OPENAI_API_ENDPOINT=your_azure_openai_endpoint
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_DEPLOYMENT=your_azure_openai_deployment
```

## Workflow de packaging et de publication

Le pipeline se déclenche sur les Pull Requests pour vérifier la qualité du code,
les fichiers markdown, les normes du dépôt et s'assurer que la version dans
`pyproject.toml` est mise à jour. Lorsqu'une PR est fusionnée, le workflow crée
automatiquement une version basée sur la version dans `pyproject.toml`. Les
dernières versions et journaux des modifications sont disponibles
[ici](https://github.com/ai-cfia/fertiscan-pipeline/releases).

Pour utiliser ce package dans d'autres projets, ajoutez-le à votre
`requirements.txt` (par exemple, dans
[fertiscan-backend](https://github.com/ai-cfia/fertiscan-backend)) :

```sh
git+https://github.com/ai-cfia/fertiscan-pipeline.git@vX.X.X
```

Où `vX.X.X` est la version provenant de la [page des
versions](https://github.com/ai-cfia/fertiscan-pipeline/releases).
