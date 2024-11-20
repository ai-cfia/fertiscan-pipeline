# Pipeline Structure

```markdown
└── Pipeline                    <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes Pipeline a Python module and defines the interface for communications
    │
    ├── component               <- Reusable components used in the pipeline
    │   └── ...
    │
    ├── schemas/                <- Pydantic schemas used
    │   └── ...
    │
    └── modules/                <- DSPy Modules used in the pipeline.
        ├── ...
        └── LanguageProgram.py  <- The Main Module of the pipeline.
```
