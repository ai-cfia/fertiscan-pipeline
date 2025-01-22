# Pipeline Structure

```markdown
└── Pipeline                    <- Source code that will be available when importing the package.
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
        └── MainModule.py
```
