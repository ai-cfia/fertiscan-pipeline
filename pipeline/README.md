# Transitioning from Direct LLM Calls to DSPy in FertiScan Pipeline

This documentation addresses issue #94, providing a comprehensive resource to understand the work done to transition the FertiScan pipeline from direct LLM calls to DSPy.

## 1. Contextualizing DSPy

[DSPy](https://github.com/stanfordnlp/dspy) is a framework developed by Stanford NLP that enables programmatic prompt engineering and LLM optimization. It provides a structured approach to building LLM-powered applications that moves beyond ad-hoc prompt engineering.

### The Main Moving Parts of a DSPy Program

1. **Signature**: Defines input → output behavior, creating a contract for what goes in and what comes out
2. **Predictor/Built-in Modules**: The strategy used to solve the task (e.g., ChainOfThought, RAG)
3. **Metrics**: A numerical representation of the quality of the output
4. **Optimizers**: Built-in tools to optimize the model toward a specific behavior

### The Main Iterative Loop of Creating a DSPy Module

1. Define the task and the overall module input and output
2. Define the initial pipeline and the data flow
3. Find examples to test the program against
4. Define an initial metric to assess what "good" looks like
5. Analyze the evaluation results and identify areas where the module struggles
6. Attempt to solve pain points using various vectors of improvement
7. Rinse and repeat

## 2. The Work Done So Far

### Setting Up a Project Structure

We've established a modular project structure that ensures clean separation of concerns:

```plaintext
└── Pipeline                    <- Source code available when importing the package
    │
    ├── __init__.py             <- Module interface for communications
    │
    ├── components/             <- Reusable components used in the pipeline
    │   ├── label.py            <- Label storage and document creation
    │   └── ocr.py              <- OCR integration with Azure Document Intelligence
    │
    ├── schemas/                <- Pydantic schemas for data validation
    │   ├── inspection.py       <- Defines the FertilizerInspection model
    │   └── settings.py         <- Configurations and environment settings
    │
    └── modules/                <- DSPy Modules used in the pipeline
        ├── __init__.py
        └── main_module.py      <- Main DSPy module implementation
```

This structure ensures:

- Clear separation between data schemas, external services, and LLM logic
- Modularity for easy testing and component replacement
- Consistent interfaces for pipeline integration

### Defining the Task, Inputs, Outputs, and Pipeline

#### The Signature

We defined an `Inspector` signature in [main_module.py](pipeline/modules/main_module.py) that establishes the contract between inputs and outputs. This signature:

- Creates a clear contract for what inputs are required (OCR text and requirements)
- Defines the expected output (a structured `FertilizerInspection` object)
- Provides context through a docstring that guides the LLM's behavior

The signature defines OCR text and requirements as input fields, while specifying a `FertilizerInspection` object as the output field. The docstring informs the model of its role as a fertilizer label inspector working for the Canadian Food Inspection Agency, tasking it with classifying information from the provided text accurately and comprehensively.

#### The Pipeline Flow

The pipeline follows these key steps:

```mermaid
flowchart LR
    A[Images] --> B[LabelStorage]
    B --> C[Document Creation]
    C --> D[Azure Document Intelligence]
    D --> E[OCR Text Extraction]
    E --> F[DSPy Inspector Module]
    F --> G[FertilizerInspection Schema]
    G --> H[Validation]
    H --> I[Return Validated Inspection]
```

### Defining the Metric Function

We've implemented a Pydantic-based schema system that serves as an implicit metric:

- The `FertilizerInspection` schema and its sub-models define what a correct extraction looks like
- Field validators enforce data quality (e.g., phone number formatting, NPK ratio validation)
- Missing or malformed data is automatically handled through validation

This schema-based approach provides immediate feedback on extraction quality and identifies specific fields where the model struggles.

### Optimization

Our current optimization work is in the early stages due to data limitations:

- The existing dataset includes approximately 35 examples
- With a typical 20/80 train/test split, this leaves only ~7 training examples

Given this constraint, we've focused on:

- Building the DSPy infrastructure with optimization in mind
- Using `ChainOfThought` reasoning to improve extraction quality
- Setting up a caching system to make iterations faster during development

## 3. The Next Steps

### System Performance Assessment

Our initial DSPy implementation has identified several areas for improvement:

1. **Bilingual Extraction**: Challenges in correctly separating English and French content
2. **Complex Structured Data**: Difficulties in consistently extracting nested information (e.g., guaranteed analysis)
3. **Formatting Inconsistencies**: Struggles with variations in how information is presented across different labels

### Vectors of Improvement

#### Breaking Down the Monolithic Module

Our current implementation uses a single monolithic DSPy module. A possible improvement for the future would be to break this down into specialized modules:

```plaintext
Inspector (Main Module)
├── OrganizationExtractor
├── ProductInfoExtractor
├── NutrientAnalysisExtractor
├── CautionsExtractor
└── InstructionsExtractor
```

This modular approach would potentially allow:

- More focused training and optimization for each subtask
- Shorter, more specific prompts
- Better handling of specialized extraction needs (e.g., nutrient values vs. safety instructions)

However, given our current time constraints, we'll continue with the monolithic approach while noting this architectural improvement for future iterations.

#### Optimizing Modules

DSPy offers several optimization strategies that could be considered for future improvements as our dataset grows:

1. **Demonstration Tuning**
   - Carefully selected few-shot examples could be added to each module
   - Targeted demonstrations could help with challenging extraction scenarios

2. **Holistic Tuning (Prompt + Demonstration)**
   - DSPy's MIPROv2 optimizer could be valuable once a larger dataset (~200 training examples) is available
   - This approach could optimize prompts and demonstrations simultaneously for better performance

3. **Metric Refinement**
   - More sophisticated metrics beyond schema validation could be developed
   - Domain-specific evaluation criteria for fertilizer information could enhance quality assessment

#### Data Flywheel Implementation

To address our data limitations, we're designing a data flywheel:

1. Deploy the initial DSPy pipeline to the production application
2. Collect user-submitted fertilizer labels and corrections
3. Automatically incorporate new examples into the training dataset
4. Periodically retrain and optimize the DSPy modules
5. Deploy improved models back to production

This approach will create a virtuous cycle where user interactions continuously improve the system.

## 4. Architecture and Implementation Details

### Key Components

#### LabelStorage

Handles image processing and document creation:

- Stores and manages multiple images of a fertilizer label
- Creates composite documents (PDF/PNG) for OCR processing
- Provides clear memory management (clearing images after processing)

#### OCR Integration

Connects with Azure Document Intelligence:

- Extracts text from label images with high accuracy
- Preserves document structure for better context
- Returns markdown-formatted text for easier processing

#### DSPy Module Implementation

The `MainModule` class implements the core DSPy functionality:

- Configures the Azure OpenAI connection with appropriate parameters
- Implements the `forward()` method that processes images through the pipeline
- Uses `ChainOfThought` reasoning for improved extraction quality

#### Schema System

Our extensive Pydantic schema system:

- Defines valid data structures for all extracted information
- Implements validation rules and data normalization
- Provides automatic type conversion and formatting
- Handles bilingual requirements (English/French fields)

### Integration With External Services

The pipeline integrates with:

1. **Azure Document Intelligence** for OCR processing
2. **Azure OpenAI** for LLM-based extraction
3. **OpenTelemetry** (optional) for observability

## 5. Lessons Learned and Best Practices

### Schema-First Design

We found that starting with a well-defined schema provided several benefits:

- Clearer communication about expected outputs
- Built-in validation reducing post-processing
- Natural structure for DSPy signatures

### Modular Component Design

Our component-based approach allowed:

- Easier testing of individual pipeline elements
- Simplified integration with external services
- Better separation of concerns

### Challenges With DSPy Integration

Some challenges we encountered:

- Adapting to DSPy's structured approach requires rethinking direct LLM calls
- Limited documentation for some advanced DSPy features
- Initial overhead of setting up the DSPy infrastructure

## 6. Conclusion

Our transition from direct LLM calls to DSPy has created a more structured, maintainable, and optimizable pipeline for fertilizer label analysis. The key advantages of this approach include:

1. **Structured Development**: Clear separation between signature definition, module implementation, and optimization
2. **Schema Validation**: Automatic validation and normalization of extracted data
3. **Optimization Potential**: Foundation for systematic improvement through DSPy's optimization tools
4. **Modularity**: Clean component interfaces that simplify testing and maintenance

While we're still in the early stages of optimization due to data limitations, the infrastructure is now in place to systematically improve extraction quality as our dataset grows. The next major focus will be breaking down the monolithic module into specialized components and implementing a data flywheel to continuously improve performance.

## 7. Resources and References

- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [Azure Document Intelligence Documentation](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
