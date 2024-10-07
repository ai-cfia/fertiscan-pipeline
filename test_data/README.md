# Test Data

This folder hosts all the data needed to perform testing on the processing
pipeline (label images and their expected output).

Please follow the following structure when adding new test cases:

```text
├── test_data/                # Test images and related data
│   ├── labels/               # Folders organized by test case
│   │   ├── label_001/        # Each folder contains images and expected 
│   │   │   ├── img_001.jpg   # output JSON
│   │   │   ├── img_002.jpg
│   │   │   └── expected_output.json
│   │   ├── label_002/
│   │   │   ├── img_001.jpg
│   │   │   ├── img_002.jpg
│   │   │   └── expected_output.json
│   │   └── ...
```
