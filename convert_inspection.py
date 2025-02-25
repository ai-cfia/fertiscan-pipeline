import json
from pathlib import Path
from typing import Dict, Any
from pipeline.schemas.inspection import (
    FertilizerInspection,
    Organization,
)

def convert_organization(data: Dict[str, Any], prefix: str) -> Organization:
    return Organization(
        name=data.get(f"{prefix}_name"),
        address=data.get(f"{prefix}_address"),
        website=data.get(f"{prefix}_website"),
        phone_number=data.get(f"{prefix}_phone_number")
    )

def convert_json(input_data: Dict[str, Any]) -> Dict[str, Any]:
    # Create organizations list from company and manufacturer data
    organizations = []

    company = convert_organization(input_data, "company")
    if any([company.name, company.address, company.website, company.phone_number]):
        organizations.append(company)

    manufacturer = convert_organization(input_data, "manufacturer")
    if any([manufacturer.name, manufacturer.address, manufacturer.website, manufacturer.phone_number]):
        organizations.append(manufacturer)

    # Convert registration number to new format
    registration_numbers = []
    if input_data.get("registration_number"):
        registration_numbers.append({
            "identifier": input_data["registration_number"],
            "type": "fertilizer_product"
        })

    # Create new data structure
    new_data = {
        "organizations": organizations,
        "fertiliser_name": input_data.get("fertiliser_name"),
        "registration_number": registration_numbers,
        "lot_number": input_data.get("lot_number"),
        "weight": input_data.get("weight", []),
        "density": input_data.get("density"),
        "volume": input_data.get("volume"),
        "npk": input_data.get("npk"),
        "guaranteed_analysis_en": input_data.get("guaranteed_analysis_en"),
        "guaranteed_analysis_fr": input_data.get("guaranteed_analysis_fr"),
        "cautions_en": input_data.get("cautions_en", []),
        "cautions_fr": input_data.get("cautions_fr", []),
        "instructions_en": input_data.get("instructions_en", []),
        "instructions_fr": input_data.get("instructions_fr", []),
        "ingredients_en": input_data.get("ingredients_en", []),
        "ingredients_fr": input_data.get("ingredients_fr", [])
    }

    # Validate using Pydantic model
    fertilizer_inspection = FertilizerInspection(**new_data)
    return fertilizer_inspection.model_dump(exclude_none=True)

def main():
    # Base directory containing all label folders
    base_dir = Path("./test_data/labels")

    # Iterate through all label directories
    for label_dir in sorted(base_dir.glob("label_*")):
        json_file = label_dir / "expected_output.json"

        if not json_file.exists():
            print(f"Skipping {label_dir}: No expected_output.json found")
            continue

        try:
            print(f"Processing {json_file}")

            # Read input JSON file
            with json_file.open("r", encoding="utf-8") as f:
                input_data = json.load(f)

            # Convert JSON
            converted_data = convert_json(input_data)

            # Write converted JSON back to file
            with json_file.open("w", encoding="utf-8") as f:
                json.dump(converted_data, f, indent=2, ensure_ascii=False)

            print(f"Successfully converted {json_file}")

        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")

if __name__ == "__main__":
    main()
