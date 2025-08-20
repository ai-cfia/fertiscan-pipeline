import pandas as pd
import json

# Function to transform JSON column
def transform_json(json_str):
    try:
        # Parse JSON string
        data = json.loads(json_str)
        print(type(data))
        print(data) 

        # Extract company and manufacturer details
        organizations = []

        if data.get("company_name") or data.get("company_address"):
            organizations.append({
                "name": data.get("company_name"),
                "address": data.get("company_address"),
                "phone_number": data.get("company_phone_number"),
                "website": data.get("company_website"),
            })

        if data.get("manufacturer_name") or data.get("manufacturer_address"):
            organizations.append({
                "name": data.get("manufacturer_name"),
                "address": data.get("manufacturer_address"),
                "phone_number": data.get("manufacturer_phone_number"),
                "website": data.get("manufacturer_website"),
            })

        # Remove old fields and replace with new structure
        for key in ["company_name", "company_address", "company_phone_number", "company_website",
                    "manufacturer_name", "manufacturer_address", "manufacturer_phone_number", "manufacturer_website"]:
            data.pop(key, None)
        
        data["organizations"] = organizations

        # Transform registration_number
        if data.get("registration_number"):
            data["registration_number"] = [{
                "identifier": data["registration_number"],
                "type": "fertilizer_product"
            }]

        return json.dumps(data, ensure_ascii=False)
    
    except json.JSONDecodeError:
        return json_str  # Return original string if parsing fails


if __name__ == "__main__":
    input_csv = "with_dspy.csv"  # Replace with your actual file name
    output_csv = "output.csv"
    # Read the CSV file
    df = pd.read_csv(input_csv)
    print()

    # Apply transformation
    df['new_dspy_output'] = df.apply(lambda row:transform_json(row['dspy_output']), axis=1)

    # Save the modified CSV file
    df.to_csv(output_csv, index=False)

    print(f"CSV file '{output_csv}' created successfully.")
