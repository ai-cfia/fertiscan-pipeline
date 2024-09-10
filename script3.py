import re
from pydantic import BaseModel, ValidationError, field_validator
from typing import Optional

# Helper function to extract first number
def extract_first_number(string: str) -> Optional[float]:
    if string is not None:
        match = re.search(r"\d+(\.\d+)?", string)
        if match:
            return float(match.group())
    return None

# NutrientValue model with updated field type
class NutrientValue(BaseModel):
    nutrient: str
    value: float | None = None
    unit: Optional[str] = None

    @field_validator("value", mode="before")
    def convert_value(cls, v):
        if type(v) in (int, float):
            return float(v)
        if type(v) is str:
            return extract_first_number(v)
        

# Test cases
def test_nutrient_value():
    # Case 1: Integer value
    nv1 = NutrientValue(nutrient="Protein", value=15, unit="g")
    print(nv1.model_dump_json())
    assert nv1.value == 15, f"Expected 15, but got {nv1.value} (type: {type(nv1.value)})"

    # Case 2: Float value
    nv2 = NutrientValue(nutrient="Fat", value=10.5, unit="g")
    assert nv2.value == 10.5, f"Expected 10.5, but got {nv2.value} (type: {type(nv2.value)})"

    # Case 3: String containing a number
    nv3 = NutrientValue(nutrient="Carbohydrates", value="20g", unit="g")
    assert nv3.value == 20.0, f"Expected 20.0, but got {nv3.value} (type: {type(nv3.value)})"

    # Case 4: String without a number
    nv4 = NutrientValue(nutrient="Fiber", value="N/A", unit="g")
    assert nv4.value is None, f"Expected None, but got {nv4.value} (type: {type(nv4.value)})"

    # Case 5: None value
    nv5 = NutrientValue(nutrient="Sugar", value=None, unit="g")
    assert nv5.value is None, f"Expected None, but got {nv5.value} (type: {type(nv5.value)})"

    # Case 6: Boolean value (should result in ValidationError as it's unsupported input)
    try:
        NutrientValue(nutrient="Vitamins", value=True, unit="mg")
    except ValidationError as e:
        print(f"Expected ValidationError, got: {e}")

    # Case 7: Boolean value (should result in ValidationError as it's unsupported input)
    try:
        nv6 = NutrientValue(nutrient="Vitamins", value=False, unit="mg")
        print(nv6)
    except ValidationError as e:
        print(f"Expected ValidationError for boolean input, got: {e}")

if __name__ == "__main__":
    try:
        test_nutrient_value()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Assertion failed: {e}")
