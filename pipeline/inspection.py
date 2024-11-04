import re
from enum import Enum
from typing import Annotated, List, Optional

import phonenumbers
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


class npkError(ValueError):
    pass


def extract_first_number(string: str) -> Optional[str]:
    if string is not None:
        match = re.search(r"\d+(\.\d+)?", string)
        if match:
            return match.group()
    return None


class NutrientValue(BaseModel):
    nutrient: str
    value: Optional[float] = None
    unit: Optional[str] = None

    @field_validator("value", mode="before", check_fields=False)
    def convert_value(cls, v):
        if isinstance(v, bool):
            return None
        elif isinstance(v, (int, float)):
            return str(v)
        elif isinstance(v, (str)):
            return extract_first_number(v)
        return None


class Value(BaseModel):
    value: Optional[float]
    unit: Optional[str]

    @field_validator("value", mode="before", check_fields=False)
    def convert_value(cls, v):
        if isinstance(v, bool):
            return None
        elif isinstance(v, (int, float)):
            return str(v)
        elif isinstance(v, (str)):
            return extract_first_number(v)
        return None

# class syntax

class RegistrationNumberType(str, Enum):
    """
    Represents the type of registration number for fertilizers.

    - INGREDIENT: Refers to a registration number associated with a specific ingredient in the ingredient list of a fertilizer.
    - FERTILIZER: Refers to the unique registration number assigned to the fertilizer product itself.
    """
    INGREDIENT = "ingredient_component"
    FERTILIZER = "fertilizer_product"

class RegistrationNumber(BaseModel):
    identifier: str = Field(..., description="A string composed of 7-digit number followed by an uppercase letter.")
    type: Optional[RegistrationNumberType] = None

    @field_validator("identifier", mode="before")
    def check_registration_number_format(cls, v):
        if v is not None:
            pattern = r"^\d{7}[A-Z]$"
            if re.match(pattern, v):
                return v
        return None

class GuaranteedAnalysis(BaseModel):
    title: Optional[str] = None
    nutrients: List[NutrientValue] = []
    is_minimal: bool | None = None

    @field_validator(
        "nutrients",
        mode="before",
    )
    def replace_none_with_empty_list(cls, v):
        if v is None:
            v = []
        return v

    @model_validator(mode="after")
    def set_is_minimal(self):
        pattern = r"\bminim\w*\b"
        if self.title:
            self.is_minimal = re.search(pattern, self.title, re.IGNORECASE) is not None
        return self


class Specification(BaseModel):
    humidity: Optional[float] = Field(..., alias="humidity")
    ph: Optional[float] = Field(..., alias="ph")
    solubility: Optional[float]

    @field_validator("humidity", "ph", "solubility", mode="before", check_fields=False)
    def convert_specification_values(cls, v):
        if isinstance(v, bool):
            return None
        elif isinstance(v, (int, float)):
            return str(v)
        elif isinstance(v, (str)):
            return extract_first_number(v)
        return None


class FertilizerInspection(BaseModel):
    company_name: Optional[str] = None
    company_address: Optional[str] = None
    company_website: Annotated[str | None, StringConstraints(to_lower=True)] = Field(
        None,
        description="Return the distributor's website, ensuring 'www.' prefix is added.",
    )
    company_phone_number: Optional[str] = Field(
        None, description="The distributor's primary phone number. Return only one."
    )
    manufacturer_name: Optional[str] = None
    manufacturer_address: Optional[str] = None
    manufacturer_website: Annotated[str | None, StringConstraints(to_lower=True)] = (
        Field(
            None,
            description="Return the manufacturer's website, ensuring 'www.' prefix is added.",
        )
    )
    manufacturer_phone_number: Optional[str] = Field(
        None, description="The manufacturer's primary phone number. Return only one."
    )
    fertiliser_name: Optional[str] = None
    registration_number: List[RegistrationNumber] = []
    lot_number: Optional[str] = None
    weight: List[Value] = []
    density: Optional[Value] = None
    volume: Optional[Value] = None
    npk: Optional[str] = Field(None)
    guaranteed_analysis_en: Optional[GuaranteedAnalysis] = None
    guaranteed_analysis_fr: Optional[GuaranteedAnalysis] = None
    cautions_en: List[str] = None
    cautions_fr: List[str] = None
    instructions_en: List[str] = []
    instructions_fr: List[str] = []
    ingredients_en: List[NutrientValue] = []
    ingredients_fr: List[NutrientValue] = []
    model_config = ConfigDict(populate_by_name=True)

    @field_validator("npk", mode="before")
    def validate_npk(cls, v):
        if v is not None:
            pattern = re.compile(r"^\d+(\.\d+)?-\d+(\.\d+)?-\d+(\.\d+)?$")
            if not pattern.match(v):
                return None
        return v

    @field_validator(
        "cautions_en",
        "cautions_fr",
        "instructions_en",
        "instructions_fr",
        "ingredients_en",
        "ingredients_fr",
        "registration_number",
        "weight",
        mode="before",
    )
    def replace_none_with_empty_list(cls, v):
        if v is None or v == 0:
            v = []
        return v

    @field_validator("company_phone_number", "manufacturer_phone_number", mode="before")
    def check_phone_number_format(cls, v):
        if v is None:
            return

        try:
            phone_number = phonenumbers.parse(v, "CA")
            if not phonenumbers.is_valid_number(phone_number):
                return
            phone_number = phonenumbers.format_number(
                phone_number, phonenumbers.PhoneNumberFormat.E164
            )
            return phone_number

        except phonenumbers.phonenumberutil.NumberParseException:
            return
