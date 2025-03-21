import re
from typing import List, Optional

import phonenumbers
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
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

class Organization(BaseModel):
    """
    Represents an organization such as a manufacturer, company, or any entity 
    associated with a fertilizer.
    """
    name: Optional[str] = Field(None, description="The name of the organization.")
    address: Optional[str] = Field(None, description="The address of the organization.")
    website: Optional[str] = Field(None, description="The website of the organization, ensuring 'www.' prefix is added..")
    phone_number: Optional[str] = Field(None, description="The primary phone number of the organization. Return only one.")

    @field_validator("phone_number", mode="before")
    def validate_phone_number(cls, v):
        if v is None:
            return None
        try:
            phone_number = phonenumbers.parse(v, "CA", _check_region=False)
            if not phonenumbers.is_valid_number(phone_number):
                return None
            phone_number = phonenumbers.format_number(
                phone_number, phonenumbers.PhoneNumberFormat.E164
            )
            return phone_number

        except phonenumbers.phonenumberutil.NumberParseException:
            return None
        
    @field_validator("website", mode="before")
    def website_lowercase(cls, v):
        if v is not None:
            if not v.startswith("www."):
                v = "www." + v
            return v.lower()
        return v


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

class RegistrationNumber(BaseModel):
    identifier: Optional[str] = Field(..., description="A string composed of 7-digit number followed by an uppercase letter.")
    type: Optional[str] = Field(None, description="Type of registration number, either 'ingredient_component' or 'fertilizer_product'.")

    @field_validator("identifier", mode="before")
    def check_registration_number_format(cls, v):
        if v is not None:
            pattern = r"^\d{7}[A-Z]$"
            if re.match(pattern, v):
                return v
        return None

    @field_validator("type", mode="before")
    def check_registration_number_type(cls, v):
        if v not in ["ingredient_component", "fertilizer_product"]:
            return None
        return v

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
    organizations: List[Organization] = []
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
        "organizations",
        "weight",
        mode="before",
    )
    def replace_none_with_empty_list(cls, v):
        if v is None:
            v = []
        return v
