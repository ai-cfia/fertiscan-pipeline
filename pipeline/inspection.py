import re
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

class npkError(ValueError):
    pass

def extract_first_number(string: str) -> Optional[str]:
    if string is not None:
        match = re.search(r'\d+(\.\d+)?', string)
        if match:
            return match.group()
    return None

class NutrientValue(BaseModel):
    nutrient: str
    value: Optional[float] = None
    unit: Optional[str] = None

    @field_validator('value', mode='before', check_fields=False)
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

    @field_validator('value', mode='before', check_fields=False)
    def convert_value(cls, v):
        if isinstance(v, bool):
            return None
        elif isinstance(v, (int, float)):
            return str(v)
        elif isinstance(v, (str)):
            return extract_first_number(v)
        return None

class Specification(BaseModel):
    humidity: Optional[float] = Field(..., alias='humidity')
    ph: Optional[float] = Field(..., alias='ph')
    solubility: Optional[float]

    @field_validator('humidity', 'ph', 'solubility', mode='before', check_fields=False)
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
    company_website: Optional[str] = None
    company_phone_number: Optional[str] = None
    manufacturer_name: Optional[str] = None
    manufacturer_address: Optional[str] = None
    manufacturer_website: Optional[str] = None
    manufacturer_phone_number: Optional[str] = None
    fertiliser_name: Optional[str] = None
    registration_number: Optional[str] = None
    lot_number: Optional[str] = None
    weight: List[Value] = []
    density: Optional[Value] = None
    volume: Optional[Value] = None
    npk: Optional[str] = Field(None)
    guaranteed_analysis: List[NutrientValue] = []
    warranty: Optional[str] = None
    cautions_en: List[str] = None
    instructions_en: List[str] = []
    micronutrients_en: List[NutrientValue] = []
    ingredients_en: List[NutrientValue] = []
    specifications_en: List[Specification] = []
    first_aid_en: List[str] = None
    cautions_fr: List[str] = None
    instructions_fr: List[str] = []
    micronutrients_fr: List[NutrientValue] = []
    ingredients_fr: List[NutrientValue] = []
    specifications_fr: List[Specification] = []
    first_aid_fr: List[str] = None
    
    @field_validator('npk', mode='before')
    def validate_npk(cls, v):
        if v is not None:
            pattern = re.compile(r'^\d+(\.\d+)?-\d+(\.\d+)?-\d+(\.\d+)?$')
            if not pattern.match(v):
                return None
        return v

    @model_validator(mode='before')
    def replace_none_with_empty_list(cls, values):
        fields_to_check = [
            'cautions_en', 'first_aid_en', 'cautions_fr', 'first_aid_fr',
            'instructions_en', 'micronutrients_en', 'ingredients_en',
            'specifications_en', 'instructions_fr',
            'micronutrients_fr', 'ingredients_fr',
            'specifications_fr', 'guaranteed_analysis'
        ]
        for field in fields_to_check:
            if values.get(field) is None:
                values[field] = []
        return values

    class Config:
        populate_by_name = True
