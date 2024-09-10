import json

from pydantic import BaseModel
from pipeline.inspection import FertilizerInspection


print(json.dumps(BaseModel.model_dump(FertilizerInspection)))
