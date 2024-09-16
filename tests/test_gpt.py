import os
import unittest
import json

from dotenv import load_dotenv
from pipeline.inspection import FertilizerInspection
from pipeline.gpt import GPT
from tests import levenshtein_similarity

class TestLanguageModel(unittest.TestCase):
    def setUp(self):
        load_dotenv()

        gpt_api_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        gpt_api_key = os.getenv("AZURE_OPENAI_KEY")
        gpt_api_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

        self.gpt = GPT(api_endpoint=gpt_api_endpoint, api_key=gpt_api_key, deployment_id=gpt_api_deployment)

        self.prompt = """
        GreenGrow Inc.  
        123 Green Road, Farmville, State, 12345  
        Website: www.greengrow.com  
        Phone: 123-456-7890

        Manufactured by:  
        AgriSupply Co.  
        456 Supply Lane, AgriTown, State, 67890  
        Website: www.agrisupply.com  
        Phone: 987-654-3210

        Product Name: GreenGrow Fertilizer 20-20-20  
        Registration Number: FG123456  
        Lot Number: LOT20240901  
        Weight: 50 kg  
        Density: 1.5 g/cm³  
        Volume: 33.3 L  
        NPK Ratio: 20-20-20

        Guaranteed Analysis (English)  
        - Total Nitrogen (N) 20% Registration No: RN123  
        - Available Phosphate (P₂O₅) 20% Registration No: RN124  
        - Soluble Potash (K₂O) 20% Registration No: RN125

        Analyse Garantie (Français)  
        - Azote total (N) 20% Numéro d'enregistrement: RN126  
        - Phosphate assimilable (P₂O₅) 20% Numéro d'enregistrement: RN127  
        - Potasse soluble (K₂O) 20% Numéro d'enregistrement: RN128

        Ingredients (English):  
        - Organic matter 15% Registration No: RN129

        Ingrédients (Français):  
        - Matière organique 15% Numéro d'enregistrement: RN130

        Cautions:  
        - Keep out of reach of children.  
        - Store in a cool, dry place.

        Précautions:  
        - Garder hors de la portée des enfants.  
        - Conserver dans un endroit frais et sec.

        Instructions (English):  
        - Apply evenly across the field at a rate of 5 kg per hectare.  
        - Water thoroughly after application.

        Instructions (Français):  
        - Appliquer uniformément sur le champ à raison de 5 kg par hectare.  
        - Arroser abondamment après l'application.
        """
    
    def check_json(self, extracted_info):
        file = open('./expected.json')
        expected_json = json.load(file)

        file.close()

        # Check if all keys are present
        for key in expected_json.keys():
            assert key in extracted_info, f"Key '{key}' is missing in the extracted information"

        # Check if the json matches the format
        FertilizerInspection(**expected_json)

        # Check if values match
        for key, expected_value in expected_json.items():
            assert levenshtein_similarity(str(extracted_info[key]), str(expected_value)) > 0.9, f"Value for key '{key}' does not match. Expected '{expected_value}', got '{extracted_info[key]}'"

    def test_generate_form_gpt(self):
        prediction = self.gpt.create_inspection(self.prompt)
        result_json = json.loads(prediction.inspection)
        # print(json.dumps(result_json, indent=2))
        self.check_json(result_json)

if __name__ == '__main__':
    unittest.main()
