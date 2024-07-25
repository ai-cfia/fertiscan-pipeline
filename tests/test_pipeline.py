import os
import unittest

from tests import curl_file
from dotenv import load_dotenv
from datetime import datetime
from tests import levenshtein_similarity
from pipeline.form import FertiliserForm, Value
from pipeline import LabelStorage, OCR, GPT, analyze

class TestPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        load_dotenv()
        # Set up the required objects
        self.log_dir_path = './test_logs'
        self.image_path = f'{self.log_dir_path}/test_image.jpg'  # Path to your test image
        
        # Ensure the log directory exists
        if not os.path.exists(self.log_dir_path):
            os.mkdir(self.log_dir_path)
        
        # Download the test image
        curl_file(url='https://tlhort.com/cdn/shop/products/10-52-0MAP.jpg', path=self.image_path)
        
        # Mock environment setup for OCR and GPT
        self.api_endpoint_ocr = os.getenv('AZURE_API_ENDPOINT')
        self.api_key_ocr = os.getenv('AZURE_API_KEY')
        self.api_endpoint_gpt = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.api_key_gpt = os.getenv('AZURE_OPENAI_KEY')
        self.api_deployment_gpt = os.getenv('AZURE_OPENAI_DEPLOYMENT')
        
        # Initialize the objects
        self.label_storage = LabelStorage()
        self.label_storage.add_image(self.image_path)
        self.ocr = OCR(api_endpoint=self.api_endpoint_ocr, api_key=self.api_key_ocr)
        self.gpt = GPT(api_endpoint=self.api_endpoint_gpt, api_key=self.api_key_gpt, deployment=self.api_deployment_gpt)

    @classmethod
    def tearDownClass(cls):
        # Clean up the test logs directory
        if os.path.exists(cls.log_dir_path):
            for file in os.listdir(cls.log_dir_path):
                file_path = os.path.join(cls.log_dir_path, file)
                os.remove(file_path)
            os.rmdir(cls.log_dir_path)

    def test_analyze(self):
        # Run the analyze function
        form = analyze(self.label_storage, self.ocr, self.gpt, log_dir_path=self.log_dir_path)
        
        # Perform assertions
        self.assertIsInstance(form, FertiliserForm)
        self.assertIn(Value(value='25', unit='kg'), form.weight)
        self.assertGreater(levenshtein_similarity(form.company_name, "TerraLink"), 0.95)
        self.assertGreater(levenshtein_similarity(form.npk, "10-52-0"), 0.90)

        # Ensure logs are created and then deleted
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        md_log_path = f"{self.log_dir_path}/{now}.md"
        json_log_path = f"{self.log_dir_path}/{now}.json"
        txt_log_path = f"{self.log_dir_path}/{now}.txt"
        
        self.assertFalse(os.path.exists(md_log_path))
        self.assertFalse(os.path.exists(json_log_path))
        self.assertFalse(os.path.exists(txt_log_path))

if __name__ == '__main__':
    unittest.main()
