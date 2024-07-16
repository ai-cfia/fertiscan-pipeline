import os
import unittest
from dotenv import load_dotenv
from datetime import datetime
from tests import levenshtein_similarity
from pipeline.form import FertiliserForm
from pipeline import LabelStorage, OCR, GPT, curl_file, analyze

class TestPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        load_dotenv()
        # Set up the required objects
        cls.log_dir_path = './test_logs'
        cls.image_path = f'{cls.log_dir_path}/test_image.jpg'  # Path to your test image
        
        # Ensure the log directory exists
        if not os.path.exists(cls.log_dir_path):
            os.mkdir(cls.log_dir_path)
        
        # Download the test image
        curl_file(url='https://tlhort.com/cdn/shop/products/10-52-0MAP.jpg', path=cls.image_path)

        # Mock environment setup for OCR and GPT
        cls.api_endpoint_ocr = os.getenv('AZURE_API_ENDPOINT')
        cls.api_key_ocr = os.getenv('AZURE_API_KEY')
        cls.api_endpoint_gpt = os.getenv('AZURE_OPENAI_ENDPOINT')
        cls.api_key_gpt = os.getenv('AZURE_OPENAI_KEY')
        cls.api_deployment_gpt = os.getenv('AZURE_OPENAI_DEPLOYMENT')
        
        # Initialize the objects
        cls.label_storage = LabelStorage()
        cls.label_storage.add_image(cls.image_path)
        cls.ocr = OCR(api_endpoint=cls.api_endpoint_ocr, api_key=cls.api_key_ocr)
        cls.gpt = GPT(api_endpoint=cls.api_endpoint_gpt, api_key=cls.api_key_gpt, deployment=cls.api_deployment_gpt)

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
        self.assertIn('25', form.weight_kg)
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
