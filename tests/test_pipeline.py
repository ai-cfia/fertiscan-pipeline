import os
import shutil
import tempfile
import unittest
from datetime import datetime

from dotenv import load_dotenv

from pipeline import GPT, OCR, LabelStorage, analyze, analyze_document
from pipeline.inspection import FertilizerInspection, Value
from tests import curl_file, levenshtein_similarity


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        load_dotenv()
        # Set up the required objects
        self.log_dir_path = "./test_logs"
        self.image_path = (
            f"{self.log_dir_path}/test_image.jpg"  # Path to your test image
        )

        # Ensure the log directory exists
        if not os.path.exists(self.log_dir_path):
            os.mkdir(self.log_dir_path)

        # Download the test image
        curl_file(
            url="https://tlhort.com/cdn/shop/products/10-52-0MAP.jpg",
            path=self.image_path,
        )

        # Mock environment setup for OCR and GPT
        self.api_endpoint_ocr = os.getenv("AZURE_API_ENDPOINT")
        self.api_key_ocr = os.getenv("AZURE_API_KEY")
        self.api_endpoint_gpt = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key_gpt = os.getenv("AZURE_OPENAI_KEY")
        self.api_deployment_gpt = os.getenv("AZURE_OPENAI_DEPLOYMENT")

        # Initialize the objects
        self.label_storage = LabelStorage()
        self.label_storage.add_image(self.image_path)
        self.ocr = OCR(api_endpoint=self.api_endpoint_ocr, api_key=self.api_key_ocr)
        self.gpt = GPT(
            api_endpoint=self.api_endpoint_gpt,
            api_key=self.api_key_gpt,
            deployment_id=self.api_deployment_gpt,
        )

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
        inspection = analyze(
            self.label_storage, self.ocr, self.gpt, log_dir_path=self.log_dir_path
        )

        # Perform assertions
        self.assertIsInstance(inspection, FertilizerInspection, inspection)
        self.assertIn(Value(value="25", unit="kg"), inspection.weight, inspection)
        manufacturer_or_company = inspection.organizations[0].name
        self.assertIsNotNone(manufacturer_or_company, inspection)
        self.assertGreater(
            levenshtein_similarity(
                manufacturer_or_company, "TerraLink Horticulture Inc."
            ),
            0.95,
            inspection,
        )
        self.assertGreater(
            levenshtein_similarity(inspection.npk, "10-52-0"), 0.90, inspection
        )

        # Ensure logs are created and then deleted
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        md_log_path = f"{self.log_dir_path}/{now}.md"
        json_log_path = f"{self.log_dir_path}/{now}.json"
        txt_log_path = f"{self.log_dir_path}/{now}.txt"

        self.assertFalse(os.path.exists(md_log_path))
        self.assertFalse(os.path.exists(json_log_path))
        self.assertFalse(os.path.exists(txt_log_path))

    def test_analyze_document(self):
        # Run the analyze function
        self.setUpClass()
        inspection = analyze_document(
            self.label_storage.get_document(), self.ocr, self.gpt, log_dir_path=self.log_dir_path
        )

        # Perform assertions
        self.assertIsInstance(inspection, FertilizerInspection, inspection)
        self.assertIn(Value(value="25", unit="kg"), inspection.weight, inspection)
        manufacturer_or_company = (
            inspection.manufacturer_name or inspection.company_name
        )
        self.assertIsNotNone(manufacturer_or_company, inspection)
        self.assertGreater(
            levenshtein_similarity(
                manufacturer_or_company, "TerraLink Horticulture Inc."
            ),
            0.95,
            inspection,
        )
        self.assertGreater(
            levenshtein_similarity(inspection.npk, "10-52-0"), 0.90, inspection
        )

        # Ensure logs are created and then deleted
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        md_log_path = f"{self.log_dir_path}/{now}.md"
        json_log_path = f"{self.log_dir_path}/{now}.json"
        txt_log_path = f"{self.log_dir_path}/{now}.txt"

        self.assertFalse(os.path.exists(md_log_path))
        self.assertFalse(os.path.exists(json_log_path))
        self.assertFalse(os.path.exists(txt_log_path))


class TestInspectionAnnotatedFields(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load environment variables
        load_dotenv()

        # Mock environment setup for OCR and GPT
        cls.api_endpoint_ocr = os.getenv("AZURE_API_ENDPOINT")
        cls.api_key_ocr = os.getenv("AZURE_API_KEY")
        cls.api_endpoint_gpt = os.getenv("AZURE_OPENAI_ENDPOINT")
        cls.api_key_gpt = os.getenv("AZURE_OPENAI_KEY")
        cls.api_deployment_gpt = os.getenv("AZURE_OPENAI_DEPLOYMENT")

        # Initialize OCR and GPT objects (real instances)
        cls.ocr = OCR(api_endpoint=cls.api_endpoint_ocr, api_key=cls.api_key_ocr)
        cls.gpt = GPT(
            api_endpoint=cls.api_endpoint_gpt,
            api_key=cls.api_key_gpt,
            deployment_id=cls.api_deployment_gpt,
        )

        # Supported image extensions
        cls.image_extensions = [".jpg", ".png"]

        # Create a temporary directory for image copies
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        # Clean up the temporary directory after tests are done
        shutil.rmtree(cls.temp_dir)

    def copy_images_to_temp_dir(self, label_folder):
        copied_files = []
        for file_name in os.listdir(label_folder):
            _, ext = os.path.splitext(file_name)
            if ext.lower() in self.image_extensions:
                image_path = os.path.join(label_folder, file_name)
                temp_image_path = os.path.join(self.temp_dir, file_name)
                shutil.copy(image_path, temp_image_path)
                copied_files.append(temp_image_path)
        return copied_files

    def add_images_to_storage(self, image_paths, label_storage):
        for image_path in image_paths:
            label_storage.add_image(image_path)

    def test_label_008_phone_number_inspection(self):
        label_folder = "test_data/labels/label_008"
        label_storage = LabelStorage()

        # Copy images to temporary directory and add to storage
        image_paths = self.copy_images_to_temp_dir(label_folder)
        self.add_images_to_storage(image_paths, label_storage)

        # Run the analyze function
        inspection = analyze(label_storage, self.ocr, self.gpt)

        # Assertions
        self.assertIn("+18003279462", str(inspection.organizations), inspection.organizations)
        # self.assertIsNone(inspection.manufacturer_phone_number)

    def test_label_024_phone_number_inspection(self):
        label_folder = "test_data/labels/label_024"
        label_storage = LabelStorage()

        # Copy images to temporary directory and add to storage
        image_paths = self.copy_images_to_temp_dir(label_folder)
        self.add_images_to_storage(image_paths, label_storage)

        # Run the analyze function
        inspection = analyze(label_storage, self.ocr, self.gpt)

        # Assertions
        self.assertIn("+14506556147", str(inspection.organizations))
        # self.assertIsNone(inspection.manufacturer_phone_number)

    def test_label_001_website_inspection(self):
        label_folder = "test_data/labels/label_001"
        label_storage = LabelStorage()

        # Copy images to temporary directory and add to storage
        image_paths = self.copy_images_to_temp_dir(label_folder)
        self.add_images_to_storage(image_paths, label_storage)

        # Run the analyze function
        inspection = analyze(label_storage, self.ocr, self.gpt)

        # Assertions for website fields
        self.assertIn("www.soil-aid.com", str(inspection.organizations))

    def test_label_006_website_inspection(self):
        label_folder = "test_data/labels/label_006"
        label_storage = LabelStorage()

        # Copy images to temporary directory and add to storage
        image_paths = self.copy_images_to_temp_dir(label_folder)
        self.add_images_to_storage(image_paths, label_storage)

        # Run the analyze function
        inspection = analyze(label_storage, self.ocr, self.gpt)

        # Assertions for website fields
        self.assertIn("www.activeagriscience.com", str(inspection.organizations))

    def test_label_034_website_inspection(self):
        label_folder = "test_data/labels/label_034"
        label_storage = LabelStorage()

        # Copy images to temporary directory and add to storage
        image_paths = self.copy_images_to_temp_dir(label_folder)
        self.add_images_to_storage(image_paths, label_storage)

        # Run the analyze function
        inspection = analyze(label_storage, self.ocr, self.gpt)

        # Assertions for website fields
        self.assertIn("www.advancednutrients.com/growersupport", str(inspection.organizations))


if __name__ == "__main__":
    unittest.main()
