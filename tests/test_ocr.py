import unittest

from tests import curl_file
from pipeline.components.ocr import OCR
from pipeline.schemas import Settings
from pipeline.components.label import LabelStorage
from tests import levenshtein_similarity

class TestOCR(unittest.TestCase):
    def setUp(self):
        settings = Settings()

        self.ocr = OCR(settings.document_api_endpoint, settings.document_api_key)

        self.sample_image_1 = curl_file('https://scotts.com/dw/image/v2/BGFS_PRD/on/demandware.static/-/Sites-consolidated-master-catalog/default/dw5764839e/images/hi-res/scotts_01291_1_2000x2000.jpg?')
        self.sample_image_2 = curl_file('https://tlhort.com/cdn/shop/products/10-52-0MAP.jpg')

    def test_extract_text(self):
        # Prepare the document bytes
        document_bytes = self.sample_image_2

        # Extract text
        result = self.ocr.extract_text(document_bytes)

        # Extract content from result
        extracted_text = result.as_dict()["content"]

        # Define patterns for each word
        patterns = [
            r'\bAmmonium\b',
            r'\bPhosphate\b',
            r'\bGranular\b',
            r'\b25\b',
            r'\bTerraLink\b',
            r'\bCAUTION\b'
        ]

        # Check if each pattern matches the extracted text
        for pattern in patterns:
            self.assertRegex(text=extracted_text, expected_regex=pattern)

    def test_composite_image_text_extraction(self):
        # Create a DocumentStorage instance and add images
        doc_storage = LabelStorage()
        doc_storage.add_image(self.sample_image_path_1)
        doc_storage.add_image(self.sample_image_path_2)

        # Get the composite image bytes
        composite_image_bytes = doc_storage.get_document()

        # Save the composite image bytes to a file
        # save_bytes_to_image(composite_image_bytes, self.composite_image_path)

        # Extract text from the composite image
        result = self.ocr.extract_text(composite_image_bytes)

        # Extract content from result
        extracted_text = result.content

        result_1 = self.ocr.extract_text(self.sample_image_path_1)
        result_2 = self.ocr.extract_text(self.sample_image_path_2)

        extracted_text_1 = result_1.content
        extracted_text_2 = result_2.content

        distance = levenshtein_similarity(extracted_text, extracted_text_1 + " " + extracted_text_2)

        self.assertGreater(distance, 0.9, "The distance between the merged text and individual extractions is too great!")
