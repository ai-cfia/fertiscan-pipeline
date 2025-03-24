import unittest

from tests import curl_file
from pipeline.components.label import LabelStorage

class TestDocumentStorage(unittest.TestCase):

    def setUp(self):
        self.label = LabelStorage()
        self.sample_image_1 = curl_file('https://lesgranulaines.com/wp-content/uploads/2024/01/IMG-5014-copie.webp')
        self.sample_image_2 = curl_file('https://tlhort.com/cdn/shop/products/10-52-0MAP.jpg')


    def test_add_image_from_null(self):
        with self.assertRaises(ValueError):
            self.label.add_image(b'')

    def test_get_document_empty(self):
        with self.assertRaises(ValueError):
            self.label.get_document()

    def test_add_image(self):
        self.label.add_image(self.sample_image_1)
        self.assertEqual(len(self.label.images), 1)

    def test_get_composite_image(self):
        self.label.add_image(self.sample_image_1)
        self.label.add_image(self.sample_image_2)

        composite_image = self.label.get_document(format='png')
        self.assertIsInstance(composite_image, bytes)

    def test_get_pdf_document(self):
        self.label.add_image(self.sample_image_1)
        self.label.add_image(self.sample_image_2)

        doc = self.label.get_document(format='pdf')
        self.assertIsInstance(doc, bytes)

    def test_clear(self):
        self.label.add_image(self.sample_image_1)
        self.label.add_image(self.sample_image_2)
        self.label.clear()

        with self.assertRaises(ValueError):
            self.label.get_document()

if __name__ == '__main__':
    unittest.main()
