import unittest
import os
from pipeline import curl_file, save_image_to_file
from pipeline.label import LabelStorage

class TestDocumentStorage(unittest.TestCase):
    
    def setUp(self):
        self.log_dir_path = './test_logs'
        if not os.path.exists(self.log_dir_path):
            os.mkdir(self.log_dir_path)

        self.label = LabelStorage()
        self.sample_image_path_1 = f'{self.log_dir_path}/label1.png'
        self.sample_image_path_2 = f'{self.log_dir_path}/label2.png'
        self.composite_image_path = f'{self.log_dir_path}/composite_test.png'
        self.composite_document_path = f'{self.log_dir_path}/composite_test.pdf'

        curl_file('https://lesgranulaines.com/wp-content/uploads/2024/01/IMG-5014-copie.webp', self.sample_image_path_1)
        curl_file('https://tlhort.com/cdn/shop/products/10-52-0MAP.jpg', self.sample_image_path_2)


    def test_add_image_from_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            self.label.add_image('fake_path')

    def test_get_document_empty(self):
        with self.assertRaises(ValueError):
            self.label.get_document()    

    def test_add_image(self):
        self.label.add_image(self.sample_image_path_1)
        self.assertEqual(len(self.label.images), 1)

    def test_get_composite_image(self):
        self.label.add_image(self.sample_image_path_1)
        self.label.add_image(self.sample_image_path_2)

        composite_image = self.label.get_document(format='png')
        save_image_to_file(composite_image, self.composite_image_path)
        self.assertTrue(os.path.exists(self.composite_image_path))
    
    def test_get_pdf_document(self):
        self.label.add_image(self.sample_image_path_1)
        self.label.add_image(self.sample_image_path_2)

        doc = self.label.get_document(format='pdf')
        save_image_to_file(doc, self.composite_document_path)
        self.assertTrue(os.path.exists(self.composite_document_path))

    def test_clear(self):
        self.label.add_image(self.sample_image_path_1)
        self.label.add_image(self.sample_image_path_2)
        self.label.clear()

        with self.assertRaises(ValueError):
            self.label.get_document()


    def tearDown(self):
        # Clean up created files after tests
        if os.path.exists(self.log_dir_path):
            for file in os.listdir(self.log_dir_path):
                file_path = os.path.join(self.log_dir_path, file)
                os.remove(file_path)
            os.rmdir(self.log_dir_path)

if __name__ == '__main__':
    unittest.main()
