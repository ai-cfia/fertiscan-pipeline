import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class DataFetcher:
    def __init__(self):
        self.api_endpoint = os.getenv('API_ENDPOINT')
        if not self.api_endpoint:
            raise ValueError("API_ENDPOINT environment variable is not set")

        self.base_path = Path('test_data/labels')
        self.base_path.mkdir(parents=True, exist_ok=True)

    def create_label_directory(self, picture_set_id):
        """Create a new label directory for a picture set"""
        new_dir = self.base_path / f'{picture_set_id}'
        new_dir.mkdir(exist_ok=True)
        return new_dir

    def fetch_picture_set(self, picture_set_id):
        """Fetch a picture set"""
        try:
            response = requests.get(f"{self.api_endpoint}/files/{picture_set_id}", stream=True)
            response.raise_for_status()
            file_ids = response.json().get('file_ids', [])

            save_path = self.create_label_directory(picture_set_id)

            for file_id in file_ids:
                file_url = f"{self.api_endpoint}/files/{picture_set_id}/{file_id}"
                file_response = requests.get(file_url, stream=True)
                file_response.raise_for_status()

                file_name = f"{file_id}.jpg"
                save_path = save_path / file_name

                with open(save_path, 'wb') as f:
                    for chunk in file_response.iter_content(chunk_size=8192):
                        f.write(chunk)

            return len(file_ids)
        except Exception as e:
            logger.error(f"Error downloading images from {picture_set_id}: {str(e)}")
            return None

    def fetch_inspection_data(self, inspection_id):
        """Fetches inspection data, images and output for a given inspection ID and stores them in a label directory"""
        try:
            # Fetch data from API
            response = requests.get(f"{self.api_endpoint}/inspections/{inspection_id}")
            response.raise_for_status()
            data = response.json()

            # Create new label directory
            picture_set_id = data.get('picture_set_id', None)[0]
            label_dir = self.create_label_directory(picture_set_id)
            logger.info(f"Created directory: {label_dir}")

            # Download images and store data
            image_count = self.fetch_picture_set(picture_set_id, label_dir)
            if image_count:
                logger.info(f"Downloaded {image_count} images in: {label_dir}")

            # Create expected output JSON
            expected_output = data

            # Save expected output
            with open(label_dir / 'expected_output.json', 'w') as f:
                json.dump(expected_output, f, indent=2)

            logger.info(f"Saved expected_output.json in {label_dir}")

        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")

    def fetch_inspection_set(self):
        """Fetches a set of inspections"""
        pass

def main():
    try:
        fetcher = DataFetcher()
        fetcher.fetch_inspection_set()
    except Exception as e:
        logger.error(f"Main execution error: {str(e)}")

if __name__ == "__main__":
    main()
