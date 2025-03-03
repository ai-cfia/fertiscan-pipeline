import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv
import logging
from urllib.parse import urlparse
from datetime import datetime

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

    def create_label_directory(self):
        """Create a new label directory with incremental numbering"""
        existing_dirs = [d for d in self.base_path.glob('label_*')]
        next_num = 1 if not existing_dirs else max(
            int(d.name.split('_')[1]) for d in existing_dirs
        ) + 1

        new_dir = self.base_path / f'label_{str(next_num).zfill(3)}'
        new_dir.mkdir(exist_ok=True)
        return new_dir

    def download_image(self, url, save_path):
        """Download an image from URL and save it"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Extract filename from URL or generate one
            filename = os.path.basename(urlparse(url).path)
            if not filename:
                filename = f"img_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

            file_path = save_path / filename

            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return filename
        except Exception as e:
            logger.error(f"Error downloading image from {url}: {str(e)}")
            return None

    def fetch_and_store_data(self):
        """Fetch data from API and store in test_data structure"""
        try:
            # Fetch data from API
            response = requests.get(self.api_endpoint)
            response.raise_for_status()
            data = response.json()

            # Create new label directory
            label_dir = self.create_label_directory()
            logger.info(f"Created directory: {label_dir}")

            # Download images and store data
            #
            # NOTE: This would require a call to another route under the api_endpoint
            for idx, image_url in enumerate(data.get('images', []), 1):
                filename = self.download_image(f"{image_url}", label_dir)
                if filename:
                    logger.info(f"Downloaded image: {filename}")

            # Create expected output JSON
            expected_output = data

            # Save expected output
            with open(label_dir / 'expected_output.json', 'w') as f:
                json.dump(expected_output, f, indent=2)

            logger.info(f"Saved expected_output.json in {label_dir}")

        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")

def main():
    try:
        fetcher = DataFetcher()
        fetcher.fetch_and_store_data()
    except Exception as e:
        logger.error(f"Main execution error: {str(e)}")

if __name__ == "__main__":
    main()
