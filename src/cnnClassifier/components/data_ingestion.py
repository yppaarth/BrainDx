import os
import zipfile
import gdown
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:
        """
        Fetch data from the Google Drive URL and download it to the specified path.
        """
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file

            os.makedirs(os.path.dirname(zip_download_dir), exist_ok=True)

            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            # Extract file ID and construct proper download URL
            file_id = dataset_url.split("/")[-2]
            download_url = f"https://drive.google.com/uc?id={file_id}"

            gdown.download(url=download_url, output=zip_download_dir, quiet=False, fuzzy=True)

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            logger.error(f"Error during file download: {e}")
            raise e

    def extract_zip_file(self):
        """
        Extract the downloaded zip file into the specified directory.
        """
        try:
            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)

            logger.info(f"Extracting zip file {self.config.local_data_file} to {unzip_path}")

            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)

            logger.info(f"Extraction completed successfully to {unzip_path}")

        except Exception as e:
            logger.error(f"Error during zip extraction: {e}")
            raise e
