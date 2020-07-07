from pathlib import Path
from shutil import copyfile
from utils import download_without_progress, fetch_file_from_zip, fetch_all_from_zip
from requests.exceptions import (HTTPError, InvalidURL, ConnectionError)
from zipfile import (BadZipFile, LargeZipFile)


class Landcover:
    tif = 'E020N60_ProbaV_LC100_epoch2015_global_v2.0.2_forest-type-layer_EPSG-4326.tif'

    def __init__(self):
        self.data_path = Path('./data')
        self.landcover_path = self.data_path / 'landcover'
        self.landcover_path.mkdir(parents=True, exist_ok=True)
        # self.tif = 'E020N60_ProbaV_LC100_epoch2015_global_v2.0.2_forest-type-layer_EPSG-4326.tif'
        print(self.landcover_path)
        self.forest_tiff = self.landcover_path / 'forest.tiff'
        print(self.forest_tiff)

    @staticmethod
    def download_landcover(url):
        """
        download file from url
        :param url: str
        :return: BytesIO
        """
        file = None
        try:
            file = download_without_progress(url)
        except (HTTPError, InvalidURL, ConnectionError, ConnectionError) as e:
            # TODO separate ConnectionError, this type of exception must be written to logs
            print('e2 =', e)
            print(f'cant download zip file from {url}')
            exit(1)
        return file

    @staticmethod
    def unzip_landcover(file, landcover_path):
        """
        extract all files from archive
        :param file: BytesIO or file path
        :param landcover_path: file path
        :return:
        """
        try:
            fetch_all_from_zip(file, landcover_path)
        except (BadZipFile, LargeZipFile) as e:
            print('zip_file exception', e)
            print(f'cant unzip files to {landcover_path}')
            exit(1)
        return

    @staticmethod
    def extract_file(file, source, destination):
        """
        extract specific file from archive
        :param file: BytesIO or file path
        :param source: file name to be extracted
        :param destination: path for extraction
        :return:
        """
        try:
            fetch_file_from_zip(file, source, destination)
        except (BadZipFile, LargeZipFile, Exception) as e:
            print('zip_file exception', e)
            print(f'cant unzip {source} to {destination}')
            exit(1)

    @staticmethod
    def copy_file(source, destination):
        """
        copy file from source to destination
        :param source: file path
        :param destination: file path
        :return:
        """
        try:
            copyfile(source, destination)
        except (OSError, Exception) as e:
            print('copy_file Exception', e)
            print(f'cant copy {source} to {destination}')
            exit(1)
