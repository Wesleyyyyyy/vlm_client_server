import asyncio
import httpx
import requests
from yarl import URL
import json
import os
from llm_authenticator import EdnaAuthenticator, get_authenticator

EDGAR_API_URL = "http://localhost:8000"

class EdgarDataConnection:
    def __init__(
        self,
        authenticator: EdnaAuthenticator,
        base_url: str = "http://api.edna.cps.cit.tum.de",
        parallel_downloads: int = 4,
    ):
        self._base_url = URL(base_url)
        self._authenticator = authenticator
        self._client = httpx.Client(base_url=base_url)
        # For position of progress bars and acting as a semaphore
        self._queue = asyncio.Queue()
        for i in range(parallel_downloads):
            self._queue.put_nowait(i + 1)

    def _get(self, *path, **kwargs):
        url = URL("/").joinpath(*path)
        response = self._client.get(str(url), **kwargs)
        response.raise_for_status()
        return response

    def vlm_query(self):
        folder_path = "./input"  # local directory, ready to be uploaded
        files_list = get_png_files(folder_path)
        response = requests.post('http://localhost:8000/chat', files=files_list, stream=True)

        for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode('utf-8'))
                text = data
                clear_lines()
                print(text)

    def upload(self):
        folder_path = "./input"  # local directory, ready to be uploaded
        files_list = get_xml_files(folder_path)
        response = requests.post('http://localhost:8000/upload', files=files_list, stream=True)

        if response.status_code == 200:
            print("Files uploaded successfully")
        else:
            print(f"Error uploading files. Status code: {response.status_code}")
            print(response.text)

def get_connection():
    authenticator = get_authenticator()
    return EdgarDataConnection(authenticator=authenticator, base_url=EDGAR_API_URL)


def get_png_files(folder_path):
    """
    return a list of the image files for request
    """
    files_list = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.png'):
                file_path = os.path.join(root, file)
                file_tuple = ('files', (file, open(file_path, 'rb'), 'image/png'))
                files_list.append(file_tuple)
    return files_list


def get_xml_files(folder_path):
    """
    return a list of the XML files for request
    """
    files_list = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.xml'):
                file_path = os.path.join(root, file)
                file_tuple = ('files', (file, open(file_path, 'rb'), 'application/xml'))
                files_list.append(file_tuple)
    return files_list

def clear_lines():
    print('\033[2J')