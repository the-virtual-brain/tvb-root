from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import os


# noinspection PyTypeChecker
class CryptoService(object):
    _instance = None
    KEY_FOLDER_NAME = 'PUBLIC_KEY'
    KEY_FILE_NAME = 'public_key.pem'

    def __init__(self):
        self.private_key = None
        self.public_key = None

    def generate_keys(self, project_path):
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )

        self.public_key = self.private_key.public_key()

        # Downloading the public key
        pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        key_folder = os.path.join(project_path, self.KEY_FOLDER_NAME)
        if not os.path.exists(key_folder):
            os.mkdir(key_folder)

        public_key_path = os.path.join(key_folder, self.KEY_FILE_NAME)
        with open(public_key_path, 'wb') as f:
            f.write(pem)

    def get_key_path(self, project_path):
        key_path = os.path.join(project_path, self.KEY_FOLDER_NAME, self.KEY_FILE_NAME)
        key_name = os.path.basename(key_path)
        return key_path, key_name
