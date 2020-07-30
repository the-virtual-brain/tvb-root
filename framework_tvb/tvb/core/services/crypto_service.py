import os
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend


# noinspection PyTypeChecker
class CryptoService(object):
    _instance = None
    KEY_FOLDER_NAME = 'PUBLIC_KEY'
    KEY_FILE_NAME = 'public_key.pem'

    @staticmethod
    def decrypt_content(view_model, trait_upload_field_name):
        if view_model.encrypted_aes_key is None:
            return

        # Open encrypted file
        upload_path = getattr(view_model, trait_upload_field_name)
        with open(upload_path, 'rb') as f:
            encrypted_content = f.read()

        # Get the encrypted AES symmetric-key
        with open(view_model.encrypted_aes_key, 'rb') as f:
            encrypted_aes_key = f.read()

        # Read the private key
        with open("../../adapters/uploaders/keys/private_key.pem", "rb") as key_file:
            private_key = serialization.load_pem_private_key(
                key_file.read(),
                password=None,
                backend=default_backend()
            )

        # Decrypt the AES symmetric key using the private key
        decrypted_aes_key = private_key.decrypt(
            encrypted_aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        iv = b"a" * 16

        # Use the decrypted AES key to decrypt the message
        cipher = Cipher(algorithms.AES(decrypted_aes_key), modes.CTR(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted_content = decryptor.update(encrypted_content)

        # Save the decrypted content and update attribute on view model
        decrypted_content_path = os.path.join(view_model.uploaded, os.pardir)
        decrypted_content_name = os.path.basename(view_model.uploaded)
        extension_index = decrypted_content_name.rfind('.')
        extension = decrypted_content_name[extension_index:]
        decrypted_content_name = decrypted_content_name[:extension_index]

        decrypted_download_path = os.path.join(decrypted_content_path, decrypted_content_name + extension)
        view_model.__setattr__(trait_upload_field_name, decrypted_download_path)

        with open(decrypted_download_path, 'wb') as f:
            f.write(decrypted_content)
