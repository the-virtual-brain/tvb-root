import pyAesCrypt
import os
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding

from tvb.basic.profile import TvbProfile
from tvb.storage.storage_interface import StorageInterface

if __name__ == '__main__':

    # 1. Load private key
    path_to_private_key = '/Users/Robert.Vincze/Documents/private_key.pem'
    with open(path_to_private_key, "rb") as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read(),
            password=None,
            backend=default_backend())

    # 2. Load encrypted password
    path_to_encrypted_password = '/Users/Robert.Vincze/Documents/encrypted_password.pem'
    with open(path_to_encrypted_password, 'rb') as f:
        encrypted_password = f.read()

    # 3. Decrypt the password using the private key
    decrypted_password = private_key.decrypt(
        encrypted_password,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    decrypted_password = decrypted_password.decode()

    path = '/Users/robert.vincze/Documents/2021-09-20_12-27_DataTypeGroup_encrypted'
    storage_interface = StorageInterface()

    # 4. a) If it is just a file, decrypt the file
    if os.path.isfile(path):
        decrypted_path = path.replace(storage_interface.ENCRYPTED_DATA_SUFFIX, storage_interface.DECRYPTED_DATA_SUFFIX)
        pyAesCrypt.decryptFile(path, decrypted_path, decrypted_password, TvbProfile.current.hpc.CRYPT_BUFFER_SIZE)

    # 4. b) If it is a directory (DatatypeGroup), we have to go through each subdirectory and decrypt the content
    if os.path.isdir(path):
        dest_path = path.replace(storage_interface.ENCRYPTED_DATA_SUFFIX, storage_interface.DECRYPTED_DATA_SUFFIX)
        if not os.path.exists(dest_path):
            os.mkdir(dest_path)
        for subdir_path in os.listdir(path):
            src_sub_folder = os.path.join(path, subdir_path)

            if not os.path.isdir(src_sub_folder):
                continue
            dest_sub_folder = os.path.join(dest_path, subdir_path)
            if not os.path.exists(dest_sub_folder):
                os.mkdir(dest_sub_folder)
            for file in os.listdir(src_sub_folder):
                src_file_path = os.path.join(src_sub_folder, file)
                dest_file_path = os.path.join(dest_sub_folder, file)
                dest_file_path = dest_file_path.replace(storage_interface.ENCRYPTED_DATA_SUFFIX, '')
                pyAesCrypt.decryptFile(src_file_path, dest_file_path, decrypted_password,
                                       TvbProfile.current.hpc.CRYPT_BUFFER_SIZE)
