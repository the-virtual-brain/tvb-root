# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

"""
This module contains the necessary methods for encryption and decryption at import/export.

.. moduleauthor:: Robert Vincze <robert.vincze@codemart.ro>
"""

import os
import pyAesCrypt

from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from tvb.basic.profile import TvbProfile


class ImportExportEncryptionHandler:
    ENCRYPTED_DATA_SUFFIX = '_encrypted'
    ENCRYPTED_PASSWORD_NAME = 'encrypted_password.pem'
    DECRYPTED_DATA_SUFFIX = '_decrypted'
    PUBLIC_KEY_NAME = 'public_key.pem'
    PRIVATE_KEY_NAME = 'private_key.pem'

    def generate_public_private_key_pair(self, key_path):
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )

        public_key = private_key.public_key()
        # The public key will be uploaded to TVB before exporting and the private key will be used for
        # decrypting the data locally

        # Step 2. Convert public key to bytes and save it so ABCUploader can use it
        pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.PKCS1,
        )

        public_key_path = os.path.join(key_path, self.PUBLIC_KEY_NAME)
        with open(public_key_path, 'wb') as f:
            f.write(pem)

        # Step 3. Convert private key to bytes and save it so it can be used later for decryption
        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        private_key_path = os.path.join(key_path, self.PRIVATE_KEY_NAME)
        with open(private_key_path, 'wb') as f:
            f.write(pem)

    def add_encrypted_suffix(self, name):
        return name.replace('.zip', self.ENCRYPTED_DATA_SUFFIX + '.zip')

    def get_path_to_encrypt(self, input_path):
        start_extension = input_path.rfind('.')
        path_to_encrypt = input_path[:start_extension]
        extension = input_path[start_extension:]

        return path_to_encrypt + self.ENCRYPTED_DATA_SUFFIX + extension

    @staticmethod
    def load_public_key(public_key_path):
        with open(public_key_path, "rb") as key_file:
            public_key = serialization.load_pem_public_key(key_file.read(), backend=default_backend())

        return public_key

    @staticmethod
    def encrypt_password(public_key, symmetric_key):

        encrypted_symmetric_key = public_key.encrypt(
            symmetric_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        return encrypted_symmetric_key

    def save_encrypted_password(self, encrypted_password, path_to_encrypted_password):
        save_password_path = os.path.join(path_to_encrypted_password, self.ENCRYPTED_PASSWORD_NAME)
        with open(os.path.join(save_password_path), 'wb') as f:
            f.write(encrypted_password)

        return save_password_path

    def encrypt_and_save_password(self, public_key_path, password, path_to_encrypted_password):
        public_key = self.load_public_key(public_key_path)
        password_bytes = str.encode(password)
        encrypted_password = self.encrypt_password(public_key, password_bytes)
        self.save_encrypted_password(encrypted_password, path_to_encrypted_password)

    def encrypt_data_at_export(self, file_path, password):
        # Encrypt the file(s) using the generated password
        buffer_size = TvbProfile.current.hpc.CRYPT_BUFFER_SIZE

        encrypted_file_path = self.get_path_to_encrypt(file_path)
        pyAesCrypt.encryptFile(file_path, encrypted_file_path, password, buffer_size)

        return encrypted_file_path

    def decrypt_content(self, encrypted_aes_key_path, upload_paths, private_key_path):

        # Get the encrypted password
        with open(encrypted_aes_key_path, 'rb') as f:
            encrypted_password = f.read()

        # Read the private key
        with open(private_key_path, "rb") as key_file:
            private_key = serialization.load_pem_private_key(
                key_file.read(),
                password=None,
                backend=default_backend()
            )

        # Decrypt the password using the private key
        decrypted_password = private_key.decrypt(
            encrypted_password,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        decrypted_password = decrypted_password.decode()

        # Get path to decrypted file
        decrypted_paths = []

        for path in upload_paths:
            decrypted_download_path = path.replace(self.ENCRYPTED_DATA_SUFFIX,
                                                   self.DECRYPTED_DATA_SUFFIX)
            decrypted_paths.append(decrypted_download_path)

            # Use the decrypted password to decrypt the message
            pyAesCrypt.decryptFile(path, decrypted_download_path, decrypted_password,
                                   TvbProfile.current.hpc.CRYPT_BUFFER_SIZE)
        return decrypted_paths

    def extract_encrypted_password_from_list(self, file_list):
        password = None
        password_idx = None

        for idx, file in enumerate(file_list):
            if self.ENCRYPTED_PASSWORD_NAME in file:
                password = file
                password_idx = idx

        del file_list[password_idx]
        return password
