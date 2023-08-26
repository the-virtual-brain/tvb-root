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
.. moduleauthor:: Robert Vincze <robert.vincze@codemart.ro>
"""

import os
import pyAesCrypt
import pytest
from tvb.datasets import TVBZenodoDataset

import tempfile
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from tvb.adapters.uploaders.zip_connectivity_importer import ZIPConnectivityImporterModel
from tvb.basic.profile import TvbProfile
from tvb.storage.h5.encryption.encryption_handler import EncryptionHandler
from tvb.storage.storage_interface import StorageInterface
from tvb.tests.framework.adapters.exporters.exporters_test import TestExporters
from tvb.tests.framework.core.base_testcase import TransactionalTestCase


class TestEncryptionDecryption(TransactionalTestCase):

    dataset = TVBZenodoDataset()
    # noinspection PyTypeChecker

    @pytest.mark.parametrize(" file_name", [('connectivity_76.zip'),
                                                     ( 'cortex_2x120k.zip'),
                                                     ( 'projection_meg_276_surface_16k.npy'),
                                                     ( 'TimeSeriesRegion.h5')])
    def test_encrypt_decrypt(self,  file_name):
        handler = StorageInterface.get_import_export_encryption_handler()


        # Generate a private key and public key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )

        public_key = private_key.public_key()

        # Convert private key to bytes and save it so ABCUploader can use it
        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        private_key_path = os.path.join(TvbProfile.current.TVB_TEMP_FOLDER, handler.PRIVATE_KEY_NAME)
        with open(private_key_path, 'wb') as f:
            f.write(pem)

        path_to_file = self.dataset.fetch_data(file_name)


        # Create model for ABCUploader
        connectivity_model = ZIPConnectivityImporterModel()

        # Generate password
        password = EncryptionHandler.generate_random_password()

        # Encrypt files using an AES symmetric key
        encrypted_file_path = handler.get_path_to_encrypt(path_to_file)
        tail = os.path.split(encrypted_file_path)[1]
        encrypted_file_path = os.path.join(tempfile.gettempdir(), tail)

        buffer_size = TvbProfile.current.hpc.CRYPT_BUFFER_SIZE
        pyAesCrypt.encryptFile(path_to_file, encrypted_file_path, password, buffer_size)

        # Asynchronously encrypt the password used at the previous step for the symmetric encryption
        password = str.encode(password)
        encrypted_password = handler.encrypt_password(public_key, password)

        # Save encrypted password
        handler.save_encrypted_password(encrypted_password, TvbProfile.current.TVB_TEMP_FOLDER)
        path_to_encrypted_password = os.path.join(TvbProfile.current.TVB_TEMP_FOLDER,
                                                  handler.ENCRYPTED_PASSWORD_NAME)

        # Prepare model for decrypting
        connectivity_model.uploaded = encrypted_file_path
        connectivity_model.encrypted_aes_key = path_to_encrypted_password
        TvbProfile.current.UPLOAD_KEY_PATH = os.path.join(TvbProfile.current.TVB_TEMP_FOLDER,
                                                          handler.PRIVATE_KEY_NAME)

        # Decrypting
        decrypted_download_path = handler.decrypt_content(connectivity_model.encrypted_aes_key,
                                                          [connectivity_model.uploaded],
                                                          TvbProfile.current.UPLOAD_KEY_PATH)[0]
        connectivity_model.uploaded = decrypted_download_path

        storage_interface = StorageInterface()
        decrypted_file_path = connectivity_model.uploaded.replace(handler.ENCRYPTED_DATA_SUFFIX,
                                                                  handler.DECRYPTED_DATA_SUFFIX)
        TestExporters.compare_files(path_to_file, decrypted_file_path)

        # Clean-up
        storage_interface.remove_files([encrypted_file_path, decrypted_file_path,
                                        private_key_path, path_to_encrypted_password])
