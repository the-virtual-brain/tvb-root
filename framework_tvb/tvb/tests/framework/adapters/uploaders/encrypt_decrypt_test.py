# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
.. moduleauthor:: Robert Vincze <robert.vincze@codemart.ro>
"""

import tvb_data
import os
import pyAesCrypt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
from tvb.adapters.uploaders.zip_connectivity_importer import ZIPConnectivityImporterModel
from tvb.core.adapters.abcuploader import ABCUploader, ENCRYPTED_PASSWORD_NAME, ENCRYPTED_DATA_SUFFIX, \
    DECRYPTED_DATA_SUFFIX
from tvb.core.services.encryption_handler import EncryptionHandler
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.basic.profile import TvbProfile


class TestEncryptionDecryption(TransactionalTestCase):

    # noinspection PyTypeChecker
    @pytest.mark.parametrize("dir_name, file_name", [('connectivity', 'connectivity_76.zip'),
                                                     ('surfaceData', 'cortex_2x120k.zip'),
                                                     ('projectionMatrix', 'projection_meg_276_surface_16k.npy'),
                                                     ('h5', 'TimeSeriesRegion.h5')])
    def test_encrypt_decrypt(self, dir_name, file_name):

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

        private_key_path = os.path.join(TvbProfile.current.TVB_TEMP_FOLDER, 'private_key.pem')
        with open(private_key_path, 'wb') as f:
            f.write(pem)

        path_to_file = os.path.join(os.path.dirname(tvb_data.__file__), dir_name, file_name)

        # Create model for ABCUploader
        connectivity_model = ZIPConnectivityImporterModel()

        # Generate password
        pass_size = TvbProfile.current.hpc.CRYPT_PASS_SIZE
        password = EncryptionHandler.generate_random_password(pass_size)

        # Encrypt files using an AES symmetric key
        encrypted_file_path = ABCUploader.get_path_to_encrypt(path_to_file)
        buffer_size = TvbProfile.current.hpc.CRYPT_BUFFER_SIZE
        pyAesCrypt.encryptFile(path_to_file, encrypted_file_path, password, buffer_size)

        # Asynchronously encrypt the password used at the previous step for the symmetric encryption
        password = str.encode(password)
        encrypted_password = ABCUploader.encrypt_password(public_key, password)

        # Save encrypted password
        ABCUploader.save_encrypted_password(encrypted_password, TvbProfile.current.TVB_TEMP_FOLDER)
        path_to_encrypted_password = os.path.join(TvbProfile.current.TVB_TEMP_FOLDER, ENCRYPTED_PASSWORD_NAME)

        # Prepare model for decrypting
        connectivity_model.uploaded = encrypted_file_path
        connectivity_model.encrypted_aes_key = path_to_encrypted_password
        TvbProfile.current.UPLOAD_KEY_PATH = os.path.join(TvbProfile.current.TVB_TEMP_FOLDER, 'private_key.pem')

        # Decrypting
        ABCUploader._decrypt_content(connectivity_model, 'uploaded')

        decrypted_file_path = connectivity_model.uploaded.replace(ENCRYPTED_DATA_SUFFIX, DECRYPTED_DATA_SUFFIX)
        with open(path_to_file, 'rb') as f_original:
            with open(decrypted_file_path, 'rb') as f_decrypted:
                while True:
                    original_content_chunk = f_original.read(buffer_size)
                    decrypted_content_chunk = f_decrypted.read(buffer_size)

                    assert original_content_chunk == decrypted_content_chunk,\
                        "Original and Decrypted chunks are not equal, so decryption hasn't been done correctly!"

                    # check if EOF was reached
                    if len(original_content_chunk) < buffer_size:
                        break

        # Clean-up
        os.remove(encrypted_file_path)
        os.remove(decrypted_file_path)
        os.remove(private_key_path)
        os.remove(path_to_encrypted_password)
