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
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""

import os
import numpy
from abc import ABCMeta
from scipy import io as scipy_io
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.core.adapters.abcadapter import ABCSynchronous, ABCAdapterForm
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.neotraits.forms import StrField, TraitUploadField
from tvb.core.neotraits.uploader_view_model import UploaderViewModel
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend


class ABCUploaderForm(ABCAdapterForm):

    def __init__(self, prefix='', project_id=None):
        super(ABCUploaderForm, self).__init__(prefix, project_id)
        self.subject_field = StrField(UploaderViewModel.data_subject, self, name='Data_Subject')
        # Show Encryption field only when the current TVB installation is capable of decryption
        supports_encrypted_files = (TvbProfile.current.UPLOAD_KEY_PATH is not None
                                    and os.path.exists(TvbProfile.current.UPLOAD_KEY_PATH))
        if supports_encrypted_files:
            self.encrypted_aes_key = TraitUploadField(UploaderViewModel.encrypted_aes_key, '.pem', self,
                                                      name='encrypted_aes_key')
        self.temporary_files = []

    @staticmethod
    def get_required_datatype():
        return None

    @staticmethod
    def get_filters():
        return None

    @staticmethod
    def get_input_name():
        return None


class ABCUploader(ABCSynchronous, metaclass=ABCMeta):
    """
    Base class of the uploaders
    """
    LOGGER = get_logger(__name__)

    def __init__(self):
        super(ABCUploader, self).__init__()

    def _prelaunch(self, operation, uid=None, available_disk_space=0, view_model=None, **kwargs):
        """
        Before going with the usual prelaunch, get from input parameters the 'subject'.
        """

        self.meta_data.update({DataTypeMetaData.KEY_SUBJECT: view_model.data_subject})
        self.generic_attributes.subject = view_model.data_subject

        trait_upload_field_names = list(self.get_form_class().get_upload_information().keys())
        for upload_field_name in trait_upload_field_names:
            self._decrypt_content(view_model, upload_field_name)

        return ABCSynchronous._prelaunch(self, operation, uid, available_disk_space, view_model, **kwargs)

    def _decrypt_content(self, view_model, trait_upload_field_name):
        if view_model.encrypted_aes_key is None:
            return

        if TvbProfile.current.UPLOAD_KEY_PATH is None or not os.path.exists(TvbProfile.current.UPLOAD_KEY_PATH):
            raise LaunchException("We can not process Encrypted files at this moment, "
                                  "due to missing PK for decryption! Please contact the administrator!")

        # Open encrypted file
        upload_path = getattr(view_model, trait_upload_field_name)
        with open(upload_path, 'rb') as f:
            encrypted_content = f.read()

        # Get the encrypted AES symmetric-key
        with open(view_model.encrypted_aes_key, 'rb') as f:
            extended_encrypted_aes_key = f.read()

        encrypted_aes_key = extended_encrypted_aes_key[:self.ENCRYPTED_AES_KEY_SIZE]
        iv = extended_encrypted_aes_key[self.ENCRYPTED_AES_KEY_SIZE:]

        # Read the private key
        with open(TvbProfile.current.UPLOAD_KEY_PATH, "rb") as key_file:
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

        # Use the decrypted AES key to decrypt the message
        cipher = Cipher(algorithms.AES(decrypted_aes_key), modes.CTR(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted_content = decryptor.update(encrypted_content)

        # Save the decrypted content and update attribute on view model
        decrypted_content_path = os.path.join(upload_path, os.pardir)
        decrypted_content_name = os.path.basename(upload_path)
        extension_index = decrypted_content_name.rfind('.')
        extension = decrypted_content_name[extension_index:]
        decrypted_content_name = decrypted_content_name[:extension_index]

        decrypted_download_path = os.path.join(decrypted_content_path, decrypted_content_name + extension)
        view_model.__setattr__(trait_upload_field_name, decrypted_download_path)

        with open(decrypted_download_path, 'wb') as f:
            f.write(decrypted_content)

    def get_required_memory_size(self, view_model):
        """
        Return the required memory to run this algorithm.
        As it is an upload algorithm and we do not have information about data, we can not approximate this.
        """
        return -1

    def get_required_disk_size(self, view_model):
        """
        As it is an upload algorithm and we do not have information about data, we can not approximate this.
        """
        return 0

    @staticmethod
    def read_list_data(full_path, dimensions=None, dtype=numpy.float64, skiprows=0, usecols=None):
        """
        Read numpy.array from a text file or a npy/npz file.
        """
        try:
            if full_path.endswith(".npy") or full_path.endswith(".npz"):
                array_result = numpy.load(full_path)
            else:
                array_result = numpy.loadtxt(full_path, dtype=dtype, skiprows=skiprows, usecols=usecols)
            if dimensions:
                return array_result.reshape(dimensions)
            return array_result
        except ValueError as exc:
            file_ending = os.path.split(full_path)[1]
            exc.args = (exc.args[0] + " In file: " + file_ending,)
            raise

    @staticmethod
    def read_matlab_data(path, matlab_data_name=None):
        """
        Read array from matlab file.
        """
        try:
            matlab_data = scipy_io.matlab.loadmat(path)
        except NotImplementedError:
            ABCUploader.LOGGER.error("Could not read Matlab content from: " + path)
            ABCUploader.LOGGER.error("Matlab files must be saved in a format <= -V7...")
            raise

        try:
            return matlab_data[matlab_data_name]
        except KeyError:
            def double__(n):
                n = str(n)
                return n.startswith('__') and n.endswith('__')

            available = [s for s in matlab_data if not double__(s)]
            raise KeyError("Could not find dataset named %s. Available datasets: %s" % (matlab_data_name, available))

    @staticmethod
    def get_upload_information():
        return NotImplementedError
