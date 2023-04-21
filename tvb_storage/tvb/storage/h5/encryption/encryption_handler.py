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
.. moduleauthor:: Paula Popa <paula.popa@codemart.ro>
"""

import os
import random
import shutil
import string
import uuid
import pyAesCrypt

from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile

LOGGER = get_logger(__name__)


class EncryptionHandler(object):
    encrypted_dir_name_regex = 'cipher_{}'
    encrypted_suffix = '.aes'

    def __init__(self, dir_gid):
        """
        :param dir_gid: the GID to use for the encrypted directory name
        """
        if isinstance(dir_gid, uuid.UUID):
            dir_gid = dir_gid.hex
        self.current_enc_dirname = self._prepare_encrypted_dir_name(dir_gid)
        self.enc_data_dir = TvbProfile.current.hpc.CRYPT_DATADIR
        self.pass_dir = TvbProfile.current.hpc.CRYPT_PASSDIR
        self.buffer_size = TvbProfile.current.hpc.CRYPT_BUFFER_SIZE
        self._generate_dirs()

    def _prepare_encrypted_dir_name(self, dir_gid):
        return self.encrypted_dir_name_regex.format(dir_gid)

    def _generate_dirs(self):
        for dirr in [self.enc_data_dir, self.pass_dir]:
            if not os.path.isdir(dirr):
                os.makedirs(dirr)

    def cleanup_encryption_handler(self):
        for path in [self.get_encrypted_dir(), self.get_password_file()]:
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)

    @staticmethod
    def generate_random_password():
        pass_size = TvbProfile.current.hpc.CRYPT_PASS_SIZE
        chars = string.ascii_letters + string.digits
        password = ''.join(random.choice(chars) for i in range(pass_size))
        return password

    def _generate_password(self):
        password_file = self.get_password_file()
        if os.path.exists(password_file):
            return password_file
        password = self.generate_random_password()
        with open(password_file, 'w') as fd:
            fd.write(password)
        os.chmod(password_file, TvbProfile.current.ACCESS_MODE_TVB_FILES)
        return password_file

    @staticmethod
    def _read_password(password_file):
        with open(password_file, 'rb') as password_file:
            password = password_file.read().decode()
        return password

    def get_encrypted_dir(self):
        return os.path.join(self.enc_data_dir, self.current_enc_dirname)

    def get_password_file(self):
        return os.path.join(self.pass_dir, self.current_enc_dirname)

    def _prepare_encryption_dir(self, subdir):
        encrypted_dir = self.get_encrypted_dir()

        if subdir:
            encrypted_dir = os.path.join(encrypted_dir, subdir)

        if not os.path.isdir(encrypted_dir):
            os.makedirs(encrypted_dir)
        return encrypted_dir

    def encrypt_inputs(self, files_to_encrypt, subdir=None):
        # type: (list, str) -> list
        """
        Receive a list with all files to encrypt.
        Prepare encryption directory and encrypt each file.
        Return a list with all files from the encrypted directory.
        """
        encryption_dir = self._prepare_encryption_dir(subdir)
        password_file = self._generate_password()

        password = self._read_password(password_file)

        for file_to_encrypt in files_to_encrypt:
            encrypted_file = os.path.join(encryption_dir, os.path.basename(file_to_encrypt) + self.encrypted_suffix)
            pyAesCrypt.encryptFile(file_to_encrypt, encrypted_file, password, self.buffer_size)

        encrypted_files = [os.path.join(encryption_dir, enc_file) for enc_file in os.listdir(encryption_dir)]
        return encrypted_files

    def _determine_plain_filename(self, dir, encrypted_file):
        # type: (str, str) -> str
        return os.path.join(dir, os.path.basename(encrypted_file).replace(self.encrypted_suffix, ''))

    def decrypt_results_to_dir(self, dir, from_subdir=None):
        # type: (str, str) -> list
        """
        Having an already encrypted directory, decrypt all files,
        then move plain files to the location specified by :param dir
        """
        password = self._read_password(self.get_password_file())

        if not os.path.isdir(dir):
            os.makedirs(dir)

        encrypted_dir = self.get_encrypted_dir()

        if from_subdir:
            encrypted_dir = os.path.join(encrypted_dir, from_subdir)

        plain_files = []
        for encrypted_file in os.listdir(encrypted_dir):
            plain_file = self._determine_plain_filename(dir, encrypted_file)
            encrypted_file_full = os.path.join(encrypted_dir, encrypted_file)
            try:
                pyAesCrypt.decryptFile(encrypted_file_full, plain_file, password, self.buffer_size)
                plain_files.append(plain_file)
            except ValueError:
                LOGGER.info('Could not decrypt file {}'.format(encrypted_file))

        return plain_files

    def decrypt_files_to_dir(self, files, dir):
        # type: (list, str) -> list
        """
        Given a list of encrypted files, decrypt them,
        then move plain files to the location specified by :param dir
        """
        password = self._read_password(self.get_password_file())

        if not os.path.isdir(dir):
            os.makedirs(dir)

        plain_files = []
        for encrypted_file in files:
            plain_file = self._determine_plain_filename(dir, encrypted_file)
            try:
                pyAesCrypt.decryptFile(encrypted_file, plain_file, password, self.buffer_size)
                plain_files.append(plain_file)
            except ValueError:
                LOGGER.info('Could not decrypt file {}'.format(encrypted_file))

        return plain_files
