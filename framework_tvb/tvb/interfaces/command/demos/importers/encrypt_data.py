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
Launch an operation from the command line

.. moduleauthor:: Robert Vincze <robert.vincze@codemart.ro>
"""


ENCRYPTED_PASSWORD_NAME = 'encrypted_password.pem'

if __name__ == "__main__":
    from tvb.basic.profile import TvbProfile
    TvbProfile.set_profile(TvbProfile.COMMAND_PROFILE)

import os
import random
import string
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

def read_public_key(path_to_public_key):
    with open(path_to_public_key, "rb") as key_file:
        public_key = serialization.load_pem_public_key(
            key_file.read(),
            backend=default_backend()
        )

    return public_key

def get_path_to_encrypt(input_path):
    start_extension = input_path.rfind('.')
    path_to_encrypt = input_path[:start_extension]
    extension = input_path[start_extension:]

    return path_to_encrypt + '_encrypted' + extension

def generate_password(password_file, pass_size):
    if os.path.exists(password_file):
        return password_file
    chars = string.ascii_letters + string.digits
    password = ''.join(random.choice(chars) for i in range(pass_size))
    with open(password_file, 'w') as fd:
        fd.write(password)
    os.chmod(password_file, 0o440)
    return password_file

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

def save_encrypted_password(encrypted_password, path_to_encrypted_password):

    with open(os.path.join(path_to_encrypted_password, ENCRYPTED_PASSWORD_NAME), 'wb') as f:
        f.write(encrypted_password)