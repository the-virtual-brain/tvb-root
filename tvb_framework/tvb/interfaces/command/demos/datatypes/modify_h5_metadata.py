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
Demo script on how to load a TVB DataType by Id and modify metadata

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import os
from uuid import UUID

from tvb.adapters.datatypes.h5.local_connectivity_h5 import LocalConnectivityH5
from tvb.core.neotraits.h5 import H5File
from tvb.core.utils import date2string


def update_local_connectivity_metadata(file_path):
    with LocalConnectivityH5(file_path) as f:
        f.storage_interface.set_metadata({'Shape': "(16384, 16384)",
                                          'format': "csc",
                                          "dtype": "<f8"},
                                         "/matrix")
        f.storage_interface.set_metadata({'cutoff': 40.0,
                                          'state': "RAW_DATA",
                                          'subject': "John Doe",
                                          'user_tag_1': "srf_16k",
                                          'user_tag_2': "",
                                          'user_tag_3': "",
                                          'user_tag_4': "",
                                          'user_tag_5': "",
                                          'type': "",
                                          'create_date': date2string(datetime.now()),
                                          'visible': True,
                                          'is_nan': False,
                                          'gid': UUID('3e551cbd-47ca-11e4-9f21-3c075431bf56').urn,
                                          'surface': UUID('10467c4f-d487-4186-afa6-d9b1fd8383d8').urn}, )


def update_written_by(folder):
    for root, _, files in os.walk(folder):
        for file_name in files:
            if file_name.endswith(".h5"):
                full_path = os.path.join(root, file_name)
                with H5File(full_path) as f:
                    prev_h5_path = f.written_by.load()
                    new_h5_path = prev_h5_path.replace("tvb.core.entities.file.datatypes", "tvb.adapters.datatypes.h5")
                    f.written_by.store(new_h5_path)


if __name__ == "__main__":
    from tvb.interfaces.command.lab import *

    update_local_connectivity_metadata(
        "/TVB/PROJECTS/Default_Project/7/LocalConnectivity_3e551cbd47ca11e49f213c075431bf56.h5")
    # update_written_by("/WORK/TVB/tvb-root/tvb_data/tvb_data/Default_Project")
