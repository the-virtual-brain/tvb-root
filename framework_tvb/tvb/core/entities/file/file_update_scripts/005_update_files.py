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
# CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
# Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
Upgrade script from H5 version 4 to version 5 (for tvb release 2.0)

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Robert Vincze <robert.vincze@codemart.ro>
"""
import os
import sys
from tvb.core.neocom.h5 import REGISTRY
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.core.entities.file.exceptions import IncompatibleFileManagerException
from tvb.core.entities.file.hdf5_storage_manager import HDF5StorageManager
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.neotraits._h5core import H5File

LOGGER = get_logger(__name__)
FIELD_SURFACE_MAPPING = "has_surface_mapping"
FIELD_VOLUME_MAPPING = "has_volume_mapping"


def _lowercase_first_character(string):
    """
    One-line function which converts the first character of a string to lowercase and
    handles empty strings and None values
    """
    return string[:1].lower() + string[1:] if string else ''


def update(input_file):
    """
    :param input_file: the file that needs to be converted to a newer file storage version.
    """

    if not os.path.isfile(input_file):
        raise IncompatibleFileManagerException("Not yet implemented update for file %s" % input_file)

    folder, file_name = os.path.split(input_file)
    storage_manager = HDF5StorageManager(folder, file_name)

    root_metadata = storage_manager.get_metadata()

    if DataTypeMetaData.KEY_CLASS_NAME not in root_metadata:
        raise IncompatibleFileManagerException("File %s received for upgrading 4 -> 5 is not valid, due to missing "
                                               "metadata: %s" % (input_file, DataTypeMetaData.KEY_CLASS_NAME))

    lowercase_keys = []
    for key, value in root_metadata.items():
        root_metadata[key] = str(value, 'utf-8')
        lowercase_keys.append(_lowercase_first_character(key))
        storage_manager.remove_metadata(key)

    root_metadata = dict(zip(lowercase_keys, list(root_metadata.values())))
    root_metadata[TvbProfile.current.version.DATA_VERSION_ATTRIBUTE] = TvbProfile.current.version.DATA_VERSION
    class_name = root_metadata["type"]

    root_metadata[DataTypeMetaData.KEY_DATE] = root_metadata[DataTypeMetaData.KEY_DATE].replace('datetime:', '')
    root_metadata[DataTypeMetaData.KEY_DATE] = root_metadata[DataTypeMetaData.KEY_DATE].replace(':', '-')
    root_metadata[DataTypeMetaData.KEY_DATE] = root_metadata[DataTypeMetaData.KEY_DATE].replace(' ', ',')

    datatype_class = getattr(sys.modules[root_metadata["module"]],
                             root_metadata["type"])
    h5_class = REGISTRY.get_h5file_for_datatype(datatype_class)
    root_metadata[H5File.KEY_WRITTEN_BY] = h5_class.__module__ + '.' + h5_class.__name__

    root_metadata['operation_tag'] = ''
    root_metadata['user_tag_1'] = ''

    root_metadata['gid'] = "urn:uuid:" + root_metadata['gid']

    root_metadata.pop("type")
    root_metadata.pop("module")
    root_metadata.pop('data_version')

    if "TimeSeries" in class_name:
        root_metadata.pop(FIELD_SURFACE_MAPPING)
        root_metadata.pop(FIELD_VOLUME_MAPPING)

        root_metadata['nr_dimensions'] = int(root_metadata['nr_dimensions'])
        root_metadata['sample_period'] = float(root_metadata['sample_period'])
        root_metadata['start_time'] = float(root_metadata['start_time'])

        root_metadata["sample_period_unit"] = root_metadata["sample_period_unit"].replace("\"", '')
        root_metadata[DataTypeMetaData.KEY_TITLE] = root_metadata[DataTypeMetaData.KEY_TITLE].replace("\"", '')
        root_metadata['region_mapping'] = "urn:uuid:" + root_metadata['region_mapping']
        root_metadata['connectivity'] = "urn:uuid:" + root_metadata['connectivity']

        root_metadata.pop('length_1d')
        root_metadata.pop('length_2d')
        root_metadata.pop('length_3d')
        root_metadata.pop('length_4d')
    elif "Connectivity" in class_name:
        root_metadata['number_of_connections'] = int(root_metadata['number_of_connections'])
        root_metadata['number_of_regions'] = int(root_metadata['number_of_regions'])

        if root_metadata['undirected'] == "0":
            root_metadata['undirected'] = "bool:False"
        else:
            root_metadata['undirected'] = "bool:True"

        if root_metadata['saved_selection'] == 'null':
            root_metadata['saved_selection'] = '[]'

    storage_manager.set_metadata(root_metadata)
