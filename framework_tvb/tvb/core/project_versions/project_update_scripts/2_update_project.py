# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
#
#
# CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#

"""
ProjectionRegion DataType has been removed, and ProjectionSurfaces got a new required field.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""
import json

import os
from tvb.basic.logger.builder import get_logger
from tvb.core.entities.file.hdf5_storage_manager import HDF5StorageManager
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.datatypes import projections_data

LOGGER = get_logger(__name__)
FIELD_PROJECTION_TYPE = "Projection_type"


def update(project_path):
    """
    Remove all ProjectionRegion entities before import
    """

    for root, _, files in os.walk(project_path):
        for file_name in files:

            if "ProjectionRegion" in file_name:
                LOGGER.info("Removing %s from %s" % (file_name, root))
                os.remove(os.path.join(root, file_name))

            if "ProjectionSurface" in file_name:
                LOGGER.info("Updating ProjectionSurface %s from %s" % (file_name, root))

                storage_manager = HDF5StorageManager(root, file_name)
                root_metadata = storage_manager.get_metadata()

                if FIELD_PROJECTION_TYPE not in root_metadata:
                    class_name = root_metadata[DataTypeMetaData.KEY_CLASS_NAME]

                    projection_type = projections_data.EEG_POLYMORPHIC_IDENTITY
                    if "SEEG" in class_name:
                        projection_type = projections_data.SEEG_POLYMORPHIC_IDENTITY
                    elif "MEG" in class_name:
                        projection_type = projections_data.MEG_POLYMORPHIC_IDENTITY

                    root_metadata[FIELD_PROJECTION_TYPE] = json.dumps(projection_type)
                    LOGGER.info("Setting %s = %s" % (FIELD_PROJECTION_TYPE, projection_type))
                    storage_manager.set_metadata(root_metadata)
