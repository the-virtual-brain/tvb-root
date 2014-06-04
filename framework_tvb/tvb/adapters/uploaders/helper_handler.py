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
Generic export/import functions.

.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
"""
import os

from tvb.basic.config.settings import TVBSettings as cfg
from tvb.core.utils import get_unique_file_name
from tvb.core.entities.storage import dao
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.datatypes.surfaces import Surface
from tvb.datatypes.time_series import TimeSeries
from tvb.datatypes.connectivity import Connectivity
from tvb.adapters.uploaders.constants import KEY_SURFACE_UID
from tvb.adapters.uploaders.constants import KEY_CONNECTIVITY_UID

NUMPY_TEMP_FOLDER = os.path.join(cfg.TVB_STORAGE, "NUMPY_TMP")


def get_uids_dict(entity_obj):
    """
    Returns a dictionary containing the UIDs for the object graph of the given entity.
    The given entity should be an instance of: Connectivity, Surface, Cortexor or CortexActivity.
    If an obj do not have an uid then its gid will be returned.
    """
    result = dict()
    datatype_obj = dao.get_datatype_by_id(entity_obj.id)

    if isinstance(entity_obj, Connectivity):
        result[KEY_CONNECTIVITY_UID] = __get_uid(datatype_obj)
    if isinstance(entity_obj, Surface):
        result[KEY_SURFACE_UID] = __get_uid(datatype_obj)
    if isinstance(entity_obj, TimeSeries):
        surface_datatype = entity_obj.surface
        connectivity_datatype = entity_obj.connectivity
        result[KEY_SURFACE_UID] = __get_uid(surface_datatype)
        result[KEY_CONNECTIVITY_UID] = __get_uid(connectivity_datatype)
    return result


def get_gifty_file_name(project_id, desired_name):
    """
    Compute non-existent file name, in the TEMP folder of
    the given project.
    Try desired_name, and if already exists, try adding a number.
    """
    if project_id:
        project = dao.get_project_by_id(project_id)
        file_helper = FilesHelper()
        temp_path = file_helper.get_project_folder(project, FilesHelper.TEMP_FOLDER)
        return get_unique_file_name(temp_path, desired_name)[0]
    return get_unique_file_name(cfg.TVB_STORAGE, desired_name)[0]


def __get_uid(datatype_obj):
    """
    @returns: the UID of the given entity.
        If the given object do not have an UID the will be returned its GID.
    """
    if datatype_obj.user_tag_1:
        return str(datatype_obj.user_tag_1)
    else:
        return str(datatype_obj.gid)
    
    
    