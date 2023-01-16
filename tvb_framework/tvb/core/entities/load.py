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
Higher level entity loading.
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""
import importlib
import uuid

from tvb.basic.logger.builder import get_logger
from tvb.core.entities.storage import dao

LOGGER = get_logger(__name__)


def get_class_by_name(fqname):
    """
    get_class_by_name("package.module.class")
    is equivalent to from package.module import class
    """
    try:
        modulename, classname = fqname.rsplit('.', 1)
        module = importlib.import_module(modulename)
        return getattr(module, classname)
    except (AttributeError, ValueError) as e:
        raise ImportError(str(e))


def load_entity_by_gid(data_gid):
    """
    Load a generic DataType, specified by GID.
    """
    if isinstance(data_gid, uuid.UUID):
        data_gid = data_gid.hex
    datatype = dao.get_datatype_by_gid(data_gid)
    return datatype


def get_filtered_datatypes(project_id, data_type_cls, filters=None, page_size=50):
    """
    Return all dataTypes that match a given name and some filters.
    :param data_type_cls: either a fully qualified class name or a class object
    """
    if isinstance(data_type_cls, str):
        data_type_cls = get_class_by_name(data_type_cls)
    LOGGER.debug('Filtering:' + str(data_type_cls))
    return dao.get_values_of_datatype(project_id, data_type_cls, filters, page_size)


def try_get_last_datatype(project_id, data_type_cls, filters=None):
    """
    Retrieve the last dataTypes matching a filter inside the current project.
    :return: instance of data_type_cls or None
    """
    result, count = get_filtered_datatypes(project_id, data_type_cls, filters=filters, page_size=1)
    if count == 0:
        return None
    dt = load_entity_by_gid(result[0][2])
    return dt
