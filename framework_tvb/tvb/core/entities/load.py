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
Higher level entity loading.
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""
import importlib
from tvb.basic.logger.builder import get_logger
from tvb.core.entities.file.exceptions import FileVersioningException
from tvb.core.entities.file.files_update_manager import FilesUpdateManager
from tvb.core.entities.storage import dao

LOGGER = get_logger(__name__)

def get_class_by_name(fqname):
    '''
    get_class_by_name("package.module.class")
    is equivalent to from package.module import class
    '''
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
    datatype = dao.get_datatype_by_gid(data_gid)
    # TODO
    # from tvb.core.traits.types_mapped import MappedType

    # if isinstance(datatype, MappedType):
    #     datatype_path = datatype.get_storage_file_path()
    #     files_update_manager = FilesUpdateManager()
    #     if not files_update_manager.is_file_up_to_date(datatype_path):
    #         datatype.invalid = True
    #         dao.store_entity(datatype)
    #         raise FileVersioningException("Encountered DataType with an incompatible storage or data version. "
    #                                       "The DataType was marked as invalid.")
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

