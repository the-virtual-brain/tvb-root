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
Root class for export functionality.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""
import os
from datetime import datetime
from abc import ABCMeta, abstractmethod

from tvb.adapters.datatypes.db.mapped_value import DatatypeMeasureIndex
from tvb.adapters.exporters.exceptions import ExportException
from tvb.core.entities.load import load_entity_by_gid
from tvb.core.entities.model.model_datatype import DataTypeGroup
from tvb.core.entities.storage import dao
from tvb.core.neocom import h5
from tvb.core.services.project_service import ProjectService

# List of DataTypes to be excluded from export due to not having a valid export mechanism implemented yet.
EXCLUDED_DATATYPES = ['Cortex', 'CortexActivity', 'CapEEGActivity', 'Cap', 'ValueWrapper', 'SpatioTermporalMask']


class ABCExporter(metaclass=ABCMeta):
    """
    Base class for all data type exporters
    This should provide common functionality for all TVB exporters.
    """

    @abstractmethod
    def get_supported_types(self):
        """
        This method specify what types are accepted by this exporter.
        Method should be implemented by each subclass and return
        an array with the supported types.

        :returns: an array with the supported data types.
        """
        pass

    def get_label(self):
        """
        This method returns a string to be used on the UI controls to initiate export

        :returns: string to be used on UI for starting this export.
                  By default class name is returned
        """
        return self.__class__.__name__

    def accepts(self, data):
        """
        This method specify if the current exporter can export provided data.
        :param data: data to be checked
        :returns: true if this data can be exported by current exporter, false otherwise.
        """
        effective_data_type = self._get_effective_data_type(data)

        # If no data present for export, makes no sense to show exporters
        if effective_data_type is None:
            return False

        # Now we should check if any data type is accepted by current exporter
        # Check if the data type is one of the global exclusions
        if hasattr(effective_data_type, "type") and effective_data_type.type in EXCLUDED_DATATYPES:
            return False

        for supported_type in self.get_supported_types():
            if isinstance(effective_data_type, supported_type):
                return True

        return False

    def _get_effective_data_type(self, data):
        """
        This method returns the data type for the provided data.
        - If current data is a simple data type is returned.
        - If it is an data type group, we return the first element. Only one element is
        necessary since all group elements are the same type.
        """
        # first check if current data is a DataTypeGroup
        if DataTypeGroup.is_data_a_group(data):
            if self.skip_group_datatypes():
                return None

            data_types = ProjectService.get_datatypes_from_datatype_group(data.id)

            if data_types is not None and len(data_types) > 0:
                # Since all objects in a group are the same type it's enough
                return load_entity_by_gid(data_types[0].gid)
            else:
                return None
        else:
            return data

    def skip_group_datatypes(self):
        return False

    @staticmethod
    def prepare_datatypes_for_export(data):
        """
        Method used for exporting data type groups. This method returns a list of all datatype indexes needed to be
        exported and a dictionary where keys are operation folder names and the values are lists containing the paths
        that belong to one particular operation folder.
        """
        all_datatypes = ProjectService.get_all_datatypes_from_data(data)
        first_datatype = all_datatypes[0]

        # We are exporting a group of datatype measures so we need to find the group of time series
        if hasattr(first_datatype, 'fk_source_gid'):
            ts = h5.load_entity_by_gid(first_datatype.fk_source_gid)
            dt_metric_group = dao.get_datatypegroup_by_op_group_id(ts.parent_operation.fk_operation_group)
            datatype_measure_list = ProjectService.get_all_datatypes_from_data(dt_metric_group)
            all_datatypes = datatype_measure_list + all_datatypes
        else:
            ts_group = dao.get_datatype_measure_group_from_ts_from_pse(first_datatype.gid, DatatypeMeasureIndex)
            time_series_list = ProjectService.get_all_datatypes_from_data(ts_group)
            all_datatypes = all_datatypes + time_series_list

        if all_datatypes is None or len(all_datatypes) == 0:
            raise ExportException("Could not export a data type group with no data!")

        op_file_dict = dict()
        for dt in all_datatypes:
            h5_path = h5.path_for_stored_index(dt)
            op_folder = os.path.dirname(h5_path)
            op_file_dict[op_folder] = [h5_path]

            op = dao.get_operation_by_id(dt.fk_from_operation)
            vms = h5.gather_references_of_view_model(op.view_model_gid, os.path.dirname(h5_path), only_view_models=True)
            op_file_dict[op_folder].extend(vms[0])

        return all_datatypes, op_file_dict

    @abstractmethod
    def export(self, data, project, public_key_path, password):
        """
        Actual export method, to be implemented in each sub-class.

        :param data: data type to be exported

        :param project: project that contains data to be exported

        :param public_key_path: path to public key that will be used for encrypting the password by TVB

        :param password: password used for encrypting the files before exporting

        :returns: a tuple with the following elements:

                        1. name of the file to be shown to user
                        2. full path of the export file (available for download)
                        3. boolean which specify if file can be deleted after download
        """
        pass

    @staticmethod
    def get_export_file_name(data, file_extension):
        data_type_name = data.__class__.__name__
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d_%H-%M")

        return "%s_%s%s" % (date_str, data_type_name, file_extension)

    def _get_export_file_name(self, data):
        """
        This method computes the name used to save exported data on user computer
        """
        file_ext = self.get_export_file_extension(data)
        return self.get_export_file_name(data, file_ext)

    @abstractmethod
    def get_export_file_extension(self, data):
        """
        This method computes the extension of the export file
        :param data: data type to be exported
        :returns: the extension of the file to be exported (e.g zip or h5)
        """
        pass
