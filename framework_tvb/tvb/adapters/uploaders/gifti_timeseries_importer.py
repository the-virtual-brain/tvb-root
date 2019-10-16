# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
"""

import json
import uuid
from tvb.basic.filters.chain import FilterChain
from tvb.adapters.uploaders.abcuploader import ABCUploader, ABCUploaderForm
from tvb.core.adapters.exceptions import LaunchException, ParseException
from tvb.adapters.uploaders.gifti.parser import GIFTIParser
from tvb.basic.logger.builder import get_logger
from tvb.core.entities.file.datatypes.time_series import TimeSeriesSurfaceH5
from tvb.core.entities.model.datatypes.surface import SurfaceIndex
from tvb.core.entities.model.datatypes.time_series import TimeSeriesSurfaceIndex
from tvb.core.neotraits._forms import UploadField, DataTypeSelectField
from tvb.core.neotraits.db import prepare_array_shape_meta
from tvb.interfaces.neocom._h5loader import DirLoader


class GIFTITimeSeriesImporterForm(ABCUploaderForm):

    def __init__(self, prefix='', project_id=None):
        super(GIFTITimeSeriesImporterForm, self).__init__(prefix, project_id)

        self.data_file = UploadField('.gii', self, name='data_file', required=True,
                                     label='Please select file to import (.gii)')
        surface_conditions = FilterChain(fields=[FilterChain.datatype + '.surface_type'], operations=["=="],
                                         values=['Cortical Surface'])
        self.surface = DataTypeSelectField(SurfaceIndex, self, name='surface', required=True,
                                           conditions=surface_conditions, label='Brain Surface',
                                           doc='The Brain Surface used to generate imported TimeSeries.')


class GIFTITimeSeriesImporter(ABCUploader):
    """
        This importer is responsible for import of a TimeSeries from GIFTI format (XML file)
        and store them in TVB.
    """
    _ui_name = "TimeSeries GIFTI"
    _ui_subsection = "gifti_timeseries_importer"
    _ui_description = "Import TimeSeries from GIFTI"

    form = None

    def get_input_tree(self): return None

    def get_upload_input_tree(self): return None

    def get_form(self):
        if self.form is None:
            return GIFTITimeSeriesImporterForm
        return self.form

    def set_form(self, form):
        self.form = form

    def get_output(self):
        return [TimeSeriesSurfaceIndex]

    def launch(self, data_file, surface=None):
        """
        Execute import operations:
        """
        if surface is None:
            raise LaunchException("No surface selected. Please initiate upload again and select a brain surface.")

        parser = GIFTIParser(self.storage_path, self.operation_id)
        try:
            partial_time_series, gifti_data_arrays = parser.parse(data_file)

            ts_idx = TimeSeriesSurfaceIndex()

            loader = DirLoader(self.storage_path)
            ts_h5_path = loader.path_for(TimeSeriesSurfaceH5, ts_idx.gid)

            ts_h5 = TimeSeriesSurfaceH5(ts_h5_path)
            # todo : make sure that write_time_slice is not required here
            for data_array in gifti_data_arrays:
                ts_h5.write_data_slice([data_array.data])

            ts_h5.store(partial_time_series, scalars_only=True, store_references=False)
            ts_h5.gid.store(uuid.UUID(ts_idx.gid))

            ts_data_shape = ts_h5.read_data_shape()
            if surface.number_of_vertices != ts_data_shape[1]:
                msg = "Imported time series doesn't have values for all surface vertices. Surface has %d vertices " \
                      "while time series has %d values." % (surface.number_of_vertices, ts_data_shape[1])
                raise LaunchException(msg)
            else:
                ts_h5.surface.store(uuid.UUID(surface.gid))
                ts_idx.surface = surface
                ts_idx.surface_id = surface.id
            ts_h5.close()

            ts_idx.sample_period_unit = partial_time_series.sample_period_unit
            ts_idx.sample_period = partial_time_series.sample_period
            ts_idx.sample_rate = partial_time_series.sample_rate
            ts_idx.labels_ordering = json.dumps(partial_time_series.labels_ordering)
            ts_idx.data_ndim = len(ts_data_shape)
            ts_idx.data_length_1d, ts_idx.data_length_2d, ts_idx.data_length_3d, ts_idx.data_length_4d = prepare_array_shape_meta(ts_data_shape)

            return [ts_idx]

        except ParseException as excep:
            logger = get_logger(__name__)
            logger.exception(excep)
            raise LaunchException(excep)