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
from tvb.adapters.uploaders.abcuploader import ABCUploader
from tvb.core.adapters.exceptions import LaunchException, ParseException
from tvb.adapters.uploaders.gifti.parser import GIFTIParser
from tvb.basic.logger.builder import get_logger
from tvb.datatypes.time_series import TimeSeriesSurface
from tvb.datatypes.surfaces import CorticalSurface


class GIFTITimeSeriesImporter(ABCUploader):
    """
        This importer is responsible for import of a TimeSeries from GIFTI format (XML file)
        and store them in TVB.
    """
    _ui_name = "TimeSeries GIFTI"
    _ui_subsection = "gifti_timeseries_importer"
    _ui_description = "Import TimeSeries from GIFTI"
    
    def get_upload_input_tree(self):
        """
            Take as input a .GII file.
        """
        return [{'name': 'data_file', 'type': 'upload', 'required_type': '.gii',
                 'label': 'Please select file to import (.gii)', 'required': True},
                {'name': 'surface', 'label': 'Brain Surface', 
                 'type': CorticalSurface, 'required': True,
                 'description': 'The Brain Surface used to generate imported TimeSeries.'}
                ]
        
        
    def get_output(self):
        return [TimeSeriesSurface]

    def launch(self, data_file, surface=None):
        """
        Execute import operations:
        """
        if surface is None:
            raise LaunchException("No surface selected. Please initiate upload again and select a brain surface.")
            
        parser = GIFTIParser(self.storage_path, self.operation_id)
        try:
            time_series = parser.parse(data_file)
            ts_data_shape = time_series.read_data_shape()

            if surface.number_of_vertices != ts_data_shape[1]:
                msg = "Imported time series doesn't have values for all surface vertices. Surface has %d vertices " \
                      "while time series has %d values." % (surface.number_of_vertices, ts_data_shape[1])
                raise LaunchException(msg)
            else:
                time_series.surface = surface

            return [time_series]

        except ParseException as excep:
            logger = get_logger(__name__)
            logger.exception(excep)
            raise LaunchException(excep)