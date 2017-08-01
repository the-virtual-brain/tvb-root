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
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""
from tvb.adapters.uploaders.abcuploader import ABCUploader
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.exceptions import ParseException, LaunchException
from tvb.core.entities.storage import transactional
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.graph import ConnectivityMeasure


class ConnectivityMeasureImporter(ABCUploader):
    """
    This imports a searies of conectivity measures from a .mat file
    """
    _ui_name = "ConnectivityMeasure"
    _ui_subsection = "connectivity_measure"
    _ui_description = "Import a searies of connectivity measures from a .mat file"


    def get_upload_input_tree(self):
        """
        Take as input a mat file
        """
        return [{'name': 'data_file', 'type': 'upload', 'required_type': '.mat',
                 'label': 'Connectivity measure file (.mat format)', 'required': True},

                {'name': 'dataset_name', 'type': 'str', 'required': True,
                 'label': 'Matlab dataset name', 'default': 'M',
                 'description': 'Name of the MATLAB dataset where data is stored'},

                {'name': 'connectivity', 'label': 'Large Scale Connectivity',
                 'type': Connectivity, 'required': True, 'datatype': True,
                 'description': 'The Connectivity for which these measurements were made'},
                ]
        
        
    def get_output(self):
        return [ConnectivityMeasure]


    @transactional
    def launch(self, data_file, dataset_name, connectivity):
        """
        Execute import operations:
        """
        try:
            data = self.read_matlab_data(data_file, dataset_name)
            measurement_count, node_count = data.shape

            if node_count != connectivity.number_of_regions:
                raise LaunchException('The measurements are for %s nodes but the selected connectivity'
                                      ' contains %s nodes' % (node_count, connectivity.number_of_regions))

            measures = []
            for i in range(measurement_count):
                measure = ConnectivityMeasure(storage_path=self.storage_path,
                                              connectivity=connectivity, array_data=data[i, :])
                measure.user_tag_2 = "nr.-%d" % (i + 1)
                measure.user_tag_3 = "conn_%d" % node_count
                measures.append(measure)
            return measures
        except ParseException as excep:
            logger = get_logger(__name__)
            logger.exception(excep)
            raise LaunchException(excep)