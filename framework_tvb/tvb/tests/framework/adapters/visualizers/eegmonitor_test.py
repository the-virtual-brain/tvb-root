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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""
import json
import os
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
import tvb_data.sensors as sensors_dataset
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.adapters.visualizers.eeg_monitor import EegMonitor
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.sensors import SensorsEEG
from tvb.tests.framework.core.factory import TestFactory
from tvb.tests.framework.datatypes.datatypes_factory import DatatypesFactory


class TestEEGMonitor(TransactionalTestCase):
    """
    Unit-tests for EEG Viewer.
    """
    def transactional_setup_method(self):
        """
        Sets up the environment for running the tests;
        creates a test user, a test project, a connectivity and a surface;
        imports a CFF data-set
        """
        self.datatypeFactory = DatatypesFactory()
        self.test_project = self.datatypeFactory.get_project()
        self.test_user = self.datatypeFactory.get_user()
        
        TestFactory.import_cff(test_user=self.test_user, test_project=self.test_project)
        self.connectivity = TestFactory.get_entity(self.test_project, Connectivity())
        assert self.connectivity is not None

                
    def transactional_teardown_method(self):
        """
        Clean-up tests data
        """
        FilesHelper().remove_project_structure(self.test_project.name)
    
    
    def test_launch(self):
        """
        Check that all required keys are present in output from BrainViewer launch.
        """
        zip_path = os.path.join(os.path.dirname(sensors_dataset.__file__),  'eeg_unitvector_62.txt.bz2')
        
        TestFactory.import_sensors(self.test_user, self.test_project, zip_path, 'EEG Sensors')
        sensors = TestFactory.get_entity(self.test_project, SensorsEEG())
        time_series = self.datatypeFactory.create_timeseries(self.connectivity, 'EEG', sensors)
        viewer = EegMonitor()
        result = viewer.launch(time_series)
        expected_keys = ['tsNames', 'groupedLabels', 'tsModes', 'tsStateVars', 'longestChannelLength',
                         'label_x', 'entities', 'page_size', 'number_of_visible_points',
                         'extended_view', 'initialSelection', 'ag_settings', 'ag_settings']

        for key in expected_keys:
            assert key in result, "key not found %s" % key

        expected_ag_settings = ['channelsPerSet', 'channelLabels', 'noOfChannels', 'translationStep',
                                'normalizedSteps', 'nan_value_found', 'baseURLS', 'pageSize',
                                'nrOfPages', 'timeSetPaths', 'totalLength', 'number_of_visible_points',
                                'extended_view', 'measurePointsSelectionGIDs']

        ag_settings = json.loads(result['ag_settings'])

        for key in expected_ag_settings:
            assert key in ag_settings, "ag_settings should have the key %s" % key

