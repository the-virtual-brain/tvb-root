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

from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.datatypes.surfaces import FaceSurface, EEGCap
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.sensors import SensorsEEG
from tvb.adapters.visualizers.brain import BrainViewer, DualBrainViewer
from tvb.tests.framework.core.factory import TestFactory
from tvb.tests.framework.datatypes.datatypes_factory import DatatypesFactory


class TestBrainViewer(TransactionalTestCase):
    """
    Unit-tests for BrainViewer.
    """

    EXPECTED_KEYS = ['urlVertices', 'urlNormals', 'urlTriangles', 'urlLines', 'urlRegionMap',
                     'base_activity_url', 'isOneToOneMapping', 'minActivity', 'maxActivity',
                     'noOfMeasurePoints', 'isAdapter']
    EXPECTED_EXTRA_KEYS = ['urlMeasurePointsLabels', 'urlMeasurePoints', 'time_series', 'pageSize', 'shelfObject',
                           'extended_view', 'legendLabels', 'labelsStateVar', 'labelsModes', 'title']


    def transactional_setup_method(self):
        """
        Sets up the environment for running the tests;
        creates a test user, a test project, a connectivity, a cortical surface and a face surface;
        imports a CFF data-set
        """
        self.datatypeFactory = DatatypesFactory()
        self.test_user = self.datatypeFactory.get_user()
        self.test_project = TestFactory.import_default_project(self.test_user)
        self.datatypeFactory.project = self.test_project

        self.connectivity = TestFactory.get_entity(self.test_project, Connectivity())
        assert self.connectivity is not None
        self.face_surface = TestFactory.get_entity(self.test_project, FaceSurface())
        assert self.face_surface is not None
        assert TestFactory.get_entity(self.test_project, EEGCap()) is not None


    def transactional_teardown_method(self):
        """
        Clean-up tests data
        """
        FilesHelper().remove_project_structure(self.test_project.name)
    
    
    def test_launch(self):
        """
        Check that all required keys are present in output from BrainViewer launch.
        """
        time_series = self.datatypeFactory.create_timeseries(self.connectivity)
        viewer = BrainViewer()
        viewer.current_project_id = self.test_project.id
        result = viewer.launch(time_series=time_series)

        for key in TestBrainViewer.EXPECTED_KEYS + TestBrainViewer.EXPECTED_EXTRA_KEYS:
            assert key in result and result[key] is not None
        assert not result['extended_view']

    
    def test_get_required_memory(self):
        """
        Brainviewer should know required memory so expect positive number and not -1.
        """
        time_series = self.datatypeFactory.create_timeseries(self.connectivity)
        assert BrainViewer().get_required_memory_size(time_series) > 0
        
        
    def test_generate_preview(self):
        """
        Check that all required keys are present in preview generate by BrainViewer.
        """
        time_series = self.datatypeFactory.create_timeseries(self.connectivity)
        viewer = BrainViewer()
        result = viewer.generate_preview(time_series, figure_size=(500, 200))
        for key in TestBrainViewer.EXPECTED_KEYS:
            assert key in result and result[key] is not None, key
        
        
    def test_launch_eeg(self):
        """
        Tests successful launch of a BrainEEG and that all required keys are present in returned template dictionary
        """
        sensors = TestFactory.get_entity(self.test_project, SensorsEEG())
        time_series = self.datatypeFactory.create_timeseries(self.connectivity, 'EEG', sensors)
        time_series.configure()
        viewer = DualBrainViewer()
        viewer.current_project_id = self.test_project.id
        result = viewer.launch(time_series)
        for key in TestBrainViewer.EXPECTED_KEYS + TestBrainViewer.EXPECTED_EXTRA_KEYS:
            assert key in result and result[key] is not None
        assert result['extended_view']
