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

import json
import os
import numpy
import tvb_data
import tvb_data.surfaceData
from tvb.adapters.creators.stimulus_creator import RegionStimulusCreator, SurfaceStimulusCreator
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.datatypes.db.patterns import StimuliRegionIndex, StimuliSurfaceIndex
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.services.operation_service import OperationService
from tvb.datatypes.equations import TemporalApplicableEquation, FiniteSupportEquation
from tvb.datatypes.surfaces import CORTICAL
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.tests.framework.core.factory import TestFactory


# TODO: THESE TESTS WORK BUT WHEN RUNNING THEM, THEY CREATE H5 FILES IN THE SAME FOLDER AS
#  THIS FILE AND THEY REMAIN THERE
class TestStimulusCreator(TransactionalTestCase):

    def transactional_setup_method(self):
        """
        Reset the database before each test.
        """
        self.test_user = TestFactory.create_user('Stim_User')
        self.test_project = TestFactory.create_project(self.test_user, "Stim_Project")

        zip_path = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_66.zip')
        TestFactory.import_zip_connectivity(self.test_user, self.test_project, zip_path)
        self.connectivity = TestFactory.get_entity(self.test_project, ConnectivityIndex)

        cortex = os.path.join(os.path.dirname(tvb_data.surfaceData.__file__), 'cortex_16384.zip')
        self.surface = TestFactory.import_surface_zip(self.test_user, self.test_project, cortex, CORTICAL)

    def transactional_teardown_method(self):
        """
        Remove project folders and clean up database.
        """
        FilesHelper().remove_project_structure(self.test_project.name)

    def test_create_stimulus_region(self):
        weight_array = numpy.zeros(self.connectivity.number_of_regions)
        region_stimulus_creator = RegionStimulusCreator()

        view_model = region_stimulus_creator.get_view_model_class()()
        view_model.connectivity = self.connectivity.gid
        view_model.weight = weight_array
        view_model.temporal = TemporalApplicableEquation()
        view_model.temporal.parameters['a'] = 1.0
        view_model.temporal.parameters['b'] = 2.0

        region_stimulus_index = region_stimulus_creator.launch(view_model)

        assert region_stimulus_index.temporal_equation == 'TemporalApplicableEquation'
        assert json.loads(region_stimulus_index.temporal_parameters) == {'a': 1.0, 'b': 2.0}
        assert region_stimulus_index.fk_connectivity_gid == self.connectivity.gid

    def test_create_stimulus_region_with_operation(self):
        weight_array = numpy.zeros(self.connectivity.number_of_regions)
        region_stimulus_creator = RegionStimulusCreator()

        view_model = region_stimulus_creator.get_view_model_class()()
        view_model.connectivity = self.connectivity.gid
        view_model.weight = weight_array
        view_model.temporal = TemporalApplicableEquation()
        view_model.temporal.parameters['a'] = 1.0
        view_model.temporal.parameters['b'] = 2.0

        OperationService().fire_operation(region_stimulus_creator, self.test_user, self.test_project.id,
                                          view_model=view_model)
        region_stimulus_index = TestFactory.get_entity(self.test_project, StimuliRegionIndex)

        assert region_stimulus_index.temporal_equation == 'TemporalApplicableEquation'
        assert json.loads(region_stimulus_index.temporal_parameters) == {'a': 1.0, 'b': 2.0}
        assert region_stimulus_index.fk_connectivity_gid == self.connectivity.gid

    def test_create_stimulus_surface(self):
        surface_stimulus_creator = SurfaceStimulusCreator()

        view_model = surface_stimulus_creator.get_view_model_class()()
        view_model.surface = self.surface.gid
        view_model.focal_points_triangles = numpy.array([1, 2, 3])
        view_model.spatial = FiniteSupportEquation()
        view_model.spatial_amp = 1.0
        view_model.spatial_sigma = 1.0
        view_model.spatial_offset = 0.0
        view_model.temporal = TemporalApplicableEquation()
        view_model.temporal.parameters['a'] = 1.0
        view_model.temporal.parameters['b'] = 0.0

        surface_stimulus_index = surface_stimulus_creator.launch(view_model)

        assert surface_stimulus_index.spatial_equation == 'FiniteSupportEquation'
        assert surface_stimulus_index.temporal_equation == 'TemporalApplicableEquation'
        assert surface_stimulus_index.fk_surface_gid == self.surface.gid

    def test_create_stimulus_surface_with_operation(self):
        surface_stimulus_creator = SurfaceStimulusCreator()

        view_model = surface_stimulus_creator.get_view_model_class()()
        view_model.surface = self.surface.gid
        view_model.focal_points_triangles = numpy.array([1, 2, 3])
        view_model.spatial = FiniteSupportEquation()
        view_model.spatial_amp = 1.0
        view_model.spatial_sigma = 1.0
        view_model.spatial_offset = 0.0
        view_model.temporal = TemporalApplicableEquation()
        view_model.temporal.parameters['a'] = 1.0
        view_model.temporal.parameters['b'] = 0.0

        OperationService().fire_operation(surface_stimulus_creator, self.test_user, self.test_project.id,
                                          view_model=view_model)
        surface_stimulus_index = TestFactory.get_entity(self.test_project, StimuliSurfaceIndex)

        assert surface_stimulus_index.spatial_equation == 'FiniteSupportEquation'
        assert surface_stimulus_index.temporal_equation == 'TemporalApplicableEquation'
        assert surface_stimulus_index.fk_surface_gid == self.surface.gid
