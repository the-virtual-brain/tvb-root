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

import json
import os
import numpy
import tvb_data
import tvb_data.surfaceData

from tvb.adapters.creators.stimulus_creator import RegionStimulusCreator, SurfaceStimulusCreator
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.datatypes.db.patterns import StimuliRegionIndex, StimuliSurfaceIndex
from tvb.adapters.forms.equation_forms import SpatialEquationsEnum, TemporalEquationsEnum
from tvb.core.services.operation_service import OperationService
from tvb.datatypes.equations import PulseTrain
from tvb.datatypes.surfaces import SurfaceTypesEnum
from tvb.storage.storage_interface import StorageInterface
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.tests.framework.core.factory import TestFactory


class TestStimulusCreator(TransactionalTestCase):

    def transactional_setup_method(self):
        """
        Reset the database before each test.
        """
        self.test_user = TestFactory.create_user('Stim_User')
        self.test_project = TestFactory.create_project(self.test_user, "Stim_Project")
        self.storage_interface = StorageInterface()

        zip_path = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_66.zip')
        TestFactory.import_zip_connectivity(self.test_user, self.test_project, zip_path)
        self.connectivity = TestFactory.get_entity(self.test_project, ConnectivityIndex)

        cortex = os.path.join(os.path.dirname(tvb_data.surfaceData.__file__), 'cortex_16384.zip')
        self.surface = TestFactory.import_surface_zip(self.test_user, self.test_project, cortex,
                                                      SurfaceTypesEnum.CORTICAL_SURFACE)

    def test_create_stimulus_region(self, operation_factory):
        weight_array = numpy.zeros(self.connectivity.number_of_regions)
        region_stimulus_creator = RegionStimulusCreator()

        view_model = region_stimulus_creator.get_view_model_class()()
        view_model.connectivity = self.connectivity.gid
        view_model.weight = weight_array
        view_model.temporal.parameters['T'] = 45.0
        view_model.temporal.parameters['amp'] = 2.0

        operation = operation_factory(test_user=self.test_user, test_project=self.test_project)
        region_stimulus_creator.extract_operation_data(operation)
        region_stimulus_index = region_stimulus_creator.launch(view_model)

        assert region_stimulus_index.temporal_equation == 'PulseTrain'
        assert json.loads(region_stimulus_index.temporal_parameters) == \
               {'T': 45.0, 'amp': 2.0, 'onset': PulseTrain.parameters.default()['onset'],
                'tau': PulseTrain.parameters.default()['tau']}
        assert region_stimulus_index.fk_connectivity_gid == self.connectivity.gid

    def test_create_stimulus_region_with_operation(self):
        weight_array = numpy.zeros(self.connectivity.number_of_regions)
        region_stimulus_creator = RegionStimulusCreator()

        view_model = region_stimulus_creator.get_view_model_class()()
        view_model.connectivity = self.connectivity.gid
        view_model.weight = weight_array
        view_model.temporal.parameters['T'] = 45.0
        view_model.temporal.parameters['amp'] = 2.0

        OperationService().fire_operation(region_stimulus_creator, self.test_user, self.test_project.id,
                                          view_model=view_model)
        region_stimulus_index = TestFactory.get_entity(self.test_project, StimuliRegionIndex)

        assert region_stimulus_index.temporal_equation == 'PulseTrain'
        assert json.loads(region_stimulus_index.temporal_parameters) == \
               {'T': 45.0, 'amp': 2.0, 'onset': PulseTrain.parameters.default()['onset'],
                'tau': PulseTrain.parameters.default()['tau']}
        assert region_stimulus_index.fk_connectivity_gid == self.connectivity.gid

    def test_create_stimulus_surface(self, operation_factory):
        surface_stimulus_creator = SurfaceStimulusCreator()

        view_model = surface_stimulus_creator.get_view_model_class()()
        view_model.surface = self.surface.gid
        view_model.focal_points_triangles = numpy.array([1, 2, 3])
        view_model.spatial = SpatialEquationsEnum.MEXICAN_HAT.instance
        view_model.spatial.parameters['amp_1'] = 0.75
        view_model.spatial.parameters['amp_2'] = 1.25
        view_model.spatial_amp = 1.0
        view_model.spatial_sigma = 1.0
        view_model.spatial_offset = 0.0
        view_model.temporal = TemporalEquationsEnum.SINUSOID.instance
        view_model.temporal.parameters['amp'] = 1.1
        view_model.temporal.parameters['frequency'] = 0.025

        operation = operation_factory(test_user=self.test_user, test_project=self.test_project)
        surface_stimulus_creator.extract_operation_data(operation)
        surface_stimulus_index = surface_stimulus_creator.launch(view_model)

        assert surface_stimulus_index.spatial_equation == 'DoubleGaussian'
        assert surface_stimulus_index.temporal_equation == 'Sinusoid'
        assert surface_stimulus_index.fk_surface_gid == self.surface.gid

        spatial_eq_params = json.loads(surface_stimulus_index.spatial_parameters)
        assert spatial_eq_params['amp_1'] == 0.75
        assert spatial_eq_params['amp_2'] == 1.25

        temporal_eq_params = json.loads(surface_stimulus_index.temporal_parameters)
        assert temporal_eq_params['amp'] == 1.1
        assert temporal_eq_params['frequency'] == 0.025

    def test_create_stimulus_surface_with_operation(self):
        surface_stimulus_creator = SurfaceStimulusCreator()

        view_model = surface_stimulus_creator.get_view_model_class()()
        view_model.surface = self.surface.gid
        view_model.focal_points_triangles = numpy.array([1, 2, 3])
        view_model.spatial = SpatialEquationsEnum.SIGMOID.instance
        view_model.spatial.parameters['radius'] = 5.5
        view_model.spatial.parameters['offset'] = 0.1
        view_model.spatial_amp = 1.0
        view_model.spatial_sigma = 1.0
        view_model.spatial_offset = 0.0
        view_model.temporal = TemporalEquationsEnum.ALPHA.instance
        view_model.temporal.parameters['alpha'] = 15.0
        view_model.temporal.parameters['beta'] = 40.0

        OperationService().fire_operation(surface_stimulus_creator, self.test_user, self.test_project.id,
                                          view_model=view_model)
        surface_stimulus_index = TestFactory.get_entity(self.test_project, StimuliSurfaceIndex)

        assert surface_stimulus_index.spatial_equation == 'Sigmoid'
        assert surface_stimulus_index.temporal_equation == 'Alpha'
        assert surface_stimulus_index.fk_surface_gid == self.surface.gid

        spatial_eq_params = json.loads(surface_stimulus_index.spatial_parameters)
        assert spatial_eq_params['radius'] == 5.5
        assert spatial_eq_params['offset'] == 0.1

        temporal_eq_params = json.loads(surface_stimulus_index.temporal_parameters)
        assert temporal_eq_params['alpha'] == 15.0
        assert temporal_eq_params['beta'] == 40.0
