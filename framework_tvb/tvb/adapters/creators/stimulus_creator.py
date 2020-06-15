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
.. Ionel Ortelecan <ionel.ortelecan@codemart.ro>
"""

import uuid
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.datatypes.db.surface import SurfaceIndex
from tvb.adapters.datatypes.db.patterns import StimuliRegionIndex, StimuliSurfaceIndex
from tvb.adapters.simulator.equation_forms import get_form_for_equation
from tvb.adapters.simulator.subforms_mapping import get_ui_name_to_equation_dict, GAUSSIAN_EQUATION
from tvb.adapters.simulator.subforms_mapping import DOUBLE_GAUSSIAN_EQUATION, SIGMOID_EQUATION
from tvb.basic.neotraits.api import Attr
from tvb.core.adapters.abcadapter import ABCSynchronous, ABCAdapterForm
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import DataTypeSelectField, FormField, SimpleStrField
from tvb.core.neotraits.forms import TraitDataTypeSelectField, SelectField
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr, Str
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.equations import Sigmoid, PulseTrain, TemporalApplicableEquation, FiniteSupportEquation
from tvb.datatypes.patterns import StimuliSurface, StimuliRegion
from tvb.datatypes.surfaces import CorticalSurface, CORTICAL


class StimulusSurfaceSelectorForm(ABCAdapterForm):

    def __init__(self, project_id=None):
        super(StimulusSurfaceSelectorForm, self).__init__(project_id=project_id)
        self.surface_stimulus = DataTypeSelectField(StimuliSurfaceIndex, self, name='existentEntitiesSelect',
                                                    label='Load Surface Stimulus')
        self.display_name = SimpleStrField(self, name='display_name', label='Display name')

    def get_rendering_dict(self):
        return {'adapter_form': self, 'legend': 'Loaded stimulus'}


class SurfaceStimulusCreatorModel(ViewModel, StimuliSurface):
    spatial = Attr(field_type=FiniteSupportEquation, label="Spatial Equation", default=Sigmoid())
    temporal = Attr(field_type=TemporalApplicableEquation, label="Temporal Equation", default=PulseTrain())

    surface = DataTypeGidAttr(
        linked_datatype=CorticalSurface,
        label=StimuliSurface.surface.label
    )


class SurfaceStimulusCreatorForm(ABCAdapterForm):
    NAME_SPATIAL_PARAMS_DIV = 'spatial_params'
    NAME_TEMPORAL_PARAMS_DIV = 'temporal_params'
    default_spatial = Sigmoid
    default_temporal = PulseTrain
    choices_temporal = get_ui_name_to_equation_dict()
    choices_spatial = {GAUSSIAN_EQUATION: choices_temporal.get(GAUSSIAN_EQUATION),
                       DOUBLE_GAUSSIAN_EQUATION: choices_temporal.get(DOUBLE_GAUSSIAN_EQUATION),
                       SIGMOID_EQUATION: choices_temporal.get(SIGMOID_EQUATION)}

    def __init__(self, project_id=None):
        super(SurfaceStimulusCreatorForm, self).__init__(project_id=project_id)

        self.surface = TraitDataTypeSelectField(SurfaceStimulusCreatorModel.surface, self, name='surface',
                                                conditions=self.get_filters())
        self.spatial = SelectField(SurfaceStimulusCreatorModel.spatial, self, name='spatial',
                                   choices=self.choices_spatial,
                                   subform=get_form_for_equation(self.default_spatial))
        self.temporal = SelectField(SurfaceStimulusCreatorModel.temporal, self, name='temporal',
                                    choices=self.choices_temporal,
                                    subform=get_form_for_equation(self.default_temporal))

    @staticmethod
    def get_view_model():
        return SurfaceStimulusCreatorModel

    @staticmethod
    def get_required_datatype():
        return SurfaceIndex

    @staticmethod
    def get_input_name():
        return 'surface'

    @staticmethod
    def get_filters():
        return FilterChain(fields=[FilterChain.datatype + '.surface_type'], operations=["=="],
                           values=[CORTICAL])

    def fill_from_trait(self, trait):
        self.surface.data = trait.surface.hex
        self.spatial.data = type(trait.spatial)
        self.temporal.data = type(trait.temporal)
        self.temporal.subform_field = FormField(get_form_for_equation(type(trait.temporal)), self,
                                                self.NAME_TEMPORAL_PARAMS_DIV)
        self.temporal.subform_field.form.fill_from_trait(trait.temporal)
        self.spatial.subform_field = FormField(get_form_for_equation(type(trait.spatial)), self,
                                               self.NAME_SPATIAL_PARAMS_DIV)
        self.spatial.subform_field.form.fill_from_trait(trait.spatial)

    def get_rendering_dict(self):
        return {'adapter_form': self, 'next_action': 'form_spatial_surface_stimulus_equations',
                'spatial_params_div': self.NAME_SPATIAL_PARAMS_DIV,
                'temporal_params_div': self.NAME_TEMPORAL_PARAMS_DIV, 'legend': 'Stimulus interface'}


class SurfaceStimulusCreator(ABCSynchronous):
    """
    The purpose of this adapter is to create a StimuliSurface.
    """
    KEY_SURFACE = 'surface'
    KEY_SPATIAL = 'spatial'
    KEY_TEMPORAL = 'temporal'
    KEY_FOCAL_POINTS_TRIANGLES = 'focal_points_triangles'

    def get_form_class(self):
        return SurfaceStimulusCreatorForm

    def get_output(self):
        """
        Describes the outputs of the launch method.
        """
        return [StimuliSurfaceIndex]

    @staticmethod
    def prepare_stimuli_surface_from_view_model(view_model, load_full_surface=False):
        # type: (SurfaceStimulusCreatorModel, bool) -> StimuliSurface
        stimuli_surface = StimuliSurface()

        stimuli_surface.focal_points_triangles = view_model.focal_points_triangles
        stimuli_surface.spatial = view_model.spatial
        stimuli_surface.temporal = view_model.temporal

        surface_index = SurfaceStimulusCreator.load_entity_by_gid(view_model.surface.hex)
        if load_full_surface:
            stimuli_surface.surface = h5.load_from_index(surface_index, CorticalSurface)
        else:
            stimuli_surface.surface = CorticalSurface()
            stimuli_surface.gid = view_model.surface
            surface_h5 = h5.h5_file_for_index(surface_index)
            # We need to load surface triangles on stimuli because focal_points_surface property needs to acces them
            stimuli_surface.surface.triangles = surface_h5.triangles.load()
            surface_h5.close()

        return stimuli_surface

    def launch(self, view_model):
        # type: (SurfaceStimulusCreatorModel) -> [StimuliSurfaceIndex]
        """
        Used for creating a `StimuliSurface` instance
        """
        stimuli_surface = self.prepare_stimuli_surface_from_view_model(view_model, view_model.surface)
        stimuli_surface_index = StimuliSurfaceIndex()
        stimuli_surface_index.fill_from_has_traits(stimuli_surface)

        h5.store_complete(stimuli_surface, self.storage_path)
        return stimuli_surface_index

    def get_required_memory_size(self, view_model):
        # type: (SurfaceStimulusCreatorModel) -> int
        """
        Return the required memory to run this algorithm.
        """
        return -1

    def get_required_disk_size(self, view_model):
        # type: (SurfaceStimulusCreatorModel) -> int
        """
        Returns the required disk size to be able to run the adapter. (in kB)
        """
        return 0


class StimulusRegionSelectorForm(ABCAdapterForm):

    def __init__(self, project_id=None):
        super(StimulusRegionSelectorForm, self).__init__(project_id=project_id)
        self.region_stimulus = DataTypeSelectField(StimuliRegionIndex, self, name='existentEntitiesSelect',
                                                   label='Load Region Stimulus')
        self.display_name = SimpleStrField(self, name='display_name', label='Display name')

    def get_rendering_dict(self):
        return {'adapter_form': self, 'legend': 'Loaded stimulus'}


class RegionStimulusCreatorModel(ViewModel, StimuliRegion):
    temporal = Attr(field_type=TemporalApplicableEquation, label="Temporal Equation", default=PulseTrain())

    connectivity = DataTypeGidAttr(
        field_type=uuid.UUID,
        linked_datatype=Connectivity,
        label="Connectivity"
    )

    display_name = Str(
        label='Display name',
        required=False
    )


class RegionStimulusCreatorForm(ABCAdapterForm):
    NAME_TEMPORAL_PARAMS_DIV = 'temporal_params'
    default_temporal = PulseTrain
    choices = get_ui_name_to_equation_dict()

    def __init__(self, project_id=None):
        super(RegionStimulusCreatorForm, self).__init__(project_id=project_id)
        self.connectivity = TraitDataTypeSelectField(RegionStimulusCreatorModel.connectivity, self, name='connectivity')
        self.temporal = SelectField(RegionStimulusCreatorModel.temporal, self, name='temporal',
                                    choices=self.choices, subform=get_form_for_equation(self.default_temporal))

    @staticmethod
    def get_view_model():
        return RegionStimulusCreatorModel

    @staticmethod
    def get_filters():
        return None

    @staticmethod
    def get_input_name():
        return 'connectivity'

    @staticmethod
    def get_required_datatype():
        return ConnectivityIndex

    def fill_from_trait(self, trait):
        # type: (RegionStimulusCreatorModel) -> None
        self.connectivity.data = trait.connectivity.hex
        self.temporal.data = type(trait.temporal)
        self.temporal.subform_field = FormField(get_form_for_equation(type(trait.temporal)), self,
                                                self.NAME_TEMPORAL_PARAMS_DIV)
        self.temporal.subform_field.form.fill_from_trait(trait.temporal)

    def get_rendering_dict(self):
        return {'adapter_form': self, 'next_action': 'form_spatial_model_param_equations',
                'temporal_params_div': self.NAME_TEMPORAL_PARAMS_DIV, 'legend': 'Stimulus interface'}


class RegionStimulusCreator(ABCSynchronous):
    """
    The purpose of this adapter is to create a StimuliRegion.
    """

    def get_form_class(self):
        return RegionStimulusCreatorForm

    def get_output(self):
        """
        Describes the outputs of the launch method.
        """
        return [StimuliRegionIndex]

    def launch(self, view_model):
        # type: (RegionStimulusCreatorModel) -> [StimuliRegionIndex]
        """
        Used for creating a `StimuliRegion` instance
        """
        stimuli_region = StimuliRegion()
        stimuli_region.connectivity = Connectivity()
        stimuli_region.connectivity.gid = view_model.connectivity
        stimuli_region.weight = view_model.weight
        stimuli_region.temporal = view_model.temporal

        stimuli_region_idx = StimuliRegionIndex()
        stimuli_region_idx.fill_from_has_traits(stimuli_region)
        self.generic_attributes.user_tag_1 = view_model.display_name

        h5.store_complete(stimuli_region, self.storage_path)
        return stimuli_region_idx

    def get_required_disk_size(self, view_model):
        # type: (RegionStimulusCreatorModel) -> int
        """
        Returns the required disk size to be able to run the adapter. (in kB)
        """
        return 0

    def get_required_memory_size(self, view_model):
        # type: (RegionStimulusCreatorModel) -> int
        """
        Return the required memory to run this algorithm.
        """
        return -1
