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
.. Ionel Ortelecan <ionel.ortelecan@codemart.ro>
"""

import uuid

from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.datatypes.db.patterns import StimuliRegionIndex, StimuliSurfaceIndex
from tvb.adapters.datatypes.db.surface import SurfaceIndex
from tvb.adapters.forms.equation_forms import get_form_for_equation, SpatialEquationsEnum, TemporalEquationsEnum
from tvb.basic.neotraits.api import Attr, EnumAttr
from tvb.core.adapters.abcadapter import ABCAdapterForm, AdapterLaunchModeEnum, ABCAdapter
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import FormField, TraitDataTypeSelectField, SelectField, StrField
from tvb.core.neotraits.spatial_model import SpatialModel
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr, Str
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.patterns import StimuliSurface, StimuliRegion
from tvb.datatypes.surfaces import CorticalSurface, SurfaceTypesEnum


class StimulusSurfaceSelectorForm(ABCAdapterForm):

    def __init__(self):
        super(StimulusSurfaceSelectorForm, self).__init__()
        traited_attr = Attr(StimuliSurfaceIndex, label='Load Surface Stimulus', required=False)
        self.surface_stimulus = TraitDataTypeSelectField(traited_attr, name='existentEntitiesSelect')
        self.display_name = StrField(SurfaceStimulusCreatorModel.display_name, name='display_name')

    def get_rendering_dict(self):
        return {'adapter_form': self, 'legend': 'Loaded stimulus'}


class SurfaceStimulusCreatorModel(ViewModel, StimuliSurface, SpatialModel):
    spatial = EnumAttr(field_type=SpatialEquationsEnum, label="Spatial Equation",
                       default=SpatialEquationsEnum.SIGMOID.instance)
    temporal = EnumAttr(field_type=TemporalEquationsEnum, label="Temporal Equation",
                        default=TemporalEquationsEnum.PULSETRAIN.instance)

    surface = DataTypeGidAttr(
        linked_datatype=CorticalSurface,
        label=StimuliSurface.surface.label
    )

    display_name = Str(
        label='Display name',
        required=False
    )

    @staticmethod
    def get_equation_information():
        return {
            SurfaceStimulusCreatorModel.spatial.label: 'spatial',
            SurfaceStimulusCreatorModel.temporal.label: 'temporal'
        }


KEY_REGION_STIMULUS = "stim-region"
KEY_SURFACE_STIMULUS = "stim-surface"


class SurfaceStimulusCreatorForm(ABCAdapterForm):
    NAME_SPATIAL_PARAMS_DIV = 'spatial_params'
    NAME_TEMPORAL_PARAMS_DIV = 'temporal_params'
    default_spatial = SpatialEquationsEnum.SIGMOID
    default_temporal = TemporalEquationsEnum.PULSETRAIN

    def __init__(self):
        super(SurfaceStimulusCreatorForm, self).__init__()

        self.surface = TraitDataTypeSelectField(SurfaceStimulusCreatorModel.surface, name='surface',
                                                conditions=self.get_filters())
        self.spatial = SelectField(SurfaceStimulusCreatorModel.spatial, name='spatial',
                                   subform=get_form_for_equation(self.default_spatial.value),
                                   session_key=KEY_SURFACE_STIMULUS)
        self.temporal = SelectField(SurfaceStimulusCreatorModel.temporal, name='temporal',
                                    subform=get_form_for_equation(self.default_temporal.value),
                                    session_key=KEY_SURFACE_STIMULUS)

        del self.spatial.choices[-1]

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
                           values=[SurfaceTypesEnum.CORTICAL_SURFACE.value])

    def fill_from_trait(self, trait):
        self.surface.data = trait.surface.hex
        self.spatial.data = type(trait.spatial)
        self.temporal.data = type(trait.temporal)
        self.temporal.subform_field = FormField(get_form_for_equation(type(trait.temporal)),
                                                self.NAME_TEMPORAL_PARAMS_DIV)
        self.temporal.subform_field.form.fill_from_trait(trait.temporal)
        self.spatial.subform_field = FormField(get_form_for_equation(type(trait.spatial)),
                                               self.NAME_SPATIAL_PARAMS_DIV)
        self.spatial.subform_field.form.fill_from_trait(trait.spatial)

    def get_rendering_dict(self):
        return {'adapter_form': self, 'next_action': 'form_spatial_surface_stimulus_equations',
                'spatial_params_div': self.NAME_SPATIAL_PARAMS_DIV,
                'temporal_params_div': self.NAME_TEMPORAL_PARAMS_DIV, 'legend': 'Stimulus interface'}


class SurfaceStimulusCreator(ABCAdapter):
    """
    The purpose of this adapter is to create a StimuliSurface.
    """
    KEY_SURFACE = 'surface'
    KEY_SPATIAL = 'spatial'
    KEY_TEMPORAL = 'temporal'
    KEY_FOCAL_POINTS_TRIANGLES = 'focal_points_triangles'
    launch_mode = AdapterLaunchModeEnum.SYNC_SAME_MEM

    def get_form_class(self):
        return SurfaceStimulusCreatorForm

    def get_output(self):
        """
        Describes the outputs of the launch method.
        """
        return [StimuliSurfaceIndex]

    def prepare_stimuli_surface_from_view_model(self, view_model, load_full_surface=False):
        # type: (SurfaceStimulusCreatorModel, bool) -> StimuliSurface
        stimuli_surface = StimuliSurface()

        stimuli_surface.focal_points_triangles = view_model.focal_points_triangles
        stimuli_surface.spatial = view_model.spatial
        stimuli_surface.temporal = view_model.temporal

        if load_full_surface:
            stimuli_surface.surface = self.load_traited_by_gid(view_model.surface)
        else:
            stimuli_surface.surface = CorticalSurface()
            stimuli_surface.gid = view_model.surface
            # We need to load surface triangles on stimuli because focal_points_surface property needs to acces them
            with h5.h5_file_for_gid(view_model.surface) as surface_h5:
                stimuli_surface.surface.triangles = surface_h5.triangles.load()

        return stimuli_surface

    def launch(self, view_model):
        # type: (SurfaceStimulusCreatorModel) -> [StimuliSurfaceIndex]
        """
        Used for creating a `StimuliSurface` instance
        """
        self.generic_attributes.user_tag_1 = view_model.display_name
        stimuli_surface = self.prepare_stimuli_surface_from_view_model(view_model, view_model.surface)
        stimuli_surface_index = self.store_complete(stimuli_surface)
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

    def __init__(self):
        super(StimulusRegionSelectorForm, self).__init__()
        traited_attr = Attr(StimuliRegionIndex, label='Load Region Stimulus', required=False)
        self.region_stimulus = TraitDataTypeSelectField(traited_attr, name='existentEntitiesSelect')
        self.display_name = StrField(RegionStimulusCreatorModel.display_name, name='display_name')

    def get_rendering_dict(self):
        return {'adapter_form': self, 'legend': 'Loaded stimulus'}


class RegionStimulusCreatorModel(ViewModel, StimuliRegion, SpatialModel):
    temporal = EnumAttr(field_type=TemporalEquationsEnum, label="Temporal Equation",
                        default=TemporalEquationsEnum.PULSETRAIN.instance)

    connectivity = DataTypeGidAttr(
        field_type=uuid.UUID,
        linked_datatype=Connectivity,
        label="Connectivity"
    )

    display_name = Str(
        label='Display name',
        required=False
    )

    @staticmethod
    def get_equation_information():
        return {
            RegionStimulusCreatorModel.temporal.label: 'temporal'
        }


class RegionStimulusCreatorForm(ABCAdapterForm):
    NAME_TEMPORAL_PARAMS_DIV = 'temporal_params'
    default_temporal = TemporalEquationsEnum.PULSETRAIN

    def __init__(self):
        super(RegionStimulusCreatorForm, self).__init__()
        self.connectivity = TraitDataTypeSelectField(RegionStimulusCreatorModel.connectivity, name='connectivity')
        self.temporal = SelectField(RegionStimulusCreatorModel.temporal, name='temporal',
                                    subform=get_form_for_equation(self.default_temporal.value),
                                    session_key=KEY_REGION_STIMULUS)

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
        self.temporal.subform_field = FormField(get_form_for_equation(type(trait.temporal)),
                                                self.NAME_TEMPORAL_PARAMS_DIV)
        self.temporal.subform_field.form.fill_from_trait(trait.temporal)

    def get_rendering_dict(self):
        return {'adapter_form': self, 'next_action': 'form_spatial_model_param_equations',
                'temporal_params_div': self.NAME_TEMPORAL_PARAMS_DIV, 'legend': 'Stimulus interface'}


class RegionStimulusCreator(ABCAdapter):
    """
    The purpose of this adapter is to create a StimuliRegion.
    """
    launch_mode = AdapterLaunchModeEnum.SYNC_SAME_MEM

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
        self.generic_attributes.user_tag_1 = view_model.display_name

        stimuli_region_idx = self.store_complete(stimuli_region)
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
