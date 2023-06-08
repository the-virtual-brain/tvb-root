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
.. moduleauthor:: Paula Popa <paula.popa@codemart.ro>
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
"""

from tvb.adapters.datatypes.db.local_connectivity import LocalConnectivityIndex
from tvb.adapters.datatypes.db.surface import SurfaceIndex
from tvb.adapters.forms.equation_forms import GaussianEquationForm, get_form_for_equation, SpatialEquationsEnum
from tvb.basic.neotraits.api import Attr, EnumAttr
from tvb.core.adapters.abcadapter import ABCAdapterForm, ABCAdapter
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import FormField, SelectField, TraitDataTypeSelectField, FloatField, StrField
from tvb.core.neotraits.spatial_model import SpatialModel
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr, Str
from tvb.datatypes.local_connectivity import LocalConnectivity
from tvb.datatypes.surfaces import Surface, SurfaceTypesEnum


class LocalConnectivitySelectorForm(ABCAdapterForm):

    def __init__(self):
        super(LocalConnectivitySelectorForm, self).__init__()
        traited_attr = Attr(self.get_required_datatype(), label='Load Local Connectivity', required=False)
        self.existentEntitiesSelect = TraitDataTypeSelectField(traited_attr, name='existentEntitiesSelect')

    @staticmethod
    def get_required_datatype():
        return LocalConnectivityIndex

    @staticmethod
    def get_input_name():
        pass

    @staticmethod
    def get_filters():
        return None

    def get_rendering_dict(self):
        return {'adapter_form': self, 'legend': 'Selected entity'}


class LocalConnectivityCreatorModel(ViewModel, LocalConnectivity, SpatialModel):
    surface = DataTypeGidAttr(
        linked_datatype=Surface,
        label=LocalConnectivity.surface.label
    )

    display_name = Str(
        label='Display name',
        required=False
    )

    @staticmethod
    def get_equation_information():
        return {
            LocalConnectivityCreatorModel.equation.label.lower(): 'equation'
        }


KEY_LCONN = "local-conn"


class LocalConnectivityCreatorForm(ABCAdapterForm):
    NAME_EQUATION_PARAMS_DIV = 'spatial_params'

    def __init__(self):
        super(LocalConnectivityCreatorForm, self).__init__()
        self.surface = TraitDataTypeSelectField(LocalConnectivityCreatorModel.surface, name=self.get_input_name(),
                                                conditions=self.get_filters())
        self.spatial = SelectField(EnumAttr(field_type=SpatialEquationsEnum,
                                            default=SpatialEquationsEnum.GAUSSIAN.instance, required=True),
                                   name='spatial', display_none_choice=False, subform=GaussianEquationForm,
                                   session_key=KEY_LCONN)

        self.cutoff = FloatField(LocalConnectivityCreatorModel.cutoff)
        self.display_name = StrField(LocalConnectivityCreatorModel.display_name, name='display_name')

        del self.spatial.choices[-1]

    @staticmethod
    def get_view_model():
        return LocalConnectivityCreatorModel

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
        # type: (LocalConnectivityCreatorModel) -> None
        self.surface.data = trait.surface.hex
        self.cutoff.data = trait.cutoff
        self.display_name.data = trait.display_name
        if trait.equation:
            lc_equation = trait.equation
        else:
            lc_equation = LocalConnectivity.equation.default
        self.spatial.data = type(lc_equation)
        self.spatial.subform_field = FormField(get_form_for_equation(type(lc_equation)),
                                               self.NAME_EQUATION_PARAMS_DIV)
        self.spatial.subform_field.form.fill_from_trait(lc_equation)

    def get_rendering_dict(self):
        return {'adapter_form': self, 'next_action': 'form_spatial_local_connectivity_data',
                'equation_params_div': self.NAME_EQUATION_PARAMS_DIV, 'legend': 'Local connectivity parameters'}


class LocalConnectivityCreator(ABCAdapter):
    """
    The purpose of this adapter is create a LocalConnectivity.
    """
    KEY_SURFACE = 'surface'
    KEY_EQUATION = 'equation'
    KEY_CUTOFF = 'cutoff'
    KEY_DISPLAY_NAME = 'display_name'

    def get_form_class(self):
        return LocalConnectivityCreatorForm

    def get_output(self):
        """
        Describes the outputs of the launch method.
        """
        return [LocalConnectivityIndex]

    def launch(self, view_model):
        # type: (LocalConnectivityCreatorModel) -> [LocalConnectivityIndex]
        """
        Used for creating a `LocalConnectivity`
        """
        local_connectivity = LocalConnectivity()
        local_connectivity.cutoff = view_model.cutoff
        if not self.surface_index:
            self.surface_index = self.load_entity_by_gid(view_model.surface)
        surface = h5.load_from_index(self.surface_index)
        local_connectivity.surface = surface
        local_connectivity.equation = view_model.equation
        local_connectivity.compute_sparse_matrix()
        self.generic_attributes.user_tag_1 = view_model.display_name

        return self.store_complete(local_connectivity)

    def get_required_disk_size(self, view_model):
        # type: (LocalConnectivityCreatorModel) -> int
        """
        Returns the required disk size to be able to run the adapter. (in kB)
        """
        if view_model.surface:
            self.surface_index = self.load_entity_by_gid(view_model.surface)
            points_no = view_model.cutoff / self.surface_index.edge_mean_length
            disk_size_b = self.surface_index.number_of_vertices * points_no * points_no * 8
            return self.array_size2kb(disk_size_b)
        return 0

    def get_required_memory_size(self, view_model):
        # type: (LocalConnectivityCreatorModel) -> int
        """
        Return the required memory to run this algorithm.
        """
        return self.get_required_disk_size(view_model)
