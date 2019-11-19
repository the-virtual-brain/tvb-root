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
.. Ionel Ortelecan <ionel.ortelecan@codemart.ro>
"""

from tvb.adapters.simulator.equation_forms import GaussianEquationForm, get_form_for_equation, \
    get_ui_name_to_equation_dict
from tvb.basic.neotraits._attr import Attr
from tvb.core.adapters.abcadapter import ABCAsynchronous, ABCAdapterForm
from tvb.core.entities.file.simulator.configurations_h5 import SimulatorConfigurationH5
from tvb.core.entities.file.simulator.h5_factory import equation_h5_factory
from tvb.datatypes.local_connectivity import LocalConnectivity
from tvb.adapters.datatypes.db.local_connectivity import LocalConnectivityIndex
from tvb.adapters.datatypes.db.surface import SurfaceIndex
from tvb.core.neotraits.forms import DataTypeSelectField, ScalarField, SimpleSelectField, FormField
from tvb.core.neocom import h5
from tvb.interfaces.web.controllers.decorators import using_template


class LocalConnectivitySelectorForm(ABCAdapterForm):

    def __init__(self, prefix='', project_id=None):
        super(LocalConnectivitySelectorForm, self).__init__(prefix, project_id)
        self.existentEntitiesSelect = DataTypeSelectField(self.get_required_datatype(), self,
                                                          name='existentEntitiesSelect',
                                                          label='Load Local Connectivity')

    @staticmethod
    def get_required_datatype():
        return LocalConnectivityIndex

    @staticmethod
    def get_input_name():
        pass

    @staticmethod
    def get_filters():
        return None

    @using_template('spatial/spatial_fragment')
    def __str__(self):
        return {'form': self}


class LocalConnectivityCreatorForm(ABCAdapterForm):
    NAME_EQUATION_PARAMS_DIV = 'spatial_params'

    def __init__(self, equation_choices, prefix='', project_id=None):
        super(LocalConnectivityCreatorForm, self).__init__(prefix, project_id)
        self.surface = DataTypeSelectField(self.get_required_datatype(), self, name=self.get_input_name(),
                                           required=LocalConnectivity.surface.required,
                                           label=LocalConnectivity.surface.label, doc=LocalConnectivity.surface.doc)
        self.spatial = SimpleSelectField(equation_choices, self, name='spatial', label='Spatial', required=True,
                                         default=type(LocalConnectivity.equation.default))

        self.spatial_params = FormField(GaussianEquationForm, self, name=self.NAME_EQUATION_PARAMS_DIV,
                                        label='Equation parameters')
        self.cutoff = ScalarField(LocalConnectivity.cutoff, self)
        self.display_name = ScalarField(Attr(str, label='Display name'), self)

    @staticmethod
    def get_required_datatype():
        return SurfaceIndex

    @staticmethod
    def get_input_name():
        return 'surface'

    @staticmethod
    def get_filters():
        return None

    def get_traited_datatype(self):
        return LocalConnectivity()

    @using_template('spatial/spatial_fragment')
    def __str__(self):
        return {'form': self, 'next_action': 'form_spatial_local_connectivity_data',
                'equation_params_div': self.NAME_EQUATION_PARAMS_DIV}


class LocalConnectivityCreator(ABCAsynchronous):
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

    def launch(self, **kwargs):
        """
        Used for creating a `LocalConnectivity`
        """
        local_connectivity = LocalConnectivity()
        local_connectivity.cutoff = float(kwargs[self.KEY_CUTOFF])
        if not self.surface_index:
            surface_gid = kwargs[self.KEY_SURFACE]
            self.surface_index = self.load_entity_by_gid(surface_gid)
        surface = h5.load_from_index(self.surface_index)
        local_connectivity.surface = surface
        local_connectivity.equation = get_ui_name_to_equation_dict().get(kwargs[self.KEY_EQUATION])()

        equation_form = get_form_for_equation(type(local_connectivity.equation))(prefix=self.KEY_EQUATION)
        equation_form.fill_from_post(kwargs)
        equation_form.fill_trait(local_connectivity.equation)
        local_connectivity.compute_sparse_matrix()

        equation_h5_class = equation_h5_factory(type(local_connectivity.equation))
        equation_h5_path = h5.path_for(self.storage_path, equation_h5_class, local_connectivity.equation.gid)
        with equation_h5_class(equation_h5_path) as equation_h5:
            equation_h5.store(local_connectivity.equation)
            equation_h5.type.store(SimulatorConfigurationH5.get_full_class_name(type(local_connectivity.equation)))

        return h5.store_complete(local_connectivity, self.storage_path)

    def get_required_disk_size(self, **kwargs):
        """
        Returns the required disk size to be able to run the adapter. (in kB)
        """
        if self.KEY_SURFACE in kwargs:
            surface_gid = kwargs[self.KEY_SURFACE]
            self.surface_index = self.load_entity_by_gid(surface_gid)
            points_no = float(kwargs[self.KEY_CUTOFF]) / self.surface_index.edge_mean_length
            disk_size_b = self.surface_index.number_of_vertices * points_no * points_no * 8
            return self.array_size2kb(disk_size_b)
        return 0

    def get_required_memory_size(self, **kwargs):
        """
        Return the required memory to run this algorithm.
        """
        return self.get_required_disk_size(**kwargs)
