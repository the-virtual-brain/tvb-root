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

from tvb.core.adapters.abcadapter import ABCAsynchronous, ABCAdapterForm
from tvb.datatypes.local_connectivity import LocalConnectivity
from tvb.datatypes.equations import Equation

from tvb.core.entities.model.datatypes.local_connectivity import LocalConnectivityIndex
from tvb.core.entities.model.datatypes.surface import SurfaceIndex
from tvb.core.neotraits._forms import DataTypeSelectField, ScalarField

class LocalConnectivitySelectorForm(ABCAdapterForm):

    def __init__(self, prefix='', project_id=None):
        super(LocalConnectivitySelectorForm, self).__init__(prefix, project_id)
        self.existentEntitiesSelect = DataTypeSelectField(self.get_required_datatype(), self, name='existentEntitiesSelect',
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


#TODO: work also on controller/template. Same for stimuli creators.
class LocalConnectivityCreatorForm(ABCAdapterForm):

    def __init__(self, prefix='', project_id=None):
        super(LocalConnectivityCreatorForm, self).__init__(prefix, project_id)
        self.surface = DataTypeSelectField(self.get_required_datatype(), self, name=self.get_input_name(),
                                           required=LocalConnectivity.surface.required,
                                           label=LocalConnectivity.surface.label, doc=LocalConnectivity.surface.doc)
        self.equation = ScalarField(LocalConnectivity.equation, self)
        self.cutoff = ScalarField(LocalConnectivity.cutoff, self)

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


class LocalConnectivityCreator(ABCAsynchronous):
    """
    The purpose of this adapter is create a LocalConnectivity.
    """
    form = None

    def get_input_tree(self): return None

    def get_select_field_form(self):
        return LocalConnectivitySelectorForm

    def get_form(self):
        if not self.form:
            return LocalConnectivityCreatorForm
        return self.form

    def set_form(self, form):
        self.form = form

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
        local_connectivity.cutoff = float(kwargs['cutoff'])
        local_connectivity.surface = kwargs['surface']
        local_connectivity.equation = self.get_lconn_equation(kwargs)
        local_connectivity.compute_sparse_matrix()

        return local_connectivity

    
    def get_lconn_equation(self, kwargs):
        """
        Get the equation for the local connectivity from a dictionary of arguments.
        """
        return Equation.build_equation_from_dict('equation', kwargs)


    def get_required_disk_size(self, **kwargs):
        """
        Returns the required disk size to be able to run the adapter. (in kB)
        """
        if 'surface' in kwargs:
            surface_index = kwargs['surface']
            points_no = float(kwargs['cutoff']) / surface_index.edge_length_mean
            disk_size_b = surface_index.number_of_vertices * points_no * points_no * 8
            return self.array_size2kb(disk_size_b)
        return 0


    def get_required_memory_size(self, **kwargs):
        """
        Return the required memory to run this algorithm.
        """
        return self.get_required_disk_size(**kwargs)



    
    
    