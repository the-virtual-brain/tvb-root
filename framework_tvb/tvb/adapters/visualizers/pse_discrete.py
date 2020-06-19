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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""
from tvb.adapters.visualizers.pse import PSEDiscreteGroupModel
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.adapters.abcdisplayer import ABCDisplayer
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.entities.model.model_datatype import DataTypeGroup
from tvb.core.neotraits.forms import TraitDataTypeSelectField
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr

MAX_NUMBER_OF_POINT_TO_SUPPORT = 512


class DiscretePSEAdapterModel(ViewModel):
    datatype_group = DataTypeGidAttr(
        linked_datatype=DataTypeGroup,
        label='Datatype Group'
    )


class DiscretePSEAdapterForm(ABCAdapterForm):

    def __init__(self, prefix='', project_id=None):
        super(DiscretePSEAdapterForm, self).__init__(prefix, project_id)
        self.datatype_group = TraitDataTypeSelectField(DiscretePSEAdapterModel.datatype_group, self,
                                                       name='datatype_group', conditions=self.get_filters())

    @staticmethod
    def get_view_model():
        return DiscretePSEAdapterModel

    @staticmethod
    def get_required_datatype():
        return DataTypeGroup

    @staticmethod
    def get_input_name():
        return 'datatype_group'

    @staticmethod
    def get_filters():
        return FilterChain(fields=[FilterChain.datatype + ".no_of_ranges", FilterChain.datatype + ".no_of_ranges",
                                   FilterChain.datatype + ".count_results"],
                           operations=["<=", ">=", "<="],
                           values=[2, 1, MAX_NUMBER_OF_POINT_TO_SUPPORT])


class DiscretePSEAdapter(ABCDisplayer):
    """
    Visualization adapter for Parameter Space Exploration.
    Will be used as a generic visualizer, accessible when input entity is DataTypeGroup.
    Will also be used in Burst as a supplementary navigation layer.
    """
    _ui_name = "Discrete Parameter Space Exploration"
    _ui_subsection = "pse"

    def get_form_class(self):
        return DiscretePSEAdapterForm

    def get_required_memory_size(self, view_model):
        # type: (DiscretePSEAdapterModel) -> int
        """
        Return the required memory to run this algorithm.
        """
        # Don't know how much memory is needed.
        return -1

    def launch(self, view_model):
        # type: (DiscretePSEAdapterModel) -> dict
        """
        Launch the visualizer.
        """
        pse_model = PSEDiscreteGroupModel(view_model.datatype_group.hex, None, None, '')
        pse_context = pse_model.pse_context
        pse_context.prepare_individual_jsons()

        return self.build_display_result('pse_discrete/view', pse_context,
                                         pages=dict(controlPage="pse_discrete/controls"))

    @staticmethod
    def prepare_parameters(datatype_group_gid, back_page, color_metric=None, size_metric=None):
        pse_model = PSEDiscreteGroupModel(datatype_group_gid, color_metric, size_metric, back_page)
        return pse_model.pse_context
