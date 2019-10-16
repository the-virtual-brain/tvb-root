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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import json
import numpy
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.adapters.abcdisplayer import ABCDisplayer
from tvb.basic.filters.chain import FilterChain
from tvb.core.entities.model.datatypes.graph import ConnectivityMeasureIndex
from tvb.core.neotraits._forms import DataTypeSelectField


class HistogramViewerForm(ABCAdapterForm):

    def __init__(self, prefix='', project_id=None):
        super(HistogramViewerForm, self).__init__(prefix, project_id)
        self.input_data = DataTypeSelectField(self.get_required_datatype(), self, name='input_data', required=True,
                                              label='Connectivity Measure', conditions=self.get_filters(),
                                              doc='A BCT computed measure for a Connectivity')

    @staticmethod
    def get_required_datatype():
        return ConnectivityMeasureIndex

    @staticmethod
    def get_input_name():
        return '_input_data'

    @staticmethod
    def get_filters():
        return FilterChain(fields=[FilterChain.datatype + '.ndim'], operations=["=="], values=[1])


class HistogramViewer(ABCDisplayer):
    """
    The viewer takes as input a result DataType as computed by BCT analyzers.
    """
    _ui_name = "Connectivity Measure Visualizer"
    form = None

    def get_form(self):
        if not self.form:
            return HistogramViewerForm
        return self.form

    def set_form(self, form):
        self.form = form

    def get_input_tree(self): return None


    #TODO: migrate to neotraits
    def launch(self, input_data):
        """
        Prepare input data for display.

        :param input_data: A BCT computed measure for a Connectivity
        :type input_data: `ConnectivityMeasureIndex`
        """
        params = self.prepare_parameters(input_data)
        return self.build_display_result("histogram/view", params, pages=dict(controlPage="histogram/controls"))


    def get_required_memory_size(self, input_data, figure_size):
        """
        Return the required memory to run this algorithm.
        """
        return numpy.prod(input_data.shape) * 2


    def generate_preview(self, input_data, figure_size):
        """
        The preview for the burst page.
        """
        params = self.prepare_parameters(input_data)
        return self.build_display_result("histogram/view", params)


    def prepare_parameters(self, input_data):
        """
        Prepare all required parameters for a launch.
        """
        labels_list = input_data.connectivity.region_labels.tolist()
        values_list = input_data.array_data.tolist()
        # A gradient of colors will be used for each node
        colors_list = values_list

        params = dict(title="Connectivity Measure - " + input_data.title, labels=json.dumps(labels_list),
                      data=json.dumps(values_list), colors=json.dumps(colors_list),
                      xposition='center' if min(values_list) < 0 else 'bottom',
                      minColor=min(colors_list), maxColor=max(colors_list))
        return params
    
    