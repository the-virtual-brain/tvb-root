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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import json
import numpy
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.adapters.abcdisplayer import ABCDisplayer
from tvb.core.entities.filters.chain import FilterChain
from tvb.adapters.datatypes.db.graph import ConnectivityMeasureIndex
from tvb.core.neotraits.forms import TraitDataTypeSelectField
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.core.utils import TVBJSONEncoder
from tvb.datatypes.graph import ConnectivityMeasure


class HistogramViewerModel(ViewModel):
    input_data = DataTypeGidAttr(
        linked_datatype=ConnectivityMeasure,
        label='Connectivity Measure',
        doc='A BCT computed measure for a Connectivity'
    )


class HistogramViewerForm(ABCAdapterForm):

    def __init__(self):
        super(HistogramViewerForm, self).__init__()
        self.input_data = TraitDataTypeSelectField(HistogramViewerModel.input_data, name='input_data',
                                                   conditions=self.get_filters())

    @staticmethod
    def get_view_model():
        return HistogramViewerModel

    @staticmethod
    def get_required_datatype():
        return ConnectivityMeasureIndex

    @staticmethod
    def get_input_name():
        return 'input_data'

    @staticmethod
    def get_filters():
        return FilterChain(fields=[FilterChain.datatype + '.ndim'], operations=["=="], values=[1])


class HistogramViewer(ABCDisplayer):
    """
    The viewer takes as input a result DataType as computed by BCT analyzers.
    """
    _ui_name = "Histogram Visualizer"

    def get_form_class(self):
        return HistogramViewerForm

    def launch(self, view_model):
        # type: (HistogramViewerModel) -> dict
        """
        Prepare input data for display.

        :param input_data: A BCT computed measure for a Connectivity
        :type input_data: `ConnectivityMeasureIndex`
        """
        params = self.prepare_parameters(view_model.input_data)
        return self.build_display_result("histogram/view", params, pages=dict(controlPage="histogram/controls"))

    def get_required_memory_size(self, view_model):
        # type: (HistogramViewerModel) -> numpy.ndarray
        """
        Return the required memory to run this algorithm.
        """
        input_data = self.load_entity_by_gid(view_model.input_data)
        return numpy.prod(input_data.shape) * 2

    @staticmethod
    def gather_params_dict(labels_list, values_list, title):
        params = dict(title=title, labels=json.dumps(labels_list, cls=TVBJSONEncoder), isSingleMode=True,
                      data=json.dumps(values_list), colors=json.dumps(values_list),
                      xposition='center' if min(values_list) < 0 else 'bottom',
                      minColor=min(values_list), maxColor=max(values_list))
        return params

    def prepare_parameters(self, connectivity_measure_gid):
        """
        Prepare all required parameters for a launch.
        """
        conn_measure = self.load_with_references(connectivity_measure_gid)
        assert isinstance(conn_measure, ConnectivityMeasure)
        labels_list = conn_measure.connectivity.region_labels.tolist()
        values_list = conn_measure.array_data.tolist()
        # A gradient of colors will be used for each node

        params = self.gather_params_dict(labels_list, values_list, "Connectivity Measure - " + conn_measure.title)
        return params
