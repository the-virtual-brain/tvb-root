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
.. moduleauthor:: Dan Pop <dan.pop@codemart.ro>
.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>

"""

import json
from tvb.adapters.datatypes.db.graph import CorrelationCoefficientsIndex
from tvb.adapters.visualizers.matrix_viewer import ABCMappedArraySVGVisualizer
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.neocom import h5
from tvb.core.adapters.abcdisplayer import URLGenerator
from tvb.core.neotraits.forms import TraitDataTypeSelectField
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.core.utils import TVBJSONEncoder
from tvb.datatypes.graph import CorrelationCoefficients


class PearsonCorrelationCoefficientVisualizerModel(ViewModel):
    datatype = DataTypeGidAttr(
        linked_datatype=CorrelationCoefficients,
        label='Correlation Coefficients'
    )


class PearsonCorrelationCoefficientVisualizerForm(ABCAdapterForm):

    def __init__(self):
        super(PearsonCorrelationCoefficientVisualizerForm, self).__init__()
        self.datatype = TraitDataTypeSelectField(PearsonCorrelationCoefficientVisualizerModel.datatype,
                                                 name='datatype', conditions=self.get_filters())

    @staticmethod
    def get_view_model():
        return PearsonCorrelationCoefficientVisualizerModel

    @staticmethod
    def get_required_datatype():
        return CorrelationCoefficientsIndex

    @staticmethod
    def get_input_name():
        return 'datatype'

    @staticmethod
    def get_filters():
        return FilterChain(fields=[FilterChain.datatype + '.has_valid_time_series'], operations=['=='], values=[True])


class PearsonCorrelationCoefficientVisualizer(ABCMappedArraySVGVisualizer):
    """
    Viewer for Pearson CorrelationCoefficients.
    Very similar to the CrossCorrelationVisualizer - this one done with Matplotlib
    """
    _ui_name = "Pearson Correlation Coefficients"
    _ui_subsection = "correlation_pearson"

    def get_form_class(self):
        return PearsonCorrelationCoefficientVisualizerForm

    def launch(self, view_model):
        """Construct data for visualization and launch it."""
        cc_gid = view_model.datatype
        cc_index = self.load_entity_by_gid(cc_gid)
        assert isinstance(cc_index, CorrelationCoefficientsIndex)
        matrix_shape = cc_index.parsed_shape[0:2]

        ts_gid = cc_index.fk_source_gid
        ts_index = self.load_entity_by_gid(ts_gid)
        state_list = ts_index.get_labels_for_dimension(1)
        mode_list = list(range(ts_index.data_length_4d))
        with h5.h5_file_for_index(ts_index) as ts_h5:
            labels = self.get_space_labels(ts_h5)
            if not labels:
                labels = None

        pars = dict(matrix_labels=json.dumps([labels, labels], cls=TVBJSONEncoder),
                    matrix_shape=json.dumps(matrix_shape),
                    viewer_title='Cross Correlation Matrix Plot',
                    url_base=URLGenerator.build_h5_url(cc_gid, 'get_correlation_data', parameter=''),
                    state_variable=state_list[0],
                    mode=mode_list[0],
                    state_list=state_list,
                    mode_list=mode_list,
                    pearson_min=CorrelationCoefficients.PEARSON_MIN,
                    pearson_max=CorrelationCoefficients.PEARSON_MAX)

        return self.build_display_result("pearson_correlation/view", pars)
