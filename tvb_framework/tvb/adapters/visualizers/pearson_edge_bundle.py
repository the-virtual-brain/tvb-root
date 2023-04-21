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
import numpy
from tvb.adapters.visualizers.pearson_cross_correlation import PearsonCorrelationCoefficientVisualizerForm, \
    PearsonCorrelationCoefficientVisualizerModel
from tvb.adapters.visualizers.time_series import ABCSpaceDisplayer
from tvb.core.adapters.abcdisplayer import URLGenerator
from tvb.core.neocom import h5
from tvb.core.utils import TVBJSONEncoder
from tvb.datatypes.graph import CorrelationCoefficients


class PearsonEdgeBundle(ABCSpaceDisplayer):
    """
    Viewer for Pearson CorrelationCoefficients.
    Very similar to the CrossCorrelationVisualizer - this one done with Matplotlib
    """
    _ui_name = "Pearson Edge Bundle"
    _ui_subsection = "correlation_pearson_edge"

    def get_form_class(self):
        return PearsonCorrelationCoefficientVisualizerForm

    def get_required_memory_size(self, view_model):
        # type: (PearsonCorrelationCoefficientVisualizerModel) -> numpy.ndarray
        """Return required memory."""
        datatype_index = self.load_entity_by_gid(view_model.datatype)
        input_size = (datatype_index.data_length_1d, datatype_index.data_length_2d,
                      datatype_index.data_length_3d, datatype_index.data_length_4d)
        return numpy.prod(input_size) * 8.0

    def launch(self, view_model):
        # type: (PearsonCorrelationCoefficientVisualizerModel) -> dict
        """Construct data for visualization and launch it."""

        with h5.h5_file_for_gid(view_model.datatype) as datatype_h5:
            matrix_shape = datatype_h5.array_data.shape[0:2]
            ts_gid = datatype_h5.source.load()

        ts_index = self.load_entity_by_gid(ts_gid)
        state_list = ts_index.get_labels_for_dimension(1)
        mode_list = list(range(ts_index.data_length_4d))

        with h5.h5_file_for_index(ts_index) as ts_h5:
            labels = self.get_space_labels(ts_h5)

        if not labels:
            labels = None
        pars = dict(matrix_labels=json.dumps(labels, cls=TVBJSONEncoder),
                    matrix_shape=json.dumps(matrix_shape),
                    viewer_title='Pearson Edge Bundle',
                    url_base=URLGenerator.build_h5_url(view_model.datatype.hex, 'get_correlation_data', flatten="True",
                                                       parameter=''),
                    state_variable=0,
                    mode=mode_list[0],
                    state_list=state_list,
                    mode_list=mode_list,
                    pearson_min=CorrelationCoefficients.PEARSON_MIN,
                    pearson_max=CorrelationCoefficients.PEARSON_MAX,
                    thresh=0.5
                    )

        return self.build_display_result("pearson_edge_bundle/view", pars)
