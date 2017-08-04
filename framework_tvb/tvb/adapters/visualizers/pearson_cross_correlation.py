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
.. moduleauthor:: Dan Pop <dan.pop@codemart.ro>
.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>

"""

import json
import numpy
from tvb.adapters.visualizers.matrix_viewer import MappedArrayVisualizer
from tvb.datatypes.graph import CorrelationCoefficients


class PearsonCorrelationCoefficientVisualizer(MappedArrayVisualizer):
    """
    Viewer for Pearson CorrelationCoefficients.
    Very similar to the CrossCorrelationVisualizer - this one done with Matplotlib
    """
    _ui_name = "Pearson Correlation Coefficients"
    _ui_subsection = "correlation_pearson"

    def get_input_tree(self):
        """ Inform caller of the data we need as input """

        return [{"name": "datatype", "type": CorrelationCoefficients,
                 "label": "Correlation Coefficients", "required": True}]

    def get_required_memory_size(self, datatype):
        """Return required memory."""

        input_size = datatype.read_data_shape()
        return numpy.prod(input_size) * 8.0

    def launch(self, datatype):
        """Construct data for visualization and launch it."""

        matrix_shape = datatype.array_data.shape[0:2]
        parent_ts = datatype.source
        parent_ts = self.load_entity_by_gid(parent_ts.gid)
        labels = parent_ts.get_space_labels()
        state_list = datatype.source.labels_dimensions.get(datatype.source.labels_ordering[1], [])
        mode_list = range(datatype.source._length_4d)
        if not labels:
            labels = None
        pars = dict(matrix_labels=json.dumps([labels, labels]),
                    matrix_shape=json.dumps(matrix_shape),
                    viewer_title='Cross Corelation Matrix plot',
                    url_base=MappedArrayVisualizer.paths2url(datatype, "get_correlation_data", parameter=""),
                    state_variable=state_list[0],
                    mode=mode_list[0],
                    state_list=state_list,
                    mode_list=mode_list,
                    pearson_min=CorrelationCoefficients.PEARSON_MIN,
                    pearson_max=CorrelationCoefficients.PEARSON_MAX)

        return self.build_display_result("pearson_correlation/view", pars)
