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
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
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
.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>

"""

import numpy
from tvb.core.adapters.abcdisplayer import ABCMPLH5Displayer
from tvb.datatypes.graph import CorrelationCoefficients
from tvb.simulator.plot.tools import plot_tri_matrix


class PearsonCorrelationCoefficientVisualizer(ABCMPLH5Displayer):
    """
    Viewer for Pearson CorrelationCoefficients.
    Very similar to the CrossCorrelationVisualizer - this one done with Matplotlib
    """
    _ui_name = "Pearson Correlation Coefficients (MPLH5 Visualizer)"
    _ui_subsection = "correlation_pearson"


    def get_input_tree(self):
        """ Inform caller of the data we need as input """

        return [{"name": "corr_coefficients", "type": CorrelationCoefficients,
                 "label": "Correlation Coefficients", "required": True}]


    def get_required_memory_size(self, corr_coefficients):
        """Return required memory."""

        input_size = corr_coefficients.read_data_shape()
        return numpy.prod(input_size) * 8.0


    def plot(self, figure, corr_coefficients):
        """Construct data for visualization and launch it."""

        # Currently only the first mode & state-var are displayed.
        # TODO: display other modes / sv
        matrix_to_display = corr_coefficients.array_data[:, :, 0, 0]

        parent_ts = corr_coefficients.source
        parent_ts = self.load_entity_by_gid(parent_ts.gid)
        labels_to_display = parent_ts.get_space_labels()
        if not labels_to_display:
            labels_to_display = None

        plot_tri_matrix(matrix_to_display, figure=figure, node_labels=labels_to_display, color_anchor=0.)
