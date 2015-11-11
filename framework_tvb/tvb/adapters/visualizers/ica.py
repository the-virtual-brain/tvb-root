# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
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
A matrix displayer for the Independent Component Analysis.
It displays the mixing matrix of siae n_features x n_components

.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>

"""

from tvb.adapters.visualizers.matrix_viewer import MappedArraySVGVisualizerMixin
from tvb.core.adapters.abcdisplayer import ABCDisplayer
from tvb.datatypes.mode_decompositions import IndependentComponents
from tvb.basic.logger.builder import get_logger

LOG = get_logger(__name__)


class ICA(MappedArraySVGVisualizerMixin, ABCDisplayer):
    _ui_name = "Independent Components Analysis Visualizer"


    def get_input_tree(self):
        """Inform caller of the data we need"""
        return [{"name": "datatype", "type": IndependentComponents,
                 "label": "Independent component analysis:", "required": True},
                {"name": "i_svar", "type": 'int', 'default': 0,
                 "label": "Index of state variable (defaults to first state variable)",
                 }]


    def launch(self, datatype, i_svar=0):
        """Construct data for visualization and launch it."""
        # get data from IndependentComponents datatype, convert to json
        # HACK: dump only a 2D array
        W = datatype.get_data('mixing_matrix')
        pars = self.compute_params(W, 'Mixing matrix plot', '(Ellipsis, %d, 0)' % (i_svar))
        return self.build_display_result("matrix/svg_view", pars)