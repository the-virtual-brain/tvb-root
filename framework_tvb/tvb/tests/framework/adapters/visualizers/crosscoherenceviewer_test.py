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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.adapters.visualizers.cross_coherence import CrossCoherenceVisualizer


class TestCrossCoherenceViewer(TransactionalTestCase):
    """
    Unit-tests for Cross Coherence Viewer.
    """

    def test_launch(self, cross_coherence_factory):
        """
        Check that all required keys are present in output from BrainViewer launch.
        """
        cross_coherence_index = cross_coherence_factory()
        viewer = CrossCoherenceVisualizer()
        view_model = viewer.get_view_model_class()()
        view_model.datatype = cross_coherence_index.gid
        result = viewer.launch(view_model)
        expected_keys = ['matrix_data', 'matrix_shape', 'frequency']
        for key in expected_keys:
            assert key in result
