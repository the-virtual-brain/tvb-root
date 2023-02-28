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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import json
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.adapters.visualizers.eeg_monitor import EegMonitor


class TestEEGMonitor(TransactionalTestCase):
    """
    Unit-tests for EEG Viewer.
    """

    def test_launch(self, time_series_index_factory):
        """
        Check that all required keys are present in output from EegMonitor launch.
        """
        time_series_index = time_series_index_factory()
        viewer = EegMonitor()
        view_model = viewer.get_view_model_class()()
        view_model.input_data = time_series_index.gid
        result = viewer.launch(view_model)
        expected_keys = ['tsNames', 'groupedLabels', 'tsModes', 'tsStateVars', 'longestChannelLength',
                         'label_x', 'entities', 'page_size', 'number_of_visible_points',
                         'extended_view', 'initialSelection', 'ag_settings', 'ag_settings']

        for key in expected_keys:
            assert key in result, "key not found %s" % key

        expected_ag_settings = ['channelsPerSet', 'channelLabels', 'noOfChannels', 'translationStep',
                                'normalizedSteps', 'nan_value_found', 'baseURLS', 'pageSize',
                                'nrOfPages', 'timeSetPaths', 'totalLength', 'number_of_visible_points',
                                'extended_view', 'measurePointsSelectionGIDs']

        ag_settings = json.loads(result['ag_settings'])

        for key in expected_ag_settings:
            assert key in ag_settings, "ag_settings should have the key %s" % key
