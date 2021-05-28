# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

import numpy
from tvb.tests.library.base_testcase import BaseTestCase
from tvb.datatypes import graph, time_series, connectivity


class TestGraph(BaseTestCase):
    """
    Tests the defaults for `tvb.datatypes.graph` module.
    """

    def test_covariance(self):
        data = numpy.random.random((10, 10))
        ts = time_series.TimeSeries(data=data, title="test")
        dt = graph.Covariance(source=ts, array_data=data)

        assert dt.array_data.shape == (10, 10)

        summary = dt.summary_info()
        assert summary['Graph type'] == "Covariance"
        assert summary['Source'] == "test"

    def test_correlation_coefficients(self):
        data = numpy.random.random((13, 14))
        ts = time_series.TimeSeries(data=data, title="test-ts")
        dt = graph.CorrelationCoefficients(source=ts, array_data=data)

        assert dt.array_data.shape == (13, 14)

        summary = dt.summary_info()
        assert summary['Graph type'] == "CorrelationCoefficients"
        assert summary['Source'] == "test-ts"
        assert summary['Dimensions'] == dt.labels_ordering

        assert dt.labels_ordering == ("Node", "Node", "State Variable", "Mode")

    def test_connectivitymeasure(self):
        conn = connectivity.Connectivity()
        dt = graph.ConnectivityMeasure(connectivity=conn, array_data = numpy.array([]))
        assert dt.array_data is not None
        assert dt.connectivity is not None
        summary = dt.summary_info()
        assert summary['Graph type'] == 'ConnectivityMeasure'
