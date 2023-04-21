# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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
Created on Mar 20, 2013

.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import numpy
from tvb.tests.library.base_testcase import BaseTestCase
from tvb.datatypes import mode_decompositions, time_series


class TestModeDecompositions(BaseTestCase):
    """
    Tests the defaults for `tvb.datatypes.mode_decompositions` module.
    """

    def test_principalcomponents(self):
        data = numpy.random.random((10, 10, 10, 10))
        ts = time_series.TimeSeries(data=data)
        dt = mode_decompositions.PrincipalComponents(source=ts,
                                                     fractions=numpy.random.random((10, 10, 10)),
                                                     weights=data)
        dt.compute_norm_source()
        dt.compute_component_time_series()
        dt.compute_normalised_component_time_series()
        dt.configure()
        summary = dt.summary_info()
        assert summary['Mode decomposition type'] == 'PrincipalComponents'
        assert dt.source is not None
        assert dt.weights.shape == (10, 10, 10, 10)
        assert dt.fractions.shape == (10, 10, 10)
        assert dt.norm_source.shape == (10, 10, 10, 10)
        assert dt.component_time_series.shape == (10, 10, 10, 10)
        assert dt.normalised_component_time_series.shape == (10, 10, 10, 10)

    def test_independentcomponents(self):
        data = numpy.random.random((10, 10, 10, 10))
        ts = time_series.TimeSeries(data=data)
        n_comp = 5
        dt = mode_decompositions.IndependentComponents(source=ts,
                                                       component_time_series=numpy.random.random((10, n_comp, 10, 10)),
                                                       prewhitening_matrix=numpy.random.random((n_comp, 10, 10, 10)),
                                                       unmixing_matrix=numpy.random.random((n_comp, n_comp, 10, 10)),
                                                       n_components=n_comp)
        dt.compute_norm_source()
        dt.compute_component_time_series()
        dt.compute_normalised_component_time_series()
        summary = dt.summary_info()
        assert summary['Mode decomposition type'] == 'IndependentComponents'
        assert dt.source is not None
        assert dt.mixing_matrix is None
        assert dt.unmixing_matrix.shape == (n_comp, n_comp, 10, 10)
        assert dt.prewhitening_matrix.shape == (n_comp, 10, 10, 10)
        assert dt.norm_source.shape == (10, 10, 10, 10)
        assert dt.component_time_series.shape == (10, 10, n_comp, 10)
