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
Test for tvb.simulator.noise module

.. moduleauthor:: Paula Sanz Leon <paula@tvb.invalid>

"""
from tvb.tests.library.base_testcase import BaseTestCase
from tvb.simulator import noise
from tvb.datatypes import equations


class TestNoise(BaseTestCase):
    def test_additive(self):
        noise_additive = noise.Additive()
        assert noise_additive.ntau == 0.0

    def test_multiplicative(self):
        noise_multiplicative = noise.Multiplicative()
        assert noise_multiplicative.ntau == 0.0
        assert isinstance(noise_multiplicative.b, equations.Linear)
