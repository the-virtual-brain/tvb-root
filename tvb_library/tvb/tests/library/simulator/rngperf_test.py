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

import numpy as np
import pytest

BitGens = [
    np.random.PCG64,
    np.random.Philox,
    np.random.MT19937,
    np.random.SFC64,
]

@pytest.mark.parametrize('BitGen', BitGens)
def test_raw(benchmark, BitGen):
    bg = BitGen(42)
    benchmark(lambda: bg.random_raw(size=1024, output=False))

@pytest.mark.parametrize('BitGen', BitGens)
def test_uniform(benchmark, BitGen):
    bg = BitGen(42)
    rng = np.random.Generator(bg)
    out = np.empty((1024,), 'f')
    benchmark(lambda: rng.random(out.shape, np.float32, out))

@pytest.mark.parametrize('BitGen', BitGens)
def test_normal(benchmark, BitGen):
    bg = BitGen(42)
    rng = np.random.Generator(bg)
    benchmark(lambda: rng.normal(0.0, 1.0, 1024))

