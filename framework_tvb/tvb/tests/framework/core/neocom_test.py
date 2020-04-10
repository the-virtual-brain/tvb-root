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
import os
import numpy
from tvb.core.neocom.h5 import load, store, load_from_dir, store_to_dir


def test_store_load(tmpdir, connectivity_factory):
    path = os.path.join(str(tmpdir), 'interface.conn.h5')
    connectivity = connectivity_factory(2)
    store(connectivity, path)
    con2 = load(path)
    numpy.testing.assert_equal(connectivity.weights, con2.weights)


def test_store_load_rec(tmpdir, connectivity_factory, region_mapping_factory):
    connectivity = connectivity_factory(2)
    region_mapping = region_mapping_factory(connectivity=connectivity)
    store_to_dir(region_mapping, str(tmpdir), recursive=True)

    rmap = load_from_dir(str(tmpdir), region_mapping.gid, recursive=True)
    numpy.testing.assert_equal(connectivity.weights, rmap.connectivity.weights)
