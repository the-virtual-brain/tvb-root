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
import numpy
import pytest
from tvb.basic.neotraits.ex import TraitAttributeError
from tvb.adapters.datatypes.h5.connectivity_h5 import ConnectivityH5
from tvb.datatypes.connectivity import Connectivity


def test_store_connectivity(tmph5factory, connectivity_factory):
    connectivity = connectivity_factory(2)
    conn_h5 = ConnectivityH5(tmph5factory())
    conn_h5.store(connectivity)
    conn_h5.close()


def test_store_load_connectivity(tmph5factory, connectivity_factory):
    connectivity = connectivity_factory(2)
    conn_h5 = ConnectivityH5(tmph5factory())
    conn_h5.store(connectivity)
    conn_h5.close()

    conn_stored = Connectivity()
    with pytest.raises(TraitAttributeError):
        conn_stored.region_labels
    conn_h5.load_into(conn_stored)
    assert conn_stored.region_labels.shape[0] == 2


def test_store_partial_connectivity(tmph5factory):
    partial_conn = Connectivity(
        region_labels=numpy.array(["a", "b"]),
        weights=numpy.zeros((2, 2)),
        tract_lengths=numpy.zeros((2, 2)),
        centres=numpy.zeros((2, 2)),
        number_of_regions=int(2),
        number_of_connections=int(4),
    )
    conn_h5 = ConnectivityH5(tmph5factory())
    conn_h5.store(partial_conn)
    conn_h5.close()
