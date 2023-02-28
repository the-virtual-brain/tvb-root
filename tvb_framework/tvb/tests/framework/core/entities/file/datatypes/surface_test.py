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
from tvb.adapters.datatypes.h5.surface_h5 import SurfaceH5
from tvb.datatypes.surfaces import Surface


def test_store_load_configured_surf(tmph5factory, surface_factory):
    surface = surface_factory(5)
    # surface.configure()
    assert surface.number_of_vertices == 5
    assert surface.edge_max_length == 2.0

    tmp_path = tmph5factory()
    with SurfaceH5(tmp_path) as f:
        f.store(surface)

    surf_stored = Surface()
    with SurfaceH5(tmp_path) as f:
        f.load_into(surf_stored)
        assert f.get_number_of_split_slices() == 1
        assert f.get_slice_vertex_boundaries(0) == (0, 5)
    assert surf_stored.number_of_vertices == 5
    assert surf_stored.number_of_triangles == 3
    assert surf_stored.edge_max_length == 2.0


def test_stored_conn_load_vertices_slice(tmph5factory, surface_factory):
    surface = surface_factory(5)
    surface.configure()
    tmp_path = tmph5factory()

    with SurfaceH5(tmp_path) as f:
        f.store(surface)
        a = f.get_vertices_slice(0)
        numpy.testing.assert_array_equal(a, numpy.zeros((5, 3)))


def test_store_surface(tmph5factory, surface_factory):
    surface = surface_factory(5)
    surf_h5 = SurfaceH5(tmph5factory())
    surf_h5.store(surface)
    surf_h5.close()


def test_store_load_surface(tmph5factory, surface_factory):
    surface = surface_factory(5)
    surf_h5 = SurfaceH5(tmph5factory())
    surf_h5.store(surface)
    surf_h5.close()

    surf_stored = Surface()
    with pytest.raises(AttributeError):
        surf_stored.vertices
    surf_h5.load_into(surf_stored)
    assert surf_stored.vertices.shape[0] == 5
