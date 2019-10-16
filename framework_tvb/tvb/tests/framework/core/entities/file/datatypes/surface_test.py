import numpy
import pytest
from tvb.core.entities.file.datatypes.surface_h5 import SurfaceH5
from tvb.datatypes.surfaces import Surface


def test_store_load_configured_surf(tmph5factory, surfaceFactory):
    surface = surfaceFactory(5)
    surface.configure()
    assert surface.number_of_vertices == 5

    tmp_path = tmph5factory()

    with SurfaceH5(tmp_path) as f:
        f.store(surface)

    surf_stored = Surface()

    with SurfaceH5(tmp_path) as f:
        f.load_into(surf_stored)
        assert surf_stored.split_slices['0']['triangles']['start_idx'] == 0


def test_stored_conn_load_vertices_slice(tmph5factory, surfaceFactory):
    surface = surfaceFactory(5)
    surface.configure()
    tmp_path = tmph5factory()

    with SurfaceH5(tmp_path) as f:
        f.store(surface)
        a = f.get_vertices_slice(0)
        numpy.testing.assert_array_equal(a, numpy.zeros((5, 3)))


def test_store_surface(tmph5factory, surfaceFactory):
    surface = surfaceFactory(5)
    surf_h5 = SurfaceH5(tmph5factory())
    surf_h5.store(surface)
    surf_h5.close()


def test_store_load_surface(tmph5factory, surfaceFactory):
    surface = surfaceFactory(5)
    surf_h5 = SurfaceH5(tmph5factory())
    surf_h5.store(surface)
    surf_h5.close()

    surf_stored = Surface()
    with pytest.raises(AttributeError):
        surf_stored.vertices
    surf_h5.load_into(surf_stored)
    assert surf_stored.vertices.shape[0] == 5