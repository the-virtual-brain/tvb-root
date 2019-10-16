import numpy
from tvb.core.entities.file.datatypes.surface_h5 import SurfaceH5
from tvb.datatypes.surfaces import Surface
from .import testdatatypes


def test_store_load_configured_surf(tmph5factory):
    surface = Surface(
        vertices=numpy.zeros((5, 3)),
        triangles=numpy.zeros((3, 3), dtype=int),
        vertex_normals=numpy.zeros((5, 3)),
        triangle_normals=numpy.zeros((3, 3)),
        zero_based_triangles=False,
        surface_type="surface",
        valid_for_simulations=True
    )
    surface.configure()
    assert surface.number_of_vertices == 5

    tmp_path = tmph5factory()

    with SurfaceH5(tmp_path) as f:
        f.store(surface)

    surf_stored = Surface()

    with SurfaceH5(tmp_path) as f:
        f.load_into(surf_stored)
        assert surf_stored.split_slices['0']['triangles']['start_idx'] == 0


def test_stored_conn_load_vertices_slice(tmph5factory):
    surface = Surface(
        vertices=numpy.zeros((5, 3)),
        triangles=numpy.zeros((3, 3), dtype=int),
        vertex_normals=numpy.zeros((5, 3)),
        triangle_normals=numpy.zeros((3, 3)),
        zero_based_triangles=False,
        surface_type="surface",
        valid_for_simulations=True
    )
    surface.configure()
    tmp_path = tmph5factory()

    with SurfaceH5(tmp_path) as f:
        f.store(surface)
        a = f.get_vertices_slice(0)
        numpy.testing.assert_array_equal(a, numpy.zeros((5, 3)))


def test_store_surface(tmph5factory):
    surf_h5 = SurfaceH5(tmph5factory())
    surf_h5.store(testdatatypes.surface)
    surf_h5.close()


def test_store_load_surface(tmph5factory):
    surf_h5 = SurfaceH5(tmph5factory())
    surf_h5.store(testdatatypes.surface)
    surf_h5.close()

    surf_stored = Surface()
    assert surf_stored.vertices is None
    # with pytest.raises(TypeError):
    surf_h5.load_into(surf_stored)
    assert surf_stored.vertices.shape[0] == 5