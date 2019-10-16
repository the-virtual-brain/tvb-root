import numpy
from tvb.datatypes.region_mapping import RegionMapping
from tvb.core.entities.file.datatypes.connectivity_h5 import ConnectivityH5
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.surfaces import Surface, CorticalSurface
from tvb.core.entities.file.datatypes.region_mapping_h5 import RegionMappingH5
from tvb.core.entities.file.datatypes.surface_h5 import SurfaceH5

conn = Connectivity(
    region_labels=numpy.array(["a", "b"]),
    weights=numpy.zeros((2, 2)),
    # undirected=False,
    tract_lengths=numpy.zeros((2, 2)),
    centres=numpy.zeros((2, 2)),
    cortical=numpy.array([[True, True], [True, True]]),
    hemispheres=numpy.array([[True, True], [True, True]]),
    orientations=numpy.zeros((2, 2)),
    areas=numpy.zeros((2, 2)),
    number_of_regions=2,
    number_of_connections=4,
    # parent_connectivity=""
)

surface = Surface(
    vertices=numpy.zeros((5, 3)),
    triangles=numpy.zeros((3, 3), dtype=int),
    vertex_normals=numpy.zeros((5, 3)),
    triangle_normals=numpy.zeros((3, 3)),
    number_of_vertices=5,
    number_of_triangles=3,
    edge_mean_length=1.0,
    edge_min_length=0.0,
    edge_max_length=2.0,
    zero_based_triangles=False,
    split_triangles=numpy.arange(0),
    number_of_split_slices=1,
    split_slices=dict(),
    bi_hemispheric=False,
    surface_type="surface",
    valid_for_simulations=True
)

rm = RegionMapping(
    array_data=numpy.arange(5),
    connectivity=conn,
    surface=surface
)

cort = CorticalSurface(
    vertices=numpy.zeros((5, 3)),
    triangles=numpy.zeros((3, 3), dtype=int),
    vertex_normals=numpy.zeros((5, 3)),
    triangle_normals=numpy.zeros((3, 3)),
    number_of_vertices=5,
    number_of_triangles=3,
    edge_mean_length=1.0,
    edge_min_length=0.0,
    edge_max_length=2.0,
    zero_based_triangles=False,
    split_triangles=numpy.arange(0),
    number_of_split_slices=1,
    split_slices=dict(),
    bi_hemispheric=False,
    # surface_type="surface",
    valid_for_simulations=True
)


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


def test_store_connectivity(tmph5factory):
    conn_h5 = ConnectivityH5(tmph5factory())
    conn_h5.store(conn)
    conn_h5.close()


def test_store_load_connectivity(tmph5factory):
    conn_h5 = ConnectivityH5(tmph5factory())
    conn_h5.store(conn)
    conn_h5.close()

    conn_stored = Connectivity()
    assert conn_stored.region_labels is None
    # Long is stored in H5 as int64 => fails to set value on traited attr with type long
    # with pytest.raises(TypeError):
    conn_h5.load_into(conn_stored)
    assert conn_stored.region_labels.shape[0] == 2


def test_store_partial_connectivity(tmph5factory):
    # Fails at _validate_set if dataset (flagged as required=False) is None
    partial_conn = Connectivity(region_labels=numpy.array(["a", "b"]),
                                weights=numpy.zeros((2, 2)),
                                # undirected=False,
                                tract_lengths=numpy.zeros((2, 2)),
                                centres=numpy.zeros((2, 2)),
                                # cortical=numpy.array([[True, True], [True, True]]),
                                # hemispheres=numpy.array([[True, True], [True, True]]),
                                # orientations=numpy.zeros((2, 2)),
                                # areas=numpy.zeros((2, 2)),
                                number_of_regions=long(2),
                                number_of_connections=long(4),
                                # parent_connectivity=""
                                )
    conn_h5 = ConnectivityH5(tmph5factory())
    # with pytest.raises(AttributeError):
    conn_h5.store(partial_conn)
    conn_h5.close()


def test_store_surface(tmph5factory):
    surf_h5 = SurfaceH5(tmph5factory())
    surf_h5.store(surface)
    surf_h5.close()


def test_store_load_surface(tmph5factory):
    surf_h5 = SurfaceH5(tmph5factory())
    surf_h5.store(surface)
    surf_h5.close()

    surf_stored = Surface()
    assert surf_stored.vertices is None
    # with pytest.raises(TypeError):
    surf_h5.load_into(surf_stored)
    assert surf_stored.vertices.shape[0] == 5


def test_store_load_region_mapping(tmph5factory):
    rm_h5 = RegionMappingH5(tmph5factory())
    rm_h5.store(rm)
    rm_h5.close()

    rm_stored = RegionMapping()
    assert rm_stored.array_data is None
    rm_h5.load_into(rm_stored)  # loads connectivity/surface as None inside rm_stored
    assert rm_stored.array_data.shape == (5,)


def test_store_load_complete_region_mapping(tmph5factory):
    conn_h5 = ConnectivityH5(tmph5factory('Connectivity_{}.h5'.format(conn.gid)))
    surf_h5 = SurfaceH5(tmph5factory('Surface_{}.h5'.format(surface.gid)))
    rm_h5 = RegionMappingH5(tmph5factory('RegionMapping_{}.h5'.format(rm.gid)))

    conn_h5.store(conn)
    conn_h5.close()  # use with

    surf_h5.store(surface)
    surf_h5.close()

    rm_h5.store(rm)
    rm_h5.close()

    # conn_stored = Connectivity()
    # surf_stored = Surface()
    # rm_stored = RegionMapping()
    #
    # conn_h5.load_into(conn_stored)
    # surf_h5.load_into(surf_stored)
    # rm_h5.load_into(rm_stored)
    # assert rm_stored.connectivity is None
    # assert rm_stored.surface is None
    # rm_stored.connectivity = conn_stored
    # rm_stored.surface = surf_stored
    # assert rm_stored.connectivity is not None
    # assert rm_stored.surface is not None
