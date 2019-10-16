from tvb.core.entities.model.datatypes.connectivity import ConnectivityIndex
from tvb.core.entities.model.datatypes.region_mapping import RegionMappingIndex
from tvb.core.entities.model.datatypes.surface import SurfaceIndex
from tvb.tests.framework.core.entities.file.datatypes.testdatatypes import connectivity, surface, region_mapping


def test_store_load_region_mapping(session):
    conn_idx = ConnectivityIndex()
    conn_idx.fill_from_has_traits(connectivity)
    session.add(conn_idx)

    surf_idx = SurfaceIndex()
    surf_idx.fill_from_has_traits(surface)
    session.add(surf_idx)

    rm_idx = RegionMappingIndex()
    rm_idx.fill_from_has_traits(region_mapping)
    rm_idx.connectivity = conn_idx
    rm_idx.surface = surf_idx
    session.add(rm_idx)

    session.commit()

    res = session.query(ConnectivityIndex)
    assert res.count() == 1
    assert res[0].number_of_regions == 2
    assert res[0].number_of_connections == 4
    assert res[0].undirected is False
    assert res[0].weights.ndim == 2
    assert res[0].weights.length_1d == 2
    assert res[0].weights.length_2d == 2
    assert res[0].weights.length_3d is None

    res = session.query(SurfaceIndex)
    assert res.count() == 1

    res = session.query(RegionMappingIndex)
    assert res.count() == 1
