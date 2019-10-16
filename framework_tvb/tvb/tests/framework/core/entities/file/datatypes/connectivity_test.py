import numpy
from tvb.core.entities.file.datatypes.connectivity_h5 import ConnectivityH5
from tvb.datatypes.connectivity import Connectivity
from tvb.tests.framework.core.entities.file.datatypes.testdatatypes import connectivity


def test_store_connectivity(tmph5factory):
    conn_h5 = ConnectivityH5(tmph5factory())
    conn_h5.store(connectivity)
    conn_h5.close()


def test_store_load_connectivity(tmph5factory):
    conn_h5 = ConnectivityH5(tmph5factory())
    conn_h5.store(connectivity)
    conn_h5.close()

    conn_stored = Connectivity()
    assert conn_stored.region_labels is None
    # Long is stored in H5 as int64 => fails to set value on traited attr with type long
    # with pytest.raises(TypeError):
    conn_h5.load_into(conn_stored)
    assert conn_stored.region_labels.shape[0] == 2


def test_store_partial_connectivity(tmph5factory):
    # Fails at _validate_set if dataset (flagged as required=False) is None
    partial_conn = Connectivity(
        region_labels=numpy.array(["a", "b"]),
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
