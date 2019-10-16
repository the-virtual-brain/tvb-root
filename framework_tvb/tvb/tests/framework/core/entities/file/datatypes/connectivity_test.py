import numpy
import pytest
from tvb.basic.neotraits.ex import TraitAttributeError
from tvb.core.entities.file.datatypes.connectivity_h5 import ConnectivityH5
from tvb.datatypes.connectivity import Connectivity


def test_store_connectivity(tmph5factory, connectivityFactory):
    connectivity = connectivityFactory(2)
    conn_h5 = ConnectivityH5(tmph5factory())
    conn_h5.store(connectivity)
    conn_h5.close()


def test_store_load_connectivity(tmph5factory, connectivityFactory):
    connectivity = connectivityFactory(2)
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
