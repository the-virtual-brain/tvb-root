import os
import numpy
from tvb.core.neocom.h5 import load, store, load_from_dir, store_to_dir


def test_store_load(tmpdir, connectivityFactory):
    path = os.path.join(str(tmpdir), 'interface.conn.h5')
    connectivity = connectivityFactory(2)
    store(connectivity, path)
    con2 = load(path)
    numpy.testing.assert_equal(connectivity.weights, con2.weights)


def test_store_load_rec(tmpdir, connectivityFactory, regionMappingFactory):
    connectivity = connectivityFactory(2)
    region_mapping = regionMappingFactory(connectivity=connectivity)
    store_to_dir(str(tmpdir), region_mapping, recursive=True)

    rmap = load_from_dir(str(tmpdir), region_mapping.gid, recursive=True)
    numpy.testing.assert_equal(connectivity.weights, rmap.connectivity.weights)
