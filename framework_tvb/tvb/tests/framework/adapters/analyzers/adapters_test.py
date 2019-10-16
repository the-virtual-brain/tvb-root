import os
from tvb.adapters.analyzers.ica_adapter import ICAAdapter
from tvb.adapters.analyzers.pca_adapter import PCAAdapter
from tvb.adapters.analyzers.wavelet_adapter import ContinuousWaveletTransformAdapter
from tvb.core.entities.file.datatypes.mode_decompositions_h5 import PrincipalComponentsH5, IndependentComponentsH5
from tvb.core.entities.file.datatypes.spectral_h5 import WaveletCoefficientsH5
from tvb.core.entities.model.datatypes.time_series import TimeSeriesIndex
from tvb.interfaces.neocom._h5loader import DirLoader
from tvb.tests.framework.adapters.analyzers.fft_test import make_ts


def test_wavelet_adapter(tmpdir, session):
    loader = DirLoader(str(tmpdir))

    two_node_simple_sin_ts = make_ts()
    loader.store(two_node_simple_sin_ts)

    ts_db = TimeSeriesIndex()
    ts_db.fill_from_has_traits(two_node_simple_sin_ts)

    session.add(ts_db)
    session.commit()

    wavelet_adapter = ContinuousWaveletTransformAdapter()
    wavelet_adapter.storage_path = str(tmpdir)

    res = session.query(TimeSeriesIndex)
    ts_index = res[0]
    wavelet_adapter.configure(ts_index)

    diskq = wavelet_adapter.get_required_disk_size()
    memq = wavelet_adapter.get_required_memory_size()

    wavelet_idx = wavelet_adapter.launch(ts_index)

    result_h5 = loader.path_for(WaveletCoefficientsH5, wavelet_idx.gid)
    assert os.path.exists(result_h5)


def test_pca_adapter(tmpdir, session):
    loader = DirLoader(str(tmpdir))

    # Input for Adapter
    two_node_simple_sin_ts = make_ts()
    loader.store(two_node_simple_sin_ts)

    ts_db = TimeSeriesIndex()
    ts_db.fill_from_has_traits(two_node_simple_sin_ts)

    session.add(ts_db)
    session.commit()

    pca_adapter = PCAAdapter()
    pca_adapter.storage_path = str(tmpdir)

    res = session.query(TimeSeriesIndex)
    ts_index = res[0]
    pca_adapter.configure(ts_index)

    disk = pca_adapter.get_required_disk_size()
    mem = pca_adapter.get_required_memory_size()

    pca_idx = pca_adapter.launch()

    result_h5 = loader.path_for(PrincipalComponentsH5, pca_idx.gid)
    assert os.path.exists(result_h5)


def test_ica_adapter(tmpdir, session):
    loader = DirLoader(str(tmpdir))

    two_node_simple_sin_ts = make_ts()
    loader.store(two_node_simple_sin_ts)

    ts_db = TimeSeriesIndex()
    ts_db.fill_from_has_traits(two_node_simple_sin_ts)

    session.add(ts_db)
    session.commit()

    ica_adapter = ICAAdapter()
    ica_adapter.storage_path = str(tmpdir)

    res = session.query(TimeSeriesIndex)
    ts_index = res[0]
    ica_adapter.configure(ts_index)

    disk = ica_adapter.get_required_disk_size()
    mem = ica_adapter.get_required_memory_size()

    ica_idx = ica_adapter.launch()

    result_h5 = loader.path_for(IndependentComponentsH5, ica_idx.gid)
    assert os.path.exists(result_h5)
