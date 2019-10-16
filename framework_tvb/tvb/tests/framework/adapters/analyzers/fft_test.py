import os
import numpy
from tvb.analyzers.fft import FFT
from tvb.datatypes.time_series import TimeSeries
from tvb.interfaces.neocom.h5 import TimeSeriesH5, store_to_dir, DirLoader
from tvb.interfaces.neocom.db import TimeSeriesIndex
from tvb.adapters.analyzers.fourier_adapter import FourierAdapter


def make_ts():
    # 1 second time series, 4000 samples
    # channel 0: a pure 40hz
    # channel 1: a pure 200hz
    # channel 2: a superposition of 100 and 300Hz equal amplitude
    time = numpy.linspace(0, 1000, 4000)
    data = numpy.zeros((time.size, 1, 3, 1))
    data[:, 0, 0, 0] = numpy.sin(2 * numpy.pi * time/1000.0 * 40)
    data[:, 0, 1, 0] = numpy.sin(2 * numpy.pi * time/1000.0 * 200)
    data[:, 0, 2, 0] = numpy.sin(2 * numpy.pi * time/1000.0 * 100) + \
                       numpy.sin(2 * numpy.pi * time/1000.0 * 300)

    return TimeSeries(time=time, data=data, sample_rate=4000.0, sample_period=1.0/4000)


def test_fourier_analyser():
    two_node_simple_sin_ts = make_ts()

    fft = FFT(time_series=two_node_simple_sin_ts, segment_length=4000.0)
    spectra = fft.evaluate()

    # a poor man's peak detection, a box low-pass filter
    peak = 40
    around_peak = spectra.array_data[peak - 10: peak + 10, 0, 0, 0].real
    assert numpy.abs(around_peak).sum() > 0.5 * 20

    peak = 80  # not an expected peak
    around_peak = spectra.array_data[peak - 10: peak + 10, 0, 0, 0].real
    assert numpy.abs(around_peak).sum() < 0.5 * 20



def test_fourier_adapter(tmpdir, session):
    loader = DirLoader(str(tmpdir))
    # make file stored and indexed time series

    two_node_simple_sin_ts = make_ts()

    loader.store(two_node_simple_sin_ts)

    ts_db = TimeSeriesIndex()
    ts_db.fill_from_has_traits(two_node_simple_sin_ts)

    session.add(ts_db)
    session.commit()

    # we have the required data to start the adapter
    # REVIEW THIS
    # The adapter methods require the shape of the full time series
    # As we do not want to load the whole time series we can not send the
    # datatype. What the adapter actually wants is metadata about the datatype
    # on disk. So the database entity is a good input.
    # But this contradicts the convention that adapters inputs are datatypes
    # Note that the previous functionality relied on the fact that datatypes
    # knew directly how to interact with storage via get_data_shape
    # Another interpretation is that the adapter really wants a data type
    # weather it is a in memory HasTraits or a H5File to access it.
    # Here we consider this last option.

    adapter = FourierAdapter()
    adapter.storage_path = str(tmpdir)

    # todo: this is a very awkward api.
    # Where will the DirLoader expect to find a time series with this gid
    # We want to open the file ourselves and need to know where did the loader put it
    ts_pth = loader.path_for(TimeSeriesH5, ts_db.gid)

    with TimeSeriesH5(ts_pth) as ts_file:

        adapter.configure(ts_file, segment_length=4000)

        diskq = adapter.get_required_disk_size(ts_file, segment_length=4000)
        memq = adapter.get_required_memory_size(ts_file, segment_length=4000)

        spectra_file = adapter.launch(ts_file, segment_length=4000)

        # REVIEW THIS
        # the same dilemma that we faced for inputs we face for outputs
        # Here the same choice was made.
        # The adapter returns a FourierSpectrumH5 instance

    with spectra_file:
        assert spectra_file.array_data.shape == (2000, 1, 3, 1, 1)
        assert spectra_file.source.load() == two_node_simple_sin_ts.gid
        assert spectra_file.path.endswith(spectra_file.gid.load().hex + '.h5')
