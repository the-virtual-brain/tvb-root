import numpy
from tvb.core.entities.file.datatypes.time_series import TimeSeriesH5
from tvb.datatypes.time_series import TimeSeries


ntime, nspace, nsv = 120, 4, 2


def make_harmonic_ts():
    return TimeSeries(
        title='harmonic',
        labels_ordering=('time', 'statevar', 'space'),
        labels_dimensions={'statevar': ['position', 'speed']},
        start_time=0.0,
        sample_period=0.5,
        sample_rate=2.0,
    )


def harmonic_chunk(time):
    data = numpy.zeros((time.size, nsv, nspace))

    for s in range(nspace):
        data[:, 0, s] = numpy.sin(s*time)
        data[:, 1, s] = numpy.cos(s*time)

    return data


def test_streaming_writes(tmph5factory):
    nchunks = 4
    t = make_harmonic_ts()
    # t.configure will fail for new files, it wants to read the data shape!
    path = tmph5factory()

    with TimeSeriesH5(path) as f:
        # when doing a partial write we still should populate the fields that we can
        # we aim to have the file in the right format even after a chunk write
        # This is similar to super.store but for specific scalars
        f.store(t, scalars_only=True)

        # todo: refuse write_data_slice unless required metadata has been written
        # otherwise we create files that will not conform to the timeseriesh5 format!

        for chunk_id in range(nchunks):
            time = numpy.linspace(chunk_id, chunk_id + 33, ntime, endpoint=False)
            data = harmonic_chunk(time)
            f.write_data_slice(data)


def test_streaming_reads(tmph5factory):
    t = make_harmonic_ts()
    path = tmph5factory()

    with TimeSeriesH5(path) as f:
        time = numpy.linspace(0, 33, ntime)
        data = harmonic_chunk(time)
        f.store(t, scalars_only=True)
        f.write_data_slice(data)

    with TimeSeriesH5(path) as f:
        data = f.read_data_slice((slice(0, 33), slice(None), 0))
        expected = numpy.zeros((33, nsv))
        expected[:, 1] = 1.0   # the cos(0) part
        numpy.testing.assert_array_equal(data, expected)
