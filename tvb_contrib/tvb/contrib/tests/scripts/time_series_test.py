# coding=utf-8
#
#
#  TheVirtualBrain-Contributors Package. This package holds simulator extensions.
#  See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

"""
.. moduleauthor:: Dionysios Perdikis <Denis@tvb.invalid>
"""

import numpy
import pytest
from tvb.contrib.scripts.datatypes.time_series import TimeSeries, TimeSeriesDimensions, PossibleVariables
from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeries as TimeSeriesXarray
from tvb.simulator.plot.config import FiguresConfig


class TestTimeSeries:
    datatypes_to_test = [TimeSeries, TimeSeriesXarray]

    def _prepare_dummy_time_series(self, dim):
        if dim == 1:
            data = numpy.array([1, 2, 3, 4, 5])
        elif dim == 2:
            data = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])[:, numpy.newaxis, :]
        elif dim == 3:
            data = numpy.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 1, 2]],
                                [[3, 4, 5], [6, 7, 8], [9, 0, 1], [2, 3, 4]],
                                [[5, 6, 7], [8, 9, 0], [1, 2, 3], [4, 5, 6]]])
        else:
            data = numpy.array([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2]],
                                 [[3, 4, 5, 6], [7, 8, 9, 0], [1, 2, 3, 4]],
                                 [[5, 6, 7, 8], [9, 0, 1, 2], [3, 4, 5, 6]],
                                 [[7, 8, 9, 0], [1, 2, 3, 4], [5, 6, 7, 8]]],
                                [[[9, 0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 0]],
                                 [[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2]],
                                 [[3, 4, 5, 6], [7, 8, 9, 0], [1, 2, 3, 4]],
                                 [[5, 6, 7, 8], [9, 0, 1, 2], [3, 4, 5, 6]]],
                                [[[7, 8, 9, 0], [1, 2, 3, 4], [5, 6, 7, 8]],
                                 [[9, 0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 0]],
                                 [[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2]],
                                 [[3, 4, 5, 6], [7, 8, 9, 0], [1, 2, 3, 4]]]]).reshape((3, 3, 4, 4))
        start_time = 0
        sample_period = 0.01
        sample_period_unit = "ms"

        return data, start_time, sample_period, sample_period_unit

    @pytest.mark.parametrize('datatype', datatypes_to_test)
    def test_timeseries_1D_definition(self, datatype):
        data, start_time, sample_period, sample_period_unit = self._prepare_dummy_time_series(1)
        with pytest.raises(ValueError):
            datatype(data=data, labels_dimensions={}, start_time=start_time,
                     sample_period=sample_period, sample_period_unit=sample_period_unit)

    @pytest.mark.parametrize('datatype', datatypes_to_test)
    def test_timeseries_2D(self, datatype):
        data, start_time, sample_period, sample_period_unit = self._prepare_dummy_time_series(2)
        ts_from_2D = datatype(data=data, labels_dimensions={TimeSeriesDimensions.SPACE.value: ["r1", "r2", "r3"]},
                              start_time=start_time, sample_period=sample_period,
                              sample_period_unit=sample_period_unit)
        ts_from_2D.configure()
        assert ts_from_2D.data.ndim == 4
        assert ts_from_2D.data.shape == (3, 1, 3, 1)

        assert ts_from_2D.end_time == 0.02
        assert all(ts_from_2D.time == numpy.array([0, 0.01, 0.02]))

        with pytest.raises(KeyError):
            ts_from_2D.get_state_variables_by_label("")

        ts_r2r3 = ts_from_2D.get_subspace_by_label(["r2", "r3"])
        assert ts_r2r3.data.ndim == 4
        assert ts_r2r3.data.shape == (3, 1, 2, 1)

        ts_r2 = ts_from_2D.get_subspace_by_label(["r2"])
        assert ts_r2.data.ndim == 4
        assert ts_r2.data.shape == (3, 1, 1, 1)
        assert ts_r2.labels_dimensions[TimeSeriesDimensions.SPACE.value] == ["r2"]
        assert ts_r2.get_subspace_by_label(["r2"]).labels_dimensions[TimeSeriesDimensions.SPACE.value] == ["r2"]

        with pytest.raises((ValueError, KeyError)):
            ts_r2.get_subspace_by_label(["r1"])

        ts_r2r3_idx = ts_from_2D.get_subspace_by_index([1, 2])
        assert ts_r2r3_idx.data.ndim == 4
        assert ts_r2r3_idx.data.shape == (3, 1, 2, 1)

        ts_r2_idx = ts_r2r3_idx.get_subspace_by_index([0])
        assert ts_r2_idx.data.ndim == 4
        assert ts_r2_idx.data.shape == (3, 1, 1, 1)
        assert ts_r2_idx.labels_dimensions[TimeSeriesDimensions.SPACE.value] == ["r2"]
        assert ts_r2_idx.get_subspace_by_index([0]).labels_dimensions[TimeSeriesDimensions.SPACE.value] == ["r2"]

        with pytest.raises((IndexError, KeyError)):
            ts_r2_idx.get_subspace_by_index([2])

        assert ts_r2r3_idx.data.all() == ts_r2r3.data.all()
        assert all(ts_r2_idx.data == ts_r2.data)

        ts_time_window = ts_from_2D.get_time_window(1, 2)
        assert ts_time_window.data.ndim == 4
        assert ts_time_window.data.shape == (1, 1, 3, 1)
        assert numpy.array_equal(ts_time_window.labels_dimensions[TimeSeriesDimensions.SPACE.value], ["r1", "r2", "r3"])
        assert ts_time_window.start_time == 0.01

        ts_time_window_units = ts_from_2D.get_time_window_by_units(0.01, 0.02)
        assert ts_time_window_units.data.ndim == 4
        assert ts_time_window_units.data.shape == (1, 1, 3, 1)
        assert numpy.array_equal(ts_time_window_units.labels_dimensions[TimeSeriesDimensions.SPACE.value],
                                 ["r1", "r2", "r3"])
        assert ts_time_window_units.start_time == 0.01

        with pytest.raises(IndexError):
            ts_from_2D.get_time_window(2, 4)

        with pytest.raises(IndexError):
            ts_from_2D.get_time_window_by_units(0, 0.025)

        with pytest.raises(AttributeError):
            ts_from_2D.lfp

    @pytest.mark.parametrize('datatype', datatypes_to_test)
    def test_timeseries_3D(self, datatype):
        data, start_time, sample_period, sample_period_unit = self._prepare_dummy_time_series(3)
        ts_3D = datatype(data=data,
                         labels_dimensions={TimeSeriesDimensions.SPACE.value: ["r1", "r2", "r3"],
                                            TimeSeriesDimensions.VARIABLES.value: ['sv1', 'sv2', 'sv3', 'sv4']},
                         start_time=start_time, sample_period=sample_period,
                         sample_period_unit=sample_period_unit)
        assert ts_3D.data.ndim == 4
        assert ts_3D.data.shape[3] == 1

    @pytest.mark.parametrize('datatype', datatypes_to_test)
    def test_timeseries_data_access(self, datatype):
        data, start_time, sample_period, sample_period_unit = self._prepare_dummy_time_series(3)
        ts = datatype(data=data,
                      labels_dimensions={TimeSeriesDimensions.SPACE.value: ["r1", "r2", "r3", ],
                                         TimeSeriesDimensions.VARIABLES.value: ["sv1", "sv2", "sv3", "sv4"]},
                      start_time=start_time, sample_period=sample_period,
                      sample_period_unit=sample_period_unit)
        assert isinstance(ts.r1, datatype)
        assert ts.r1.data.shape == (3, 4, 1, 1)

        assert isinstance(ts.sv1, datatype)
        assert ts.sv1.data.shape == (3, 1, 3, 1)

        with pytest.raises(AttributeError):
            ts.r9

        with pytest.raises(AttributeError):
            ts.sv0

        assert ts[:, :, :, :].shape == ts.data.shape
        assert ts[1:, :, :, :].shape == ts.data[1:, :, :, :].shape
        assert ts[1:2, :, :, :].shape == ts.data[1:2, :, :, :].shape

        assert ts[:, 1:, :, :].shape == ts.data[:, 1:, :, :].shape
        assert ts[:, :1, :, :].shape == ts.data[:, :1, :, :].shape
        assert ts[:, 1:3, :, :].shape == ts.data[:, 1:3, :, :].shape

        assert ts[:, :, "r2":, :].shape == ts.data[:, :, 1:, :].shape
        assert ts[:, :, :"r2", :].shape == ts.data[:, :, :2, :].shape
        assert ts[:, :, "r2":"r3", :].shape == ts.data[:, :, 1:3, :].shape
        assert ts[1:2, :, "r2":"r3", :].shape == ts.data[1:2, :, 1:3, :].shape

        assert ts[:, :, 1:, :].shape == ts.data[:, :, 1:, :].shape
        assert ts[:, :, :1, :].shape == ts.data[:, :, :1, :].shape
        assert ts[:, :, 0:2, :].shape == ts.data[:, :, 0:2, :].shape

        assert ts[:, "sv2":, :, :].shape == ts.data[:, 1:, :, :].shape
        assert ts[:, :"sv2", :, :].shape == ts.data[:, :2, :, :].shape
        assert ts[:, "sv1":"sv3", :, :].shape == ts.data[:, 0:3, :, :].shape

        assert ts[1:2, "sv2":, :, :].shape == ts.data[1:2, 1:, :, :].shape
        assert ts[1:2, :"sv2", :, :].shape == ts.data[1:2, :2, :, :].shape
        assert ts[1:2, "sv1":"sv3", :, :].shape == ts.data[1:2, 0:3, :, :].shape

        assert numpy.all(ts[0:2, :"sv2", "r1":"r2", :].data == ts.data[0:2, :2, 0:2, :])

        if isinstance(ts, TimeSeries):

            assert ts[1, :, :, :].shape == ts.data[1, :, :, :].shape
            assert ts[:, 1, :, :].shape == ts.data[:, 1, :, :].shape
            assert ts[1, :, "r2":"r3", :].shape == ts.data[1, :, 1:3, :].shape
            assert ts[:, :, 2, :].shape == ts.data[:, :, 2, :].shape
            assert ts[:, "sv3", :, :].shape == ts.data[:, 2, :, :].shape
            assert ts[2, "sv3", :, :].shape == ts.data[2, 2, :, :].shape
            assert ts[2, "sv3", 0:3, :].shape == ts.data[2, 2, 0:3, :].shape
            assert ts[2, "sv3", "r1":"r3", :].shape == ts.data[2, 2, 0:3, :].shape
            assert ts[0:2, "sv3", "r1":"r3", :].shape == ts.data[0:2, 2, 0:3, :].shape
            assert ts[0:2, "sv3", :"r2", :].shape == ts.data[0:2, 2, :2, :].shape
            assert ts[0:2, "sv3", "r2":, :].shape == ts.data[0:2, 2, 1:, :].shape
            assert ts[0:2, "sv3", "r1", :].shape == ts.data[0:2, 2, 0, :].shape
            assert ts[1:2, "sv3", :, :].shape == ts.data[1:2, 2, :, :].shape

            assert numpy.all(ts[0:2, "sv3", "r1":"r2", :].data == ts.data[0:2, 2, 0:2, :])
            assert numpy.all(ts[0:2, :"sv2", "r1":"r2", :].data == ts.data[0:2, :2, 0:2, :])
            assert numpy.all(ts[0:2, "sv3", "r1", :].data == ts.data[0:2, 2, 0, :])
            assert numpy.all(ts[2, :"sv2", "r1":"r3", :].data == ts.data[2, :2, 0:3, :])
            assert numpy.all(ts[2, "sv2", "r3", :].data == ts.data[2, 1, 2, :])
            assert numpy.all(ts[2, "sv2", "r3", 0].data == ts.data[2, 1, 2, 0])

        else:  # Different behavior for __get_item__() method...

            assert ts[1, :, :, :].shape == ts.data[[1], :, :, :].shape
            assert ts[:, 1, :, :].shape == ts.data[:, [1], :, :].shape
            assert ts[1, :, "r2":"r3", :].shape == ts.data[[1], :, 1:3, :].shape
            assert ts[:, :, 2, :].shape == ts.data[:, :, [2], :].shape
            assert ts[:, "sv3", :, :].shape == ts.data[:, [2], :, :].shape
            assert ts[2, "sv3", :, :].shape == ts.data[[[2]], [[2]], :, :].shape
            assert ts[2, "sv3", 0:3, :].shape == ts.data[[[2]], [[2]], 0:3, :].shape
            assert ts[2, "sv3", "r1":"r3", :].shape == ts.data[[[2]], [[2]], 0:3, :].shape
            assert ts[0:2, "sv3", "r1":"r3", :].shape == ts.data[0:2, [2], 0:3, :].shape
            assert ts[0:2, "sv3", :"r2", :].shape == ts.data[0:2, [2], :2, :].shape
            assert ts[0:2, "sv3", "r2":, :].shape == ts.data[0:2, [2], 1:, :].shape
            assert ts[0:2, "sv3", "r1", :].shape == ts.data[0:2, [[2]], [[0]], :].shape
            assert ts[1:2, "sv3", :, :].shape == ts.data[1:2, [2], :, :].shape

            assert numpy.all(ts[0:2, "sv3", "r1":"r2", :].data == ts.data[0:2, [2], 0:2, :])
            assert numpy.all(ts[0:2, :"sv2", "r1":"r2", :].data == ts.data[0:2, :2, 0:2, :])
            assert numpy.all(ts[0:2, "sv3", "r1", :].data == ts.data[0:2, [[2]], [[0]], :])
            assert numpy.all(ts[2, :"sv2", "r1":"r3", :].data == ts.data[[2], :2, 0:3, :])
            assert numpy.all(ts[2, "sv2", "r3", :].data == ts.data[[[[2]]], [[[1]]], [[[2]]], :])
            assert numpy.all(ts[2, "sv2", "r3", 0].data == ts.data[[[[[2]]]], [[[[1]]]], [[[[2]]]], [[[[0]]]]])

        # IndexError because of [0][0] on numpy array in TSget_index_of_state_variable
        with pytest.raises(ValueError):
            ts[:, "sv0", :, :]

        with pytest.raises(ValueError):
            ts[0, :, "r1":"r5", :]

        with pytest.raises(IndexError):
            ts[0, 10, :, :]

        with pytest.raises(AttributeError):
            ts.lfp

    @pytest.mark.parametrize('datatype', datatypes_to_test)
    def test_timeseries_4D(self, datatype):
        data, start_time, sample_period, sample_period_unit = self._prepare_dummy_time_series(4)
        ts_4D = datatype(data=data,
                         labels_dimensions={TimeSeriesDimensions.SPACE.value: ["r1", "r2", "r3", "r4"],
                                            TimeSeriesDimensions.VARIABLES.value: [
                                                PossibleVariables.X.value, PossibleVariables.Y.value,
                                                "sv3"]},
                         start_time=start_time, sample_period=sample_period,
                         sample_period_unit=sample_period_unit)
        assert ts_4D.data.shape == (3, 3, 4, 4)
        assert ts_4D.x.data.shape == (3, 1, 4, 4)

    def test_plot_ts_xarray(self):
        data, start_time, sample_period, sample_period_unit = self._prepare_dummy_time_series(2)
        ts_xarray = TimeSeriesXarray(data=data, labels_dimensions={TimeSeriesDimensions.SPACE.value: ["r1", "r2", "r3"],
                                                                   TimeSeriesDimensions.VARIABLES.value: ["SV1"]},
                                     start_time=start_time, sample_period=sample_period,
                                     sample_period_unit=sample_period_unit)
        ts_xarray.configure()
        ts_xarray.plot_timeseries(plotter_config=FiguresConfig(),
                        hue="Region" if data.shape[2] > 10 else None,
                        per_variable=data.shape[1] > 3)
