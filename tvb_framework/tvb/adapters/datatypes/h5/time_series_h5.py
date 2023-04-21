# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
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
.. moduleauthor:: Paula Popa <paula.popa@codemart.ro>
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""

import json

from tvb.basic.neotraits.api import Int
from tvb.core.adapters.arguments_serialisation import *
from tvb.core.neotraits.h5 import H5File, Scalar, DataSet, Reference, Json
from tvb.core.utils import prepare_time_slice
from tvb.datatypes.time_series import *

NO_OF_DEFAULT_SELECTED_CHANNELS = 20


class TimeSeriesH5(H5File):
    def __init__(self, path):
        super(TimeSeriesH5, self).__init__(path)
        self.title = Scalar(TimeSeries.title, self)
        self.data = DataSet(TimeSeries.data, self, expand_dimension=0)
        self.nr_dimensions = Scalar(Int(), self, name="nr_dimensions")

        # omitted length_nd , these are indexing props, to be removed from datatype too
        self.labels_ordering = Json(TimeSeries.labels_ordering, self)
        self.labels_dimensions = Json(TimeSeries.labels_dimensions, self)

        self.time = DataSet(TimeSeries.time, self, expand_dimension=0)
        self.start_time = Scalar(TimeSeries.start_time, self)
        self.sample_period = Scalar(TimeSeries.sample_period, self)
        self.sample_period_unit = Scalar(TimeSeries.sample_period_unit, self)
        self.sample_rate = Scalar(Float(), self, name="sample_rate")

        # omitted has_surface_mapping, has_volume_mapping, as they are indexing props to be filled only in DB

        # experiment: load header data eagerly, see surface for a lazy approach
        # as we do not explicitly make a difference between opening for read or write
        # the file might not yet exist, so loading headers makes no sense

        if not self.is_new_file:
            self._sample_period = self.sample_period.load()
            self._start_time = self.start_time.load()

    # experimental port of some of the data access apis from the datatype
    # NOTE: some methods can not be here as they load data from dependent data types
    #       or they assume that dependent data has been loaded
    #       Those belong to a higher level where dependent h5 files are handles and
    #       partially loaded datatypes are filled

    def read_data_shape(self):
        return self.data.shape

    def read_data_slice(self, data_slice):
        """
        Expose chunked-data access.
        """
        return self.data[data_slice]

    def read_time_page(self, current_page, page_size, max_size=None):
        """
        Compute time for current page.
        :param current_page: Starting from 0
        """
        # todo: why are we even storing the time array if we return a synthetized version?
        current_page = int(current_page)
        page_size = int(page_size)

        if max_size is None:
            max_size = page_size
        else:
            max_size = int(max_size)

        page_real_size = page_size * self._sample_period
        start_time = self._start_time + current_page * page_real_size
        end_time = start_time + min(page_real_size, max_size * self._sample_period)

        return numpy.arange(start_time, end_time, self._sample_period)

    def read_channels_page(self, from_idx, to_idx, step=None, specific_slices=None, channels_list=None):
        """
        Read and return only the data page for the specified channels list.

        :param from_idx: the starting time idx from which to read data
        :param to_idx: the end time idx up until to which you read data
        :param step: increments in which to read the data. Optional, default to 1.
        :param specific_slices: optional parameter. If speficied slices the data accordingly.
        :param channels_list: the list of channels for which we want data
        """
        if channels_list:
            channels_list = json.loads(channels_list)
            for i in range(len(channels_list)):
                channels_list[i] = int(channels_list[i])

        if channels_list:
            channel_slice = tuple(channels_list)
        else:
            channel_slice = slice(None)

        data_page = self.read_data_page(from_idx, to_idx, step, specific_slices)
        # This is just a 1D array like in the case of Global Average monitor.
        # No need for the channels list
        if len(data_page.shape) == 1:
            return data_page.reshape(data_page.shape[0], 1)
        else:
            return data_page[:, channel_slice]

    def read_data_page(self, from_idx, to_idx, step=None, specific_slices=None):
        """
        Retrieve one page of data (paging done based on time).
        """
        from_idx, to_idx = int(from_idx), int(to_idx)

        if isinstance(specific_slices, str):
            specific_slices = json.loads(specific_slices)
        if step is None:
            step = 1
        else:
            step = int(step)

        slices = []
        overall_shape = self.data.shape
        for i in range(len(overall_shape)):
            if i == 0:
                # Time slice
                slices.append(
                    slice(from_idx, min(to_idx, overall_shape[0]), step))
                continue
            if i == 2:
                # Read full of the main_dimension (space for the simulator)
                slices.append(slice(overall_shape[i]))
                continue
            if specific_slices is None:
                slices.append(slice(0, 1))
            else:
                slices.append(slice(specific_slices[i], min(specific_slices[i] + 1, overall_shape[i]), 1))

        data = self.data[tuple(slices)]
        data = data.squeeze()

        if len(data.shape) == 1:
            # Do not allow time dimension to get squeezed, a 2D result need to
            # come out of this method.
            data = data.reshape((1, len(data)))

        return data

    def write_time_slice(self, partial_result):
        """
        Append a new value to the ``time`` attribute.
        """
        self.time.append(partial_result, False)

    def write_data_slice(self, partial_result):
        """
        Append a chunk of time-series data to the ``data`` attribute.
        """
        self.data.append(partial_result, False)

    def write_data_slice_on_grow_dimension(self, partial_result, grow_dimension=0):
        self.data.append(partial_result, grow_dimension=grow_dimension, close_file=False)

    def get_min_max_values(self):
        """
        Retrieve the minimum and maximum values from the metadata.
        :returns: (minimum_value, maximum_value)
        """
        metadata = self.data.get_cached_metadata()
        return metadata.min, metadata.max

    def get_space_labels(self):
        """
        It assumes that we want to select in the 3'rd dimension,
        and generates labels for each point in that dimension.
        Subclasses are more specific.
        :return: An array of strings.
        """
        if self.nr_dimensions.load() > 2:
            return ['signal-%d' % i for i in range(self.data.shape[2])]
        else:
            return []

    def get_grouped_space_labels(self):
        """
        :return: A list of label groups. A label group is a tuple (name, [(label_idx, label)...]).
                 Default all labels in a group named ''
        """
        return [('', list(enumerate(self.get_space_labels())))]

    def get_default_selection(self):
        """
        :return: The measure point indices that have to be shown by default. By default show all.
        """
        return list(range(min(NO_OF_DEFAULT_SELECTED_CHANNELS, self.data.shape[2])))

    def get_measure_points_selection_gid(self):
        """
        :return: a datatype gid with which to obtain al valid measure point selection for this time series
                 We have to decide if the default should be all selections or none
        """
        return ''

    def store_references(self, ts):
        pass


class TimeSeriesRegionH5(TimeSeriesH5):
    def __init__(self, path):
        super(TimeSeriesRegionH5, self).__init__(path)
        self.connectivity = Reference(TimeSeriesRegion.connectivity, self)
        self.region_mapping_volume = Reference(TimeSeriesRegion.region_mapping_volume, self)
        self.region_mapping = Reference(TimeSeriesRegion.region_mapping, self)
        self.labels_ordering = Json(TimeSeriesRegion.labels_ordering, self)

    @staticmethod
    def out_of_range(min_value):
        return round(min_value) - 1

    def get_measure_points_selection_gid(self):
        """
        :return: the associated connectivity gid
        """
        connectivity_gid = self.connectivity.load()
        if connectivity_gid is None:
            return ''
        return connectivity_gid.hex

    def store_references(self, ts):
        self.connectivity.store(ts.connectivity.gid)

        if ts.region_mapping_volume is not None:
            self.region_mapping_volume.store(ts.region_mapping_volume.gid)

        if ts.region_mapping is not None:
            self.region_mapping.store(ts.region_mapping.gid)


class TimeSeriesSurfaceH5(TimeSeriesH5):
    SELECTION_LIMIT = 100

    def __init__(self, path):
        super(TimeSeriesSurfaceH5, self).__init__(path)
        self.surface = Reference(TimeSeriesSurface.surface, self)
        self.labels_ordering = Json(TimeSeriesSurface.labels_ordering, self)

    def store_references(self, ts):
        self.surface.store(ts.surface)

    def get_space_labels(self):
        """
        Return only the first `SELECTION_LIMIT` vertices/channels
        """
        return ['signal-%d' % i for i in range(min(self.data.shape[2], self.SELECTION_LIMIT))]


class TimeSeriesVolumeH5(TimeSeriesH5):
    def __init__(self, path):
        super(TimeSeriesVolumeH5, self).__init__(path)
        self.volume = Reference(TimeSeriesVolume.volume, self)
        self.labels_ordering = Json(TimeSeriesVolume.labels_ordering, self)

    def store_references(self, ts):
        self.volume.store(ts.volume.gid)

    def get_volume_view(self, from_idx, to_idx, x_plane, y_plane, z_plane, **kwargs):
        """
        Retrieve 3 slices through the Volume TS, at the given X, y and Z coordinates, and in time [from_idx .. to_idx].

        :param from_idx: int This will be the limit on the first dimension (time)
        :param to_idx: int Also limit on the first Dimension (time)
        :param x_plane: int coordinate
        :param y_plane: int coordinate
        :param z_plane: int coordinate

        :return: An array of 3 Matrices 2D, each containing the values to display in planes xy, yz and xy.
        """

        overall_shape = self.data.shape
        from_idx, to_idx, time = preprocess_time_parameters(from_idx, to_idx, overall_shape[0])
        x_plane, y_plane, z_plane = preprocess_space_parameters(x_plane, y_plane, z_plane,
                                                                overall_shape[1], overall_shape[2], overall_shape[3])

        slices = slice(from_idx, to_idx), slice(overall_shape[1]), slice(overall_shape[2]), slice(z_plane, z_plane + 1)
        slicex = self.read_data_slice(slices)[:, :, :, 0].tolist()

        slices = slice(from_idx, to_idx), slice(x_plane, x_plane + 1), slice(overall_shape[2]), slice(overall_shape[3])
        slicey = self.read_data_slice(slices)[:, 0, :, :][..., ::-1].tolist()

        slices = slice(from_idx, to_idx), slice(overall_shape[1]), slice(y_plane, y_plane + 1), slice(overall_shape[3])
        slicez = self.read_data_slice(slices)[:, :, 0, :][..., ::-1].tolist()

        return [slicex, slicey, slicez]

    def get_voxel_time_series(self, x_plane, y_plane, z_plane, **kwargs):
        """
        Retrieve for a given voxel (x,y,z) the entire timeline.

        :param x: int coordinate
        :param y: int coordinate
        :param z: int coordinate

        :return: A complex dictionary with information about current voxel.
                The main part will be a vector with all the values over time from the x,y,z coordinates.
        """

        overall_shape = self.data.shape
        x, y, z = preprocess_space_parameters(x_plane, y_plane, z_plane, overall_shape[1], overall_shape[2],
                                              overall_shape[3])

        slices = prepare_time_slice(overall_shape[0]), slice(x, x + 1), slice(y, y + 1), slice(z, z + 1)

        result = postprocess_voxel_ts(self, slices)
        return result


class TimeSeriesSensorsH5(TimeSeriesH5):
    def get_measure_points_selection_gid(self):
        sensors_gid = self.sensors.load()
        if sensors_gid is not None:
            return sensors_gid
        return ''

    def store_references(self, ts):
        pass


class TimeSeriesEEGH5(TimeSeriesSensorsH5):
    def __init__(self, path):
        super(TimeSeriesEEGH5, self).__init__(path)
        self.sensors = Reference(TimeSeriesEEG.sensors, self)
        self.labels_order = Json(TimeSeriesEEG.labels_ordering, self)

    def store_references(self, ts):
        self.sensors.store(ts.sensors.gid)


class TimeSeriesMEGH5(TimeSeriesSensorsH5):
    def __init__(self, path):
        super(TimeSeriesMEGH5, self).__init__(path)
        self.sensors = Reference(TimeSeriesMEG.sensors, self)
        self.labels_order = Json(TimeSeriesMEG.labels_ordering, self)

    def store_references(self, ts):
        self.sensors.store(ts.sensors.gid)


class TimeSeriesSEEGH5(TimeSeriesSensorsH5):
    def __init__(self, path):
        super(TimeSeriesSEEGH5, self).__init__(path)
        self.sensors = Reference(TimeSeriesSEEG.sensors, self)
        self.labels_order = Json(TimeSeriesSEEG.labels_ordering, self)

    def store_references(self, ts):
        self.sensors.store(ts.sensors.gid)
