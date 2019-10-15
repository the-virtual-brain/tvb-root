# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
The TimeSeries datatypes. This brings together the scientific and framework 
methods that are associated with the time-series data.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""

import json
import numpy
from tvb.basic.traits import exceptions, types_mapped
from tvb.datatypes import sensors, surfaces, volumes, region_mapping, connectivity
from tvb.basic.arguments_serialisation import (preprocess_space_parameters, preprocess_time_parameters,
    postprocess_voxel_ts)
from tvb.basic.logger.builder import get_logger
from tvb.basic.neotraits.api import HasTraits, Attr, NArray, List, Int, Float

LOG = get_logger(__name__)


def prepare_time_slice(total_time_length, max_length=10 ** 4):
    """
    Limit the time dimension when retrieving from TS.
    If total time length is greater than MAX, then retrieve only the last part of the TS

    :param total_time_length: TS time dimension
    :param max_length: limiting number of TS steps

    :return: python slice
    """

    if total_time_length < max_length:
        return slice(total_time_length)

    return slice(total_time_length - max_length, total_time_length)


class TimeSeries(HasTraits):
    """
    Base time-series dataType.
    """

    title = Attr(str)

    data = NArray(
        label="Time-series data",
        # file_storage=core.FILE_STORAGE_EXPAND,
        doc="""An array of time-series data, with a shape of [tpts, :], where ':' represents 1 or more dimensions""")

    # mhtodo: should this not be a property
    nr_dimensions = Int(
        label="Number of dimension in timeseries",
        default=4
    )

    # length_1d, length_2d, length_3d, length_4d = [Int() for _ in range(4)]

    labels_ordering = List(
        default=("Time", "State Variable", "Space", "Mode"),
        label="Dimension Names",
        doc="""List of strings representing names of each data dimension"""
    )

    labels_dimensions = Attr(
        field_type=dict,
        default={},
        label="Specific labels for each dimension for the data stored in this timeseries.",
        doc=""" A dictionary containing mappings of the form {'dimension_name' : [labels for this dimension] }""")

    time = NArray(
        # file_storage=core.FILE_STORAGE_EXPAND,
        label="Time-series time",
        required=False,
        doc="""An array of time values for the time-series, with a shape of [tpts,].
            This is 'time' as returned by the simulator's monitors."""
    )

    start_time = Float(label="Start Time:")

    sample_period = Float(label="Sample period", default=1.0)

    # Specify the measure unit for sample period (e.g sec, msec, usec, ...)
    sample_period_unit = Attr(
        field_type=str,
        label="Sample Period Measure Unit",
        default="ms"
    )

    sample_rate = Float(
        label="Sample rate",
        doc="""The sample rate of the timeseries"""
    )

    # has_surface_mapping = Attr(field_type=bool, default=True)
    # has_volume_mapping = Attr(field_type=bool, default=False)

    def configure(self):
        """
        After populating few fields, compute the rest of the fields
        """
        super(TimeSeries, self).configure()
        self.nr_dimensions = self.data.ndim
        self.sample_rate = 1.0 / self.sample_period

        # for i in range(min(self.nr_dimensions, 4)):
        #     setattr(self, 'length_%dd' % (i + 1), int(data_shape[i]))

    def read_data_shape(self):
        """
        Expose shape read on field data.
        """
        try:
            return self.get_data_shape('data')
        except exceptions.TVBException:
            self.logger.exception("Could not read data shape for TS!")
            raise exceptions.TVBException("Invalid empty TimeSeries!")

    def read_data_page_split(self, from_idx, to_idx, step=None, specific_slices=None):
        """
        No Split needed in case of basic TS (sensors and region level)
        """
        return self.read_data_page(from_idx, to_idx, step, specific_slices)

    def get_space_labels(self):
        """
        It assumes that we want to select in the 3'rd dimension,
        and generates labels for each point in that dimension.
        Subclasses are more specific.
        :return: An array of strings.
        """
        if self.nr_dimensions > 2:
            return ['signal-%d' % i for i in range(self._length_3d)]
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
        return range(len(self.get_space_labels()))

    def get_measure_points_selection_gid(self):
        """
        :return: a datatype gid with which to obtain al valid measure point selection for this time series
                 We have to decide if the default should be all selections or none
        """
        return ''

    @staticmethod
    def accepted_filters():
        filters = types_mapped.MappedType.accepted_filters()
        filters.update({'datatype_class._nr_dimensions': {'type': 'int', 'display': 'No of Dimensions',
                                                          'operations': ['==', '<', '>']},
                        'datatype_class._sample_period': {'type': 'float', 'display': 'Sample Period',
                                                          'operations': ['==', '<', '>']},
                        'datatype_class._sample_rate': {'type': 'float', 'display': 'Sample Rate',
                                                        'operations': ['==', '<', '>']},
                        'datatype_class._title': {'type': 'string', 'display': 'Title',
                                                  'operations': ['==', '!=', 'like']}})
        return filters

    def _find_summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = {"Time-series type": self.__class__.__name__,
                   "Time-series name": self.title,
                   "Dimensions": self.labels_ordering,
                   "Time units": self.sample_period_unit,
                   "Sample period": self.sample_period,
                   "Length": self.sample_period * self.get_data_shape('data')[0]}
        summary.update(self.get_info_about_array('data'))
        return summary


class SensorsTSBase(TimeSeries):
    """
    Add framework related functionality for TS Sensor classes

    """
    __tablename__ = None


    def get_space_labels(self):
        """
        :return: An array of strings with the sensors labels.
        """
        if self.sensors is not None:
            return list(self.sensors.labels)
        return []

    def get_measure_points_selection_gid(self):
        if self.sensors is not None:
            return self.sensors.gid
        return ''

    def get_default_selection(self):
        if self.sensors is not None:
            # select only the first 8 channels
            return range(min(8, len(self.get_space_labels())))
        return []

    def _find_summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = super(SensorsTSBase, self)._find_summary_info()
        summary.update({"Source Sensors": self.sensors.display_name})
        return summary


class TimeSeriesEEG(SensorsTSBase):
    """ A time series associated with a set of EEG sensors. """
    _ui_name = "EEG time-series"
    __generate_table__ = True

    sensors = Attr(field_type=sensors.SensorsEEG)
    labels_ordering = List(of=str, default=("Time", "1", "EEG Sensor", "1"))


class TimeSeriesMEG(SensorsTSBase):
    """ A time series associated with a set of MEG sensors. """
    _ui_name = "MEG time-series"
    __generate_table__ = True

    sensors = Attr(field_type=sensors.SensorsMEG)
    labels_ordering = List(of=str, default=("Time", "1", "MEG Sensor", "1"))


class TimeSeriesSEEG(SensorsTSBase):
    """ A time series associated with a set of Internal sensors. """
    _ui_name = "Stereo-EEG time-series"
    __generate_table__ = True

    sensors = Attr(field_type=sensors.SensorsInternal)
    labels_ordering = List(of=str, default=("Time", "1", "sEEG Sensor", "1"))


class TimeSeriesRegion(TimeSeries):
    """ A time-series associated with the regions of a connectivity. """
    _ui_name = "Region time-series"
    connectivity = Attr(field_type=connectivity.Connectivity)
    region_mapping_volume = Attr(field_type=region_mapping.RegionVolumeMapping, required=False)
    region_mapping = Attr(field_type=region_mapping.RegionMapping, required=False)
    labels_ordering = List(of=str, default=("Time", "State Variable", "Region", "Mode"))

    def configure(self):
        """
        After populating few fields, compute the rest of the fields
        """
        super(TimeSeriesRegion, self).configure()
        # self.has_surface_mapping = self.region_mapping is not None or self._region_mapping is not None
        # self.has_volume_mapping = self.region_mapping_volume is not None or self._region_mapping_volume is not None

    def _find_summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = super(TimeSeriesRegion, self)._find_summary_info()
        summary.update({"Source Connectivity": self.connectivity.display_name,
                        "Region Mapping": self.region_mapping.display_name if self.region_mapping else "None",
                        "Region Mapping Volume": (self.region_mapping_volume.display_name
                                                  if self.region_mapping_volume else "None")})
        return summary

    def get_space_labels(self):
        """
        :return: An array of strings with the connectivity node labels.
        """
        if self.connectivity is not None:
            return list(self.connectivity.region_labels)
        return []

    def get_grouped_space_labels(self):
        """
        :return: A structure of this form [('left', [(idx, lh_label)...]), ('right': [(idx, rh_label) ...])]
        """
        if self.connectivity is not None:
            return self.connectivity.get_grouped_space_labels()
        else:
            return super(TimeSeriesRegion, self).get_grouped_space_labels()

    def get_default_selection(self):
        """
        :return: If the connectivity of this time series is edited from another
                 return the nodes of the parent that are present in the connectivity.
        """
        if self.connectivity is not None:
            return self.connectivity.get_default_selection()
        else:
            return super(TimeSeriesRegion, self).get_default_selection()

    def get_measure_points_selection_gid(self):
        """
        :return: the associated connectivity gid
        """
        if self.connectivity is not None:
            return self.connectivity.get_measure_points_selection_gid()
        else:
            return super(TimeSeriesRegion, self).get_measure_points_selection_gid()

    # TODO: move to higher level
    def get_volume_view(self, from_idx, to_idx, x_plane, y_plane, z_plane, var=0, mode=0):
        """
        Retrieve 3 slices through the Volume TS, at the given X, y and Z coordinates, and in time [from_idx .. to_idx].

        :param from_idx: int This will be the limit on the first dimension (time)
        :param to_idx: int Also limit on the first Dimension (time)
        :param x_plane: int coordinate
        :param y_plane: int coordinate
        :param z_plane: int coordinate

        :return: An array of 3 Matrices 2D, each containing the values to display in planes xy, yz and xy.
        """

        if self.region_mapping_volume is None:
            raise exceptions.TVBException("Invalid method called for TS without Volume Mapping!")

        volume_rm = self.region_mapping_volume

        # Work with space inside Volume:
        x_plane, y_plane, z_plane = preprocess_space_parameters(x_plane, y_plane, z_plane, volume_rm.length_1d,
                                                                volume_rm.length_2d, volume_rm.length_3d)
        var, mode = int(var), int(mode)
        slice_x, slice_y, slice_z = self.region_mapping_volume.get_volume_slice(x_plane, y_plane, z_plane)

        # Read from the current TS:
        from_idx, to_idx, current_time_length = preprocess_time_parameters(from_idx, to_idx, self.read_data_shape()[0])
        no_of_regions = self.read_data_shape()[2]
        time_slices = slice(from_idx, to_idx), slice(var, var + 1), slice(no_of_regions), slice(mode, mode + 1)

        min_signal = self.get_min_max_values()[0]
        regions_ts = self.read_data_slice(time_slices)[:, 0, :, 0]
        regions_ts = numpy.hstack((regions_ts, numpy.ones((current_time_length, 1)) * self.out_of_range(min_signal)))

        # Index from TS with the space mapping:
        result_x, result_y, result_z = [], [], []

        for i in range(0, current_time_length):
            result_x.append(regions_ts[i][slice_x].tolist())
            result_y.append(regions_ts[i][slice_y].tolist())
            result_z.append(regions_ts[i][slice_z].tolist())

        return [result_x, result_y, result_z]

    # TODO: move to higher level
    def get_voxel_time_series(self, x, y, z, var=0, mode=0):
        """
        Retrieve for a given voxel (x,y,z) the entire timeline.

        :param x: int coordinate
        :param y: int coordinate
        :param z: int coordinate

        :return: A complex dictionary with information about current voxel.
                The main part will be a vector with all the values over time from the x,y,z coordinates.
        """

        if self.region_mapping_volume is None:
            raise exceptions.TVBException("Invalid method called for TS without Volume Mapping!")

        volume_rm = self.region_mapping_volume
        x, y, z = preprocess_space_parameters(x, y, z, volume_rm.length_1d, volume_rm.length_2d, volume_rm.length_3d)
        idx_slices = slice(x, x + 1), slice(y, y + 1), slice(z, z + 1)

        idx = int(volume_rm.get_data('array_data', idx_slices))

        time_length = self.read_data_shape()[0]
        var, mode = int(var), int(mode)
        voxel_slices = prepare_time_slice(time_length), slice(var, var + 1), slice(idx, idx + 1), slice(mode, mode + 1)
        label = volume_rm.connectivity.region_labels[idx]

        background, back_min, back_max = None, None, None
        if idx < 0:
            back_min, back_max = self.get_min_max_values()
            background = numpy.ones((time_length, 1)) * self.out_of_range(back_min)
            label = 'background'

        result = postprocess_voxel_ts(self, voxel_slices, background, back_min, back_max, label)
        return result

    @staticmethod
    def out_of_range(min_value):
        return round(min_value) - 1


class TimeSeriesSurface(TimeSeries):
    """ A time-series associated with a Surface. """
    _ui_name = "Surface time-series"
    surface = Attr(field_type=surfaces.CorticalSurface)
    labels_ordering = List(of=str, default=("Time", "State Variable", "Vertex", "Mode"))
    SELECTION_LIMIT = 100

    def get_space_labels(self):
        """
        Return only the first `SELECTION_LIMIT` vertices/channels
        """
        return ['signal-%d' % i for i in range(min(self._length_3d, self.SELECTION_LIMIT))]

    def _find_summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = super(TimeSeriesSurface, self)._find_summary_info()
        summary.update({"Source Surface": self.surface.display_name})
        return summary

    def read_data_page_split(self, from_idx, to_idx, step=None, specific_slices=None):

        basic_result = self.read_data_page(from_idx, to_idx, step, specific_slices)
        result = []
        if self.surface.number_of_split_slices <= 1:
            result.append(basic_result.tolist())
        else:
            for slice_number in range(self.surface.number_of_split_slices):
                start_idx, end_idx = self.surface._get_slice_vertex_boundaries(slice_number)
                result.append(basic_result[:,start_idx:end_idx].tolist())

        return result


class TimeSeriesVolume(TimeSeries):
    """ A time-series associated with a Volume. """
    _ui_name = "Volume time-series"
    volume = Attr(field_type=volumes.Volume)
    labels_ordering = List(of=str, default=("Time", "X", "Y", "Z"))

    def _find_summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = super(TimeSeriesVolume, self)._find_summary_info()
        summary.update({"Source Volume": self.volume.display_name})
        return summary

    def configure(self):
        """
        After populating few fields, compute the rest of the fields
        """
        super(TimeSeriesVolume, self).configure()
        self.has_volume_mapping = True

