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
A Javascript displayer for time series, using SVG.

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import json
from abc import ABCMeta
from six import add_metaclass

from tvb.adapters.datatypes.h5.time_series_h5 import TimeSeriesRegionH5, TimeSeriesSensorsH5, TimeSeriesH5
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.adapters.abcdisplayer import ABCDisplayer, URLGenerator
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex
from tvb.core.neotraits.forms import TraitDataTypeSelectField
from tvb.core.neocom import h5
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.core.utils import TVBJSONEncoder
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.time_series import TimeSeries


class TimeSeriesModel(ViewModel):
    time_series = DataTypeGidAttr(
        linked_datatype=TimeSeries,
        label="Time series to be displayed in a 2D form."
    )


class TimeSeriesForm(ABCAdapterForm):

    def __init__(self):
        super(TimeSeriesForm, self).__init__()

        self.time_series = TraitDataTypeSelectField(TimeSeriesModel.time_series, name='time_series',
                                                    conditions=self.get_filters())

    @staticmethod
    def get_view_model():
        return TimeSeriesModel

    @staticmethod
    def get_required_datatype():
        return TimeSeriesIndex

    @staticmethod
    def get_input_name():
        return 'time_series'

    @staticmethod
    def get_filters():
        return FilterChain(fields=[FilterChain.datatype + '.time_series_type'], operations=["in"],
                           values=[['TimeSeriesEEG', 'TimeSeriesSEEG', 'TimeSeriesMEG', 'TimeSeriesRegion',
                                    'TimeSeriesSurface']])


@add_metaclass(ABCMeta)
class ABCSpaceDisplayer(ABCDisplayer):

    @staticmethod
    def build_params_for_selectable_connectivity(connectivity):
        # type: (Connectivity) -> dict
        return {'measurePointsSelectionGID': connectivity.gid,
                'initialSelection': connectivity.saved_selection or list(range(len(connectivity.region_labels))),
                'groupedLabels': connectivity.get_grouped_space_labels()}

    def build_params_for_subselectable_ts(self, ts_h5):
        """
        creates a template dict with the initial selection to be
        displayed in a time series viewer
        """
        return {'measurePointsSelectionGID': ts_h5.get_measure_points_selection_gid(),
                'initialSelection': ts_h5.get_default_selection(),
                'groupedLabels': self.get_grouped_space_labels(ts_h5)}

    def get_grouped_space_labels(self, ts_h5):
        """
        :return: A structure of this form [('left', [(idx, lh_label)...]), ('right': [(idx, rh_label) ...])]
        """
        if isinstance(ts_h5, TimeSeriesSensorsH5):
            sensors_gid = ts_h5.sensors.load()
            with h5.h5_file_for_gid(sensors_gid) as sensors_h5:
                labels = sensors_h5.labels.load()
                # TODO uncomment this when the UI component will be able to scale for many groups
                # if isinstance(ts_h5, TimeSeriesSEEGH5):
                #     return SensorsInternal.group_sensors_to_electrodes(labels)
                return [('', list(enumerate(labels)))]

        if isinstance(ts_h5, TimeSeriesRegionH5):
            connectivity_gid = ts_h5.connectivity.load()
            conn = self.load_traited_by_gid(connectivity_gid)
            assert isinstance(conn, Connectivity)
            return conn.get_grouped_space_labels()

        return ts_h5.get_grouped_space_labels()

    def get_space_labels(self, ts_h5):
        """
        :return: An array of strings with the connectivity node labels.
        """
        if type(ts_h5) is TimeSeriesRegionH5:
            connectivity_gid = ts_h5.connectivity.load()
            if connectivity_gid is None:
                return []

            with h5.h5_file_for_gid(connectivity_gid) as conn_h5:
                return list(conn_h5.region_labels.load())

        if isinstance(ts_h5, TimeSeriesSensorsH5):
            sensors_gid = ts_h5.sensors.load()
            if sensors_gid is None:
                return []
            with h5.h5_file_for_gid(sensors_gid) as sensors_h5:
                return list(sensors_h5.labels.load())

        return ts_h5.get_space_labels()


class TimeSeriesDisplay(ABCSpaceDisplayer):
    _ui_name = "Time Series Visualizer (SVG/d3)"
    _ui_subsection = "timeseries"

    MAX_PREVIEW_DATA_LENGTH = 200

    def get_form_class(self):
        return TimeSeriesForm

    def get_required_memory_size(self, view_model):
        # type: (TimeSeriesModel) -> int
        """Return required memory."""
        return -1

    def _launch(self, view_model, figsize, preview=False):
        time_series_index = self.load_entity_by_gid(view_model.time_series)
        h5_file = h5.h5_file_for_index(time_series_index)
        assert isinstance(h5_file, TimeSeriesH5)
        shape = list(h5_file.read_data_shape())
        ts = h5_file.time.load()
        state_variables = time_series_index.get_labels_for_dimension(1)
        labels = self.get_space_labels(h5_file)

        # Assume that the first dimension is the time since that is the case so far
        if preview and shape[0] > self.MAX_PREVIEW_DATA_LENGTH:
            shape[0] = self.MAX_PREVIEW_DATA_LENGTH

        # when surface-result, the labels will be empty, so fill some of them,
        # but not all, otherwise the viewer will take ages to load.
        if shape[2] > 0 and len(labels) == 0:
            for n in range(min(self.MAX_PREVIEW_DATA_LENGTH, shape[2])):
                labels.append("Node-" + str(n))

        pars = {'baseURL': URLGenerator.build_base_h5_url(time_series_index.gid),
                'labels': labels, 'labels_json': json.dumps(labels, cls=TVBJSONEncoder),
                'ts_title': time_series_index.title, 'preview': preview, 'figsize': figsize,
                'shape': repr(shape), 't0': ts[0],
                'dt': ts[1] - ts[0] if len(ts) > 1 else 1,
                'labelsStateVar': state_variables, 'labelsModes': list(range(shape[3]))
                }
        pars.update(self.build_params_for_subselectable_ts(h5_file))
        h5_file.close()

        return self.build_display_result("time_series/view", pars, pages=dict(controlPage="time_series/control"))

    def launch(self, view_model):
        # type: (TimeSeriesModel) -> dict
        """Construct data for visualization and launch it."""
        return self._launch(view_model, None)
