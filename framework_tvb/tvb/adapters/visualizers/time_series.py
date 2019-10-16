# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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
A Javascript displayer for time series, using SVG.

.. moduleauthor:: Marmaduke Woodman <mw@eml.cc>

"""

import json
from abc import ABCMeta
from six import add_metaclass
from tvb.core.entities.file.datatypes.time_series_h5 import TimeSeriesRegionH5, TimeSeriesSensorsH5, TimeSeriesH5
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.adapters.abcdisplayer import ABCDisplayer, URLGenerator
from tvb.core.entities.model.datatypes.time_series import TimeSeriesIndex
from tvb.core.neotraits.forms import DataTypeSelectField
from tvb.core.neocom import h5
from tvb.datatypes.connectivity import Connectivity


class TimeSeriesForm(ABCAdapterForm):

    def __init__(self, prefix='', project_id=None):
        super(TimeSeriesForm, self).__init__(prefix, project_id, False)

        self.time_series = DataTypeSelectField(self.get_required_datatype(), self, name='time_series', required=True,
                                               label="Time series to be displayed in a 2D form.",
                                               conditions=self.get_filters())

    @staticmethod
    def get_required_datatype():
        return TimeSeriesIndex

    @staticmethod
    def get_input_name():
        return '_time_series'

    @staticmethod
    def get_filters():
        return FilterChain(fields=[FilterChain.datatype + '.time_series_type'], operations=["in"],
                           values=[['TimeSeriesEEG', 'TimeSeriesSEEG', 'TimeSeriesMEG', 'TimeSeriesRegion',
                                    'TimeSeriesSurface']])


@add_metaclass(ABCMeta)
class ABCSpaceDisplayer(ABCDisplayer):

    def build_params_for_selectable_connectivity(self, connectivity):
        # type: (Connectivity) -> dict
        return {'measurePointsSelectionGID': connectivity.gid,
                'initialSelection': connectivity.saved_selection or list(range(len(connectivity.region_labels))),
                'groupedLabels': self._connectivity_grouped_space_labels(connectivity)}

    @staticmethod
    def _connectivity_grouped_space_labels(connectivity):
        """
        :return: A list [('left', [lh_labels)], ('right': [rh_labels])]
        """
        hemispheres = connectivity.hemispheres
        region_labels = connectivity.region_labels
        if hemispheres is not None and hemispheres.size:
            l, r = [], []

            for i, (is_right, label) in enumerate(zip(hemispheres, region_labels)):
                if is_right:
                    r.append((i, label))
                else:
                    l.append((i, label))
            return [('left', l), ('right', r)]
        else:
            return [('', list(enumerate(region_labels)))]

    def build_params_for_subselectable_ts(self, ts_h5):
        """
        creates a template dict with the initial selection to be
        displayed in a time series viewer
        """
        return {'measurePointsSelectionGID': ts_h5.get_measure_points_selection_gid(),
                'initialSelection': ts_h5.get_default_selection(),
                'groupedLabels': self._get_grouped_space_labels(ts_h5)}

    def _get_grouped_space_labels(self, ts_h5):
        """
        :return: A structure of this form [('left', [(idx, lh_label)...]), ('right': [(idx, rh_label) ...])]
        """
        if isinstance(ts_h5, TimeSeriesRegionH5):
            connectivity_gid = ts_h5.connectivity.load()
            conn_idx = self.load_entity_by_gid(connectivity_gid.hex)
            conn = h5.load_from_index(conn_idx)
            return self._connectivity_grouped_space_labels(conn)

        ts_h5.get_grouped_space_labels()

    def get_space_labels(self, ts_h5):
        """
        :return: An array of strings with the connectivity node labels.
        """
        if type(ts_h5) is TimeSeriesRegionH5:
            connectivity_gid = ts_h5.connectivity.load()
            if connectivity_gid is None:
                return []
            conn_idx = self.load_entity_by_gid(connectivity_gid.hex)
            with h5.h5_file_for_index(conn_idx) as conn_h5:
                return list(conn_h5.region_labels.load())

        if type(ts_h5) is TimeSeriesSensorsH5:
            sensors_gid = ts_h5.sensors.load()
            if sensors_gid is None:
                return []
            sensors_idx = self.load_entity_by_gid(sensors_gid.hex)
            with h5.h5_file_for_index(sensors_idx) as sensors_h5:
                return list(sensors_h5.labels.load())

        return ts_h5.get_space_labels()


class TimeSeries(ABCSpaceDisplayer):
    _ui_name = "Time Series Visualizer (SVG/d3)"
    _ui_subsection = "timeseries"

    MAX_PREVIEW_DATA_LENGTH = 200

    def get_form_class(self):
        return TimeSeriesForm

    def get_required_memory_size(self, **kwargs):
        """Return required memory."""
        return -1

    def launch(self, time_series, preview=False, figsize=None):
        """Construct data for visualization and launch it."""
        h5_file = h5.h5_file_for_index(time_series)
        assert isinstance(h5_file, TimeSeriesH5)
        shape = list(h5_file.read_data_shape())
        ts = h5_file.storage_manager.get_data('time')
        state_variables = h5_file.labels_dimensions.load().get(time_series.labels_ordering[1], [])
        labels = self.get_space_labels(h5_file)

        # Assume that the first dimension is the time since that is the case so far
        if preview and shape[0] > self.MAX_PREVIEW_DATA_LENGTH:
            shape[0] = self.MAX_PREVIEW_DATA_LENGTH

        # when surface-result, the labels will be empty, so fill some of them,
        # but not all, otherwise the viewer will take ages to load.
        if shape[2] > 0 and len(labels) == 0:
            for n in range(min(self.MAX_PREVIEW_DATA_LENGTH, shape[2])):
                labels.append("Node-" + str(n))

        pars = {'baseURL': URLGenerator.build_base_h5_url(time_series.gid),
                'labels': labels, 'labels_json': json.dumps(labels),
                'ts_title': time_series.title, 'preview': preview, 'figsize': figsize,
                'shape': repr(shape), 't0': ts[0],
                'dt': ts[1] - ts[0] if len(ts) > 1 else 1,
                'labelsStateVar': state_variables, 'labelsModes': list(range(shape[3]))
                }
        pars.update(self.build_params_for_subselectable_ts(h5_file))
        h5_file.close()

        return self.build_display_result("time_series/view", pars, pages=dict(controlPage="time_series/control"))

    def generate_preview(self, time_series, figure_size):
        return self.launch(time_series, preview=True, figsize=figure_size)
