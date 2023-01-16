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
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""
import json
import numpy
from tvb.adapters.visualizers.time_series import ABCSpaceDisplayer
from tvb.adapters.datatypes.h5.time_series_h5 import TimeSeriesH5
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.adapters.abcdisplayer import URLGenerator
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.neotraits.forms import TraitDataTypeSelectField
from tvb.core.neocom import h5
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr, replace_nan_values
from tvb.datatypes.time_series import TimeSeries


class EegMonitorModel(ViewModel):
    input_data = DataTypeGidAttr(
        linked_datatype=TimeSeries,
        label='Input Data',
        doc='Time series to display.'
    )

    data_2 = DataTypeGidAttr(
        linked_datatype=TimeSeries,
        required=False,
        label='Input Data 2',
        doc='Time series to display.'
    )

    data_3 = DataTypeGidAttr(
        linked_datatype=TimeSeries,
        required=False,
        label='Input Data 3',
        doc='Time series to display.'
    )


class EegMonitorForm(ABCAdapterForm):

    def __init__(self):
        super(EegMonitorForm, self).__init__()
        self.input_data = TraitDataTypeSelectField(EegMonitorModel.input_data, name='input_data')
        self.data_2 = TraitDataTypeSelectField(EegMonitorModel.data_2, name='data_2')
        self.data_3 = TraitDataTypeSelectField(EegMonitorModel.data_3, name='data_3')

    @staticmethod
    def get_view_model():
        return EegMonitorModel

    @staticmethod
    def get_required_datatype():
        return TimeSeriesIndex

    @staticmethod
    def get_input_name():
        return 'input_data'

    @staticmethod
    def get_filters():
        return None


class EegMonitor(ABCSpaceDisplayer):
    """
    This viewer takes as inputs at least one ArrayWrapper and at most 3 
    ArrayWrappers, and returns the needed parameters for a 2D representation 
    of the values from these arrays, in EEG form. So far arrays of at most 3
    dimensions are supported.
    """
    has_nan = False
    _ui_name = "Animated Time Series Visualizer"
    _ui_subsection = "animated_timeseries"

    page_size = 4000
    preview_page_size = 250
    current_page = 0

    def get_form_class(self):
        return EegMonitorForm

    def get_required_memory_size(self, view_model):
        # type: (EegMonitorModel) -> int
        """
        Return the required memory to run this algorithm.
        """
        return -1

    @staticmethod
    def _get_input_time_series(input_data, data_2=None, data_3=None, is_preview=False):
        """
        Returns a list of the distinct time series to be viewed
        Convert Original ArrayWrappers into a 2D list.
        :param input_data: Time series to display
        :type input_data: `TimeSeriesEEG`
        :param data_2: additional input data
        :param data_3: additional input data
        """
        original_timeseries = [input_data]

        error_sample = "The input TimeSeries have different sample periods. You cannot view them in the same time !"
        if data_2 is not None and data_2.gid != input_data.gid and is_preview is False:
            if data_2.sample_period != input_data.sample_period:
                raise LaunchException(error_sample)
            original_timeseries.append(data_2)

        if (data_3 is not None and data_3.gid != input_data.gid
                and (data_2 is None or data_2.gid != data_3.gid) and is_preview is False):
            if data_3.sample_period != input_data.sample_period:
                raise LaunchException(error_sample)
            original_timeseries.append(data_3)

        return original_timeseries

    def _compute_ag_settings(self, original_timeseries, is_preview, graph_labels, no_of_channels, total_time_length,
                             points_visible, is_extended_view, measure_points_sel_gids):
        # Compute distance between channels
        step, translations, channels_per_set = self.compute_required_info(original_timeseries)
        base_urls, page_size, total_pages, time_set_urls = self._get_data_set_urls(original_timeseries, is_preview)

        return dict(channelsPerSet=channels_per_set,
                    channelLabels=graph_labels,
                    noOfChannels=no_of_channels,
                    translationStep=step,
                    normalizedSteps=translations,
                    nan_value_found=self.has_nan,
                    baseURLS=base_urls,
                    pageSize=page_size,
                    nrOfPages=total_pages,
                    timeSetPaths=time_set_urls,
                    totalLength=total_time_length,
                    number_of_visible_points=points_visible,
                    extended_view=is_extended_view,
                    measurePointsSelectionGIDs=measure_points_sel_gids)

    def compute_parameters(self, input_data, data_2=None, data_3=None, is_preview=False,
                           is_extended_view=False, selected_dimensions=None):
        """
        Start the JS visualizer, similar to EEG-lab

        :param is_preview: Boolean True wne shown on Burst page
        :param selected_dimensions: from GUI int
        :param is_extended_view: Boolean when to display as dual-viewer
        :param input_data: Time series to display
        :type input_data: `TimeSeriesEEG`
        :param data_2: additional input data
        :param data_3: additional input data

        :returns: the needed parameters for a 2D representation
        :rtype: dict

        :raises LaunchException: when at least two input data parameters are provided and they sample periods differ
        """
        original_timeseries = self._get_input_time_series(input_data, data_2, data_3)
        self.selected_dimensions = selected_dimensions or [0, 2]

        # Hardcoded now 1st dimension is time
        h5_timeseries = []
        for timeseries in original_timeseries:
            if timeseries is None:
                continue
            ts_h5 = h5.h5_file_for_index(timeseries)
            assert isinstance(ts_h5, TimeSeriesH5)
            h5_timeseries.append(ts_h5)

        if not is_preview:
            max_chunck_length = max([ts_h5.read_data_shape()[0] for ts_h5 in h5_timeseries])
        else:
            max_chunck_length = min(self.preview_page_size, h5_timeseries[0].read_data_shape()[0])
        # compute how many elements will be visible on the screen
        points_visible = min(max_chunck_length, 500)

        (no_of_channels, ts_names, grouped_labels, total_time_length,
         graph_labels, initial_selections, measure_points_selectionGIDs,
         modes, state_vars) = self._pre_process(h5_timeseries)
        # ts_names : a string representing the time series
        # labels, modes, state_vars are maps ts_name -> list(...)
        # The label values must reach the client in ascending ordered. ts_names preserves the
        # order created by _pre_process
        if is_preview:
            total_time_length = max_chunck_length

        ag_settings = self._compute_ag_settings(h5_timeseries, is_preview, graph_labels, no_of_channels,
                                                total_time_length, points_visible, is_extended_view,
                                                measure_points_selectionGIDs)

        parameters = dict(title=self._get_sub_title(original_timeseries),
                          tsNames=ts_names,
                          groupedLabels=grouped_labels,
                          tsModes=modes,
                          tsStateVars=state_vars,
                          longestChannelLength=max_chunck_length,
                          label_x=self._get_label_x(original_timeseries[0]),
                          entities=h5_timeseries,
                          page_size=min(self.page_size, max_chunck_length),
                          number_of_visible_points=points_visible,
                          extended_view=is_extended_view,
                          initialSelection=initial_selections,
                          ag_settings=json.dumps(ag_settings))

        for ts_h5 in h5_timeseries:
            ts_h5.close()

        return parameters

    def _load_input_indexes(self, view_model):
        main_time_series_index = self.load_entity_by_gid(view_model.input_data)
        time_series_index2 = None
        time_series_index3 = None

        if view_model.data_2:
            time_series_index2 = self.load_entity_by_gid(view_model.data_2)
        if view_model.data_3:
            time_series_index3 = self.load_entity_by_gid(view_model.data_3)
        return main_time_series_index, time_series_index2, time_series_index3

    def launch(self, view_model):
        # type: (EegMonitorModel) -> dict
        """
        Compute visualizer's page
        """
        main_time_series_index, time_series_index2, time_series_index3 = self._load_input_indexes(view_model)
        params = self.compute_parameters(main_time_series_index, time_series_index2, time_series_index3)
        pages = dict(controlPage="eeg/controls", channelsPage="commons/channel_selector.html")
        return self.build_display_result("eeg/view", params, pages=pages)

    def _pre_process(self, timeseries_list):
        """From input, Compute no of lines and labels."""
        multiple_inputs = len(timeseries_list) > 1
        no_of_lines, max_length = 0, 0
        modes, state_vars = {}, {}
        # all these arrays are consistently indexed. At index idx they all refer to the same time series
        initial_selections, measures_sel_gids = [], []
        ts_names, graph_labels, grouped_labels = [], [], []

        for idx, timeseries in enumerate(timeseries_list):
            shape = timeseries.read_data_shape()
            no_of_lines += shape[self.selected_dimensions[1]]
            max_length = max(max_length, shape[0])

            self._fill_graph_labels(timeseries, graph_labels, multiple_inputs, idx)

            ts_name = timeseries.title.load()
            ts_names.append(ts_name)

            if multiple_inputs:
                # for multiple inputs the default selections might be too big: select the first few
                # warn: assumes that the selection values are a range
                initial_selections.append(list(range(4)))
            else:
                initial_selections.append(timeseries.get_default_selection())

            if isinstance(timeseries.get_measure_points_selection_gid(), str):
                measures_sel_gids.append(timeseries.get_measure_points_selection_gid())
            else:
                measures_sel_gids.append(timeseries.get_measure_points_selection_gid().hex)
            grouped_labels.append(self.get_grouped_space_labels(timeseries))

            state_vars[ts_name] = timeseries.labels_dimensions.load().get(timeseries.labels_ordering.load()[1], [])
            modes[ts_name] = list(range(shape[3]))

        return (no_of_lines, ts_names, grouped_labels, max_length, graph_labels,
                initial_selections, measures_sel_gids, modes, state_vars)

    def _fill_graph_labels(self, timeseries, graph_labels, mult_inp, idx):
        """ Fill graph labels in the graph_labels parameter """
        shape = timeseries.read_data_shape()
        space_labels = self.get_space_labels(timeseries)
        for j in range(shape[self.selected_dimensions[1]]):
            if space_labels:
                if j >= len(space_labels):
                    # for surface time series get_space_labels will return labels up to a limit,
                    # not a label for each signal.
                    # to honor that behaviour we break the loop if we run out of labels.
                    # todo a robust cap on signals. 
                    break
                this_label = str(space_labels[j])
            else:
                this_label = "channel_" + str(j)
            if mult_inp:
                this_label = str(idx + 1) + '.' + this_label
            graph_labels.append(this_label)

    def compute_required_info(self, list_of_timeseries):
        """Compute average difference between Max and Min."""
        # The values computed by this function will be serialized to json and passed to the client.
        # The time series might be of numpy.float32 a data type that is not serializable.
        # To overcome this we convert numpy scalars to python floats

        step = []
        translations = []
        channels_per_set = []
        for timeseries in list_of_timeseries:
            data_shape = timeseries.read_data_shape()
            resulting_shape = []
            for idx, shape in enumerate(data_shape):
                if idx in self.selected_dimensions:
                    resulting_shape.append(shape)

            page_chunk_data = timeseries.read_data_page(self.current_page * self.page_size,
                                                        (self.current_page + 1) * self.page_size)
            channels_per_set.append(int(resulting_shape[1]))

            for idx in range(resulting_shape[1]):
                self.has_nan = self.has_nan or replace_nan_values(page_chunk_data[:, idx])
                array_max = numpy.max(page_chunk_data[:, idx])
                array_min = numpy.min(page_chunk_data[:, idx])
                translations.append(float((array_max + array_min) / 2))
                if array_max == array_min:
                    array_max += 1
                step.append(abs(array_max - array_min))

        return float(max(step)), translations, channels_per_set

    @staticmethod
    def _get_sub_title(datatype_list):
        """ Compute sub-title for current page"""
        return "_".join(d.display_name for d in datatype_list)

    @staticmethod
    def _get_label_x(original_timeseries):
        """ Compute the label displayed on the x axis """
        return "Time(%s)" % original_timeseries.sample_period_unit

    def _get_data_set_urls(self, list_of_timeseries, is_preview=False):
        """
        Returns a list of lists. Each list contains the urls to the files
        containing the data for a certain array wrapper.
        """
        base_urls = []
        time_set_urls = []
        total_pages_set = []
        if is_preview is False:
            page_size = self.page_size
            for timeseries in list_of_timeseries:
                overall_shape = timeseries.read_data_shape()
                total_pages = overall_shape[0] // self.page_size
                if overall_shape[0] % self.page_size > 0:
                    total_pages += 1
                timeline_urls = []
                ts_gid = timeseries.gid.load().hex
                for i in range(total_pages):
                    current_max_size = min((i + 1) * self.page_size, overall_shape[0]) - i * self.page_size
                    params = "current_page=" + str(i) + ";page_size=" + str(self.page_size) + \
                             ";max_size=" + str(current_max_size)
                    timeline_urls.append(URLGenerator.build_h5_url(ts_gid, 'read_time_page', parameter=params))
                base_urls.append(URLGenerator.build_base_h5_url(ts_gid))
                time_set_urls.append(timeline_urls)
                total_pages_set.append(total_pages)
        else:
            ts_gid = list_of_timeseries[0].gid.load().hex
            base_urls.append(URLGenerator.build_base_h5_url(ts_gid))
            total_pages_set.append(1)
            page_size = self.preview_page_size
            params = "current_page=0;page_size=" + str(self.preview_page_size) + ";max_size=" + \
                     str(min(self.preview_page_size, list_of_timeseries[0].read_data_shape()[0]))
            time_set_urls.append([URLGenerator.build_h5_url(ts_gid, 'read_time_page', parameter=params)])
        return base_urls, page_size, total_pages_set, time_set_urls
