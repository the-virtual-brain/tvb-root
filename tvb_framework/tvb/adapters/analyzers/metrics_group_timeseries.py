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
Adapter that uses the traits module to generate interfaces for group of 
Analyzer used to calculate a single measure for TimeSeries.

.. moduleauthor:: Paula Sanz Leon <paula@tvb.invalid>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>

"""
import numpy
from tvb.adapters.datatypes.db.mapped_value import DatatypeMeasureIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex
from tvb.basic.neotraits.api import Int, Float
from tvb.basic.neotraits.api import List
from tvb.config import ALGORITHMS
from tvb.core.adapters.abcadapter import ABCAdapterForm, ABCAdapter
from tvb.core.entities.file.simulator.datatype_measure_h5 import DatatypeMeasure
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import TraitDataTypeSelectField, MultiSelectField, FloatField, IntField
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.datatypes.time_series import TimeSeries


class TimeseriesMetricsAdapterModel(ViewModel):
    time_series = DataTypeGidAttr(
        linked_datatype=TimeSeries,
        label="Time Series",
        required=True,
        doc="The TimeSeries for which the metric(s) will be computed."
    )

    algorithms = List(
        of=str,
        choices=tuple(ALGORITHMS.keys()),
        label='Selected metrics to be applied',
        doc='The selected algorithms will all be applied on the input TimeSeries'
    )

    start_point = Float(
        label="Start point (ms)",
        default=500.0,
        required=False,
        doc=""" The start point determines how many points of the TimeSeries will
        be discarded before computing the metric. By default it drops the
        first 500 ms.""")

    segment = Int(
        label="Segmentation factor",
        default=4,
        required=False,
        doc=""" Divide the input time-series into discrete equally sized sequences and
        use the last segment to compute the metric. It is only used when
        the start point is larger than the time-series length.""")


class TimeseriesMetricsAdapterForm(ABCAdapterForm):

    @staticmethod
    def get_extra_algorithm_filters():
        return {"KuramotoIndex": FilterChain(fields=[FilterChain.datatype + '.data_length_2d'], operations=[">="],
                                             values=[2])}

    def __init__(self):
        super(TimeseriesMetricsAdapterForm, self).__init__()
        self.time_series = TraitDataTypeSelectField(TimeseriesMetricsAdapterModel.time_series, name="time_series")
        self.start_point = FloatField(TimeseriesMetricsAdapterModel.start_point)
        self.segment = IntField(TimeseriesMetricsAdapterModel.segment)
        self.algorithms = MultiSelectField(TimeseriesMetricsAdapterModel.algorithms, name="algorithms")

    @staticmethod
    def get_view_model():
        return TimeseriesMetricsAdapterModel

    @staticmethod
    def get_required_datatype():
        return TimeSeriesIndex

    @staticmethod
    def get_input_name():
        return 'time_series'

    @staticmethod
    def get_filters():
        return FilterChain(fields=[FilterChain.datatype + '.data_ndim'], operations=["=="], values=[4])


class TimeseriesMetricsAdapter(ABCAdapter):
    """
    TVB adapter for exposing as a group the measure algorithm.
    """

    _ui_name = "TimeSeries Metrics"
    _ui_description = "Compute a single number for a TimeSeries input DataType."
    _ui_subsection = "timeseries"
    input_shape = ()

    def get_form_class(self):
        return TimeseriesMetricsAdapterForm

    def get_output(self):
        return [DatatypeMeasureIndex]

    def configure(self, view_model):
        # type: (TimeseriesMetricsAdapterModel) -> None
        """
        Store the input shape to be later used to estimate memory usage.
        """
        self.input_time_series_index = self.load_entity_by_gid(view_model.time_series)
        self.input_shape = (self.input_time_series_index.data_length_1d,
                            self.input_time_series_index.data_length_2d,
                            self.input_time_series_index.data_length_3d,
                            self.input_time_series_index.data_length_4d)

    def get_required_memory_size(self, view_model):
        # type: (TimeseriesMetricsAdapterModel) -> int
        """
        Return the required memory to run this algorithm.
        """
        input_size = numpy.prod(self.input_shape) * 8.0
        return input_size

    def get_required_disk_size(self, view_model):
        # type: (TimeseriesMetricsAdapterModel) -> int
        """
        Returns the required disk size to be able to run the adapter (in kB).
        """
        return 0

    def launch(self, view_model):
        # type: (TimeseriesMetricsAdapterModel) -> [DatatypeMeasureIndex]
        """ 
        Launch algorithm and build results.
        :param view_model: the ViewModel keeping the algorithm inputs
        """
        algorithms = view_model.algorithms
        if algorithms is None or len(algorithms) == 0:
            algorithms = list(ALGORITHMS)

        self.log.debug("time_series shape is %s" % str(self.input_shape))
        dt_timeseries = self.load_traited_by_gid(self.input_time_series_index.gid)

        metrics_results = {}
        for algorithm_name in algorithms:

            algorithm_func = ALGORITHMS[algorithm_name]

            # Validate that current algorithm's filter is valid.
            algorithm_filter = TimeseriesMetricsAdapterForm.get_extra_algorithm_filters().get(algorithm_name)
            if algorithm_filter is not None \
                    and not algorithm_filter.get_python_filter_equivalent(self.input_time_series_index):
                self.log.warning('Measure algorithm will not be computed because of incompatibility on input. '
                                 'Filters failed on algo: ' + str(algorithm_name))
                continue
            else:
                self.log.debug("Applying measure: " + str(algorithm_name))

            unstored_result = algorithm_func({'time_series': dt_timeseries, 'start_point': view_model.start_point,
                                              'segment': view_model.segment})
            # ----------------- Prepare a Float object(s) for result ----------------##
            if isinstance(unstored_result, dict):
                metrics_results.update(unstored_result)
            else:
                metrics_results[algorithm_name] = unstored_result

        dt_metric = DatatypeMeasure(analyzed_datatype=dt_timeseries, metrics=metrics_results)

        result = self.store_complete(dt_metric)

        return result
