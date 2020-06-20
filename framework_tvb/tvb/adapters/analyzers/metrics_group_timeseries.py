# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
Adapter that uses the traits module to generate interfaces for group of 
Analyzer used to calculate a single measure for TimeSeries.

.. moduleauthor:: Paula Sanz Leon <pau.sleon@gmail.com>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>

"""
import uuid
import numpy
import json
from tvb.adapters.datatypes.h5.mapped_value_h5 import DatatypeMeasureH5
from tvb.adapters.datatypes.db.mapped_value import DatatypeMeasureIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex
from tvb.analyzers.metrics_base import BaseTimeseriesMetricAlgorithm
from tvb.basic.neotraits.api import List
from tvb.config import choices, ALGORITHMS
from tvb.core.adapters.abcadapter import ABCAsynchronous, ABCAdapterForm
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import ScalarField, TraitDataTypeSelectField, MultiSelectField
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.datatypes.time_series import TimeSeries


class TimeseriesMetricsAdapterModel(ViewModel, BaseTimeseriesMetricAlgorithm):
    time_series = DataTypeGidAttr(
        linked_datatype=TimeSeries,
        label="Time Series",
        required=True,
        doc="The TimeSeries for which the metric(s) will be computed."
    )

    algorithms = List(
        of=str,
        choices=tuple(choices.values()),
        label='Selected metrics to be applied',
        doc='The selected algorithms will all be applied on the input TimeSeries'
    )


class TimeseriesMetricsAdapterForm(ABCAdapterForm):

    @staticmethod
    def get_extra_algorithm_filters():
        return {"KuramotoIndex": FilterChain(fields=[FilterChain.datatype + '.data_length_2d'], operations=[">="],
                                             values=[2])}

    def __init__(self, prefix='', project_id=None):
        super(TimeseriesMetricsAdapterForm, self).__init__(prefix, project_id)
        self.time_series = TraitDataTypeSelectField(TimeseriesMetricsAdapterModel.time_series, self, name="time_series")
        self.start_point = ScalarField(TimeseriesMetricsAdapterModel.start_point, self)
        self.segment = ScalarField(TimeseriesMetricsAdapterModel.segment, self)
        self.algorithms = MultiSelectField(TimeseriesMetricsAdapterModel.algorithms, self, name="algorithms")

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


class TimeseriesMetricsAdapter(ABCAsynchronous):
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
        self.input_time_series_index = self.load_entity_by_gid(view_model.time_series.hex)
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

        :param time_series: the time series on which the algorithms are run
        :param algorithms:  the algorithms to be run for computing measures on the time series
        :type  algorithms:  any subclass of BaseTimeseriesMetricAlgorithm
                            (KuramotoIndex, GlobalVariance, VarianceNodeVariance)
        :rtype: `DatatypeMeasureIndex`
        """
        algorithms = view_model.algorithms
        if algorithms is None or len(algorithms) == 0:
            algorithms = list(ALGORITHMS)

        self.log.debug("time_series shape is %s" % str(self.input_shape))
        dt_timeseries = h5.load_from_index(self.input_time_series_index)

        metrics_results = {}
        for algorithm_name in algorithms:

            algorithm = ALGORITHMS[algorithm_name](time_series=dt_timeseries)
            if view_model.segment is not None:
                algorithm.segment = view_model.segment
            if view_model.start_point is not None:
                algorithm.start_point = view_model.start_point

            # Validate that current algorithm's filter is valid.
            algorithm_filter = TimeseriesMetricsAdapterForm.get_extra_algorithm_filters().get(algorithm_name)
            if algorithm_filter is not None \
                    and not algorithm_filter.get_python_filter_equivalent(self.input_time_series_index):
                self.log.warning('Measure algorithm will not be computed because of incompatibility on input. '
                                 'Filters failed on algo: ' + str(algorithm_name))
                continue
            else:
                self.log.debug("Applying measure: " + str(algorithm_name))

            unstored_result = algorithm.evaluate()
            # ----------------- Prepare a Float object(s) for result ----------------##
            if isinstance(unstored_result, dict):
                metrics_results.update(unstored_result)
            else:
                metrics_results[algorithm_name] = unstored_result

        result = DatatypeMeasureIndex()
        result.fk_source_gid = self.input_time_series_index.gid
        result.metrics = json.dumps(metrics_results)

        result_path = h5.path_for(self.storage_path, DatatypeMeasureH5, result.gid)
        with DatatypeMeasureH5(result_path) as result_h5:
            result_h5.metrics.store(metrics_results)
            result_h5.analyzed_datatype.store(dt_timeseries)
            result_h5.gid.store(uuid.UUID(result.gid))

        return result
