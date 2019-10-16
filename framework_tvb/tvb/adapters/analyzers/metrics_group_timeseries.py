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
from collections import OrderedDict
from tvb.analyzers.metrics_base import BaseTimeseriesMetricAlgorithm
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.abcadapter import ABCAsynchronous, ABCAdapterForm
from tvb.adapters.datatypes.h5.mapped_value_h5 import DatatypeMeasureH5
from tvb.core.entities.filters.chain import FilterChain
from tvb.adapters.datatypes.db.mapped_value import DatatypeMeasureIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import DataTypeSelectField, ScalarField, MultipleSelectField
# Import metrics here, so that Traits will find them and return them as known subclasses
import tvb.analyzers.metric_kuramoto_index
import tvb.analyzers.metric_proxy_metastability
import tvb.analyzers.metric_variance_global
import tvb.analyzers.metric_variance_of_node_variance

LOG = get_logger(__name__)
ALGORITHMS = BaseTimeseriesMetricAlgorithm.get_known_subclasses(include_itself=False)


class TimeseriesMetricsAdapterForm(ABCAdapterForm):

    @staticmethod
    def get_extra_algorithm_filters():
        return {"KuramotoIndex": FilterChain(fields=[FilterChain.datatype + '.data_length_2d'], operations=[">="],
                                             values=[2])}

    def __init__(self, prefix='', project_id=None):
        super(TimeseriesMetricsAdapterForm, self).__init__(prefix, project_id)
        self.time_series = DataTypeSelectField(self.get_required_datatype(), self, name="time_series",
                                               required=True, label=BaseTimeseriesMetricAlgorithm.time_series.label,
                                               doc = BaseTimeseriesMetricAlgorithm.time_series.doc)
        self.start_point = ScalarField(BaseTimeseriesMetricAlgorithm.start_point, self)
        self.segment = ScalarField(BaseTimeseriesMetricAlgorithm.segment, self)

        algo_names = list(ALGORITHMS)
        algo_names.sort()
        choices = OrderedDict()
        for name in algo_names:
            choices[name] = name

        self.algorithms = MultipleSelectField(choices, self, name="algorithms", include_none=False,
                                              label='Selected metrics to be applied',
                                              doc='The selected algorithms will all be applied on the input TimeSeries')

    @staticmethod
    def get_required_datatype():
        return TimeSeriesIndex

    @staticmethod
    def get_input_name():
        return '_time_series'

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

    def configure(self, time_series, **kwargs):
        """
        Store the input shape to be later used to estimate memory usage.
        """
        self.input_shape = (time_series.data_length_1d, time_series.data_length_2d,
                            time_series.data_length_3d, time_series.data_length_4d)

    def get_required_memory_size(self, **kwargs):
        """
        Return the required memory to run this algorithm.
        """
        input_size = numpy.prod(self.input_shape) * 8.0
        return input_size

    def get_required_disk_size(self, **kwargs):
        """
        Returns the required disk size to be able to run the adapter (in kB).
        """
        return 0

    def launch(self, time_series, algorithms=None, start_point=None, segment=None):
        # type: (TimeSeriesIndex, list, float, int) -> DatatypeMeasureIndex
        """ 
        Launch algorithm and build results.

        :param time_series: the time series on which the algorithms are run
        :param algorithms:  the algorithms to be run for computing measures on the time series
        :type  algorithms:  any subclass of BaseTimeseriesMetricAlgorithm
                            (KuramotoIndex, GlobalVariance, VarianceNodeVariance)
        :rtype: `DatatypeMeasureIndex`
        """
        if algorithms is None:
            algorithms = list(ALGORITHMS)

        LOG.debug("time_series shape is %s" % str(self.input_shape))
        dt_timeseries = h5.load_from_index(time_series)

        metrics_results = {}
        for algorithm_name in algorithms:

            algorithm = ALGORITHMS[algorithm_name](time_series=dt_timeseries)
            if segment is not None:
                algorithm.segment = segment
            if start_point is not None:
                algorithm.start_point = start_point

            # Validate that current algorithm's filter is valid.
            algorithm_filter = TimeseriesMetricsAdapterForm.get_extra_algorithm_filters().get(algorithm_name)
            if algorithm_filter is not None and not algorithm_filter.get_python_filter_equivalent(time_series):
                LOG.warning('Measure algorithm will not be computed because of incompatibility on input. '
                            'Filters failed on algo: ' + str(algorithm_name))
                continue
            else:
                LOG.debug("Applying measure: " + str(algorithm_name))

            unstored_result = algorithm.evaluate()
            # ----------------- Prepare a Float object(s) for result ----------------##
            if isinstance(unstored_result, dict):
                metrics_results.update(unstored_result)
            else:
                metrics_results[algorithm_name] = unstored_result

        result = DatatypeMeasureIndex()
        result.source_gid = time_series.gid
        result.metrics = json.dumps(metrics_results)

        result_path = h5.path_for(self.storage_path, DatatypeMeasureH5, result.gid)
        with DatatypeMeasureH5(result_path) as result_h5:
            result_h5.metrics.store(metrics_results)
            result_h5.analyzed_datatype.store(dt_timeseries)
            result_h5.gid.store(uuid.UUID(result.gid))

        return result
