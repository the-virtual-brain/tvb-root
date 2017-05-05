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

import numpy
from tvb.analyzers.metrics_base import BaseTimeseriesMetricAlgorithm
from tvb.basic.traits.util import log_debug_array
from tvb.basic.traits.parameters_factory import get_traited_subclasses
from tvb.basic.filters.chain import FilterChain
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.abcadapter import ABCAsynchronous, ABCAdapter
from tvb.datatypes.time_series import TimeSeries
from tvb.datatypes.mapped_values import DatatypeMeasure


LOG = get_logger(__name__)



class TimeseriesMetricsAdapter(ABCAsynchronous):
    """
    TVB adapter for exposing as a group the measure algorithm.
    """

    _ui_name = "TimeSeries Metrics"
    _ui_description = "Compute a single number for a TimeSeries input DataType."
    _ui_subsection = "timeseries"
    available_algorithms = get_traited_subclasses(BaseTimeseriesMetricAlgorithm)


    def get_input_tree(self):
        """
        Compute interface based on introspected algorithms found.
        """
        algorithm = BaseTimeseriesMetricAlgorithm()
        algorithm.trait.bound = self.INTERFACE_ATTRIBUTES_ONLY
        tree = algorithm.interface[self.INTERFACE_ATTRIBUTES]
        tree[0]['conditions'] = FilterChain(fields=[FilterChain.datatype + '._nr_dimensions'],
                                            operations=["=="], values=[4])

        algo_names = self.available_algorithms.keys()
        options = []
        for name in algo_names:
            options.append({ABCAdapter.KEY_NAME: name, ABCAdapter.KEY_VALUE: name})
        tree.append({'name': 'algorithms', 'label': 'Selected metrics to be applied',
                     'type': ABCAdapter.TYPE_MULTIPLE, 'required': False, 'options': options,
                     'description': 'The selected metric algorithms will be applied on the input TimeSeries'})

        return tree


    def get_output(self):
        return [DatatypeMeasure]


    def configure(self, time_series, **kwargs):
        """
        Store the input shape to be later used to estimate memory usage.
        """
        self.input_shape = time_series.read_data_shape()


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
        """ 
        Launch algorithm and build results.

        :param time_series: the time series on which the algorithms are run
        :param algorithms:  the algorithms to be run for computing measures on the time series
        :type  algorithms:  any subclass of BaseTimeseriesMetricAlgorithm
                            (KuramotoIndex, GlobalVariance, VarianceNodeVariance)
        :rtype: `DatatypeMeasure`
        """
        if algorithms is None:
            algorithms = self.available_algorithms.keys()

        shape = time_series.read_data_shape()
        log_debug_array(LOG, time_series, "time_series")

        metrics_results = {}
        for algorithm_name in algorithms:
            ##------------- NOTE: Assumes 4D, Simulator timeSeries. --------------##
            node_slice = [slice(shape[0]), slice(shape[1]), slice(shape[2]), slice(shape[3])]

            ##---------- Iterate over slices and compose final result ------------##
            unstored_ts = TimeSeries(use_storage=False)

            unstored_ts.data = time_series.read_data_slice(tuple(node_slice))

            ##-------------------- Fill Algorithm for Analysis -------------------##
            algorithm = self.available_algorithms[algorithm_name](time_series=unstored_ts)
            if segment is not None:
                algorithm.segment = segment
            if start_point is not None:
                algorithm.start_point = start_point

            ## Validate that current algorithm's filter is valid.
            if (algorithm.accept_filter is not None and
                    not algorithm.accept_filter.get_python_filter_equivalent(time_series)):
                LOG.warning('Measure algorithm will not be computed because of incompatibility on input. '
                            'Filters failed on algo: ' + str(algorithm_name))
                continue
            else:
                LOG.debug("Applying measure: " + str(algorithm_name))

            unstored_result = algorithm.evaluate()
            ##----------------- Prepare a Float object(s) for result ----------------##
            if isinstance(unstored_result, dict):
                metrics_results.update(unstored_result)
            else:
                metrics_results[algorithm_name] = unstored_result

        result = DatatypeMeasure(analyzed_datatype=time_series, storage_path=self.storage_path,
                                 data_name=self._ui_name, metrics=metrics_results)
        return result


