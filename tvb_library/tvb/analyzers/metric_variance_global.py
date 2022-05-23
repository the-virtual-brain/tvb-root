# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
Filler analyzer: Takes a TimeSeries object and returns a Float.

.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""

from tvb.basic.logger.builder import get_logger

"""
Zero-centres all the time-series and then calculates the variance over all 
data points.
    
Input:
TimeSeries DataType
    
Output: 
Float
    
This is a crude indicator of "excitability" or oscillation amplitude of the
models over the entire network.
"""

log = get_logger(__name__)


def compute_variance_global_metric(params):
    """
    # type: dict(TimeSeries, float, int) -> float
    Compute the zero centered global variance of the time_series.

    Parameters
    ----------
    params : a dictionary containing
        time_series : TimeSeries
        Input time series for which the metric will be computed.

        start_point : float
        Determines how many points of the TimeSeries will be discarded before computing the metric

        segment : int
        Divides the input time-series into discrete equally sized sequences and use the last segment to compute
        the metric. Only used when the start point is larger than the time-series length
    """

    time_series = params['time_series']
    start_point = params['start_point']
    segment = params['segment']

    shape = time_series.data.shape
    tpts = shape[0]

    if start_point != 0.0:
        start_tpt = start_point / time_series.sample_period
        log.debug("Will discard: %s time points" % start_tpt)
    else:
        start_tpt = 0

    if start_tpt > tpts:
        log.warning("The time-series is shorter than the starting point")
        log.debug("Will divide the time-series into %d segments." % segment)
        # Lazy strategy
        start_tpt = int((segment - 1) * (tpts // segment))

    start_tpt = int(start_tpt)
    zero_mean_data = (time_series.data[start_tpt:, :] - time_series.data[start_tpt:, :].mean(axis=0))
    global_variance = zero_mean_data.var()

    return global_variance
