# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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
Filler analyzer: Takes a TimeSeries object and returns two Floats.


These metrics are described and used in:

Hellyer et al. The Control of Global Brain Dynamics: Opposing Actions
of Frontoparietal Control and Default Mode Networks on Attention. 
The Journal of Neuroscience, January 8, 2014,  34(2):451– 461

Proxy of spatial coherence (V): 

Proxy metastability (M): the variability in spatial coherence of the signal
globally or locally (within a network) over time.

Proxy synchrony (S) : the reciprocal of mean spatial variance across time.

.. moduleauthor:: Paula Sanz Leon <paulala@tvb.invalid>

"""

import numpy
from tvb.basic.logger.builder import get_logger


def remove_mean(x, axis):
    """
    Remove mean from numpy array along axis
    """
    # Example for demean(x, 2) with x.shape == 2,3,4,5
    # m = x.mean(axis=2) collapses the 2'nd dimension making m and x incompatible
    # so we add it back m[:,:, np.newaxis, :]
    # Since the shape and axis are known only at runtime
    # Calculate the slicing dynamically
    return x - numpy.expand_dims(x.mean(axis=axis), axis)


r"""
Subtract the mean time-series and compute. 

Input:
TimeSeries DataType
    
Output: 
Float, Float
    
The two metrics given by this analyzers are a proxy for metastability and synchrony. 
The underlying dynamical model used in the article was the Kuramoto model.

.. math::
        V(t) &= \frac{1}{N} \sum_{i=1}^{N} |S_i(t) - <S(t)>| \\
        M(t) &= \sqrt{E[V(t)^{2}]-(E[V(t)])^{2}} \\
        S(t) &= \frac{1}{\bar{V(t)}}

"""

log = get_logger(__name__)


def compute_proxy_metastability_metric(params):
    """
    # type: dict(TimeSeries, float, int) -> (float, float)
    Compute the zero centered variance of node variances for the time_series.

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
    time_series_diffs = remove_mean(time_series.data[start_tpt:, :], axis=2)
    v_data = abs(time_series_diffs).mean(axis=2)

    # handle state-variables & modes
    cat_tpts = v_data.shape[0] * shape[1] * shape[3]
    v_data = v_data.reshape((cat_tpts,), order="F")
    # std across time-points
    metastability = v_data.std(axis=0)
    synchrony = 1. / v_data.mean(axis=0)

    return {"Metastability": metastability,
            "Synchrony": synchrony}
