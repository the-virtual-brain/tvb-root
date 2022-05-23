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
Perform Principal Component Analysis (PCA) on a TimeSeries datatype and return
a PrincipalComponents datatype.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""

import numpy
import tvb.datatypes.mode_decompositions as mode_decompositions
from tvb.basic.logger.builder import get_logger
from tvb.basic.neotraits.api import HasTraits, Attr, narray_describe

log = get_logger(__name__)


def _compute_weights_and_fractions(data):
    """
    The code for this function has been taken and adapted from Matplotlib 2.1.0
    Aug 2019
    """
    n, m = data.shape
    if n < m:
        raise RuntimeError('we assume data in input is organized with '
                            'numrows>numcols')

    mu = data.mean(axis=0)
    sigma = data.std(axis=0)

    a = (data - mu) / sigma
    U, s, Vh = numpy.linalg.svd(a, full_matrices=False)
    Wt = Vh
    s = s ** 2

    vars = s / len(s)
    fracs = vars / vars.sum()

    return fracs, Wt

"""
Return principal component weights and the fraction of the variance that 
they explain. 
    
PCA takes time-points as observations and nodes as variables.
    
NOTE: The TimeSeries must be longer(more time-points) than the number of
        nodes -- Mostly a problem for TimeSeriesSurface datatypes, which, if 
        sampled at 1024Hz, would need to be greater than 16 seconds long.
"""


# # TODO: Maybe should support first N components or neccessary components to
# #      explain X% of the variance. NOTE: For default surface the weights
# #      matrix has a size ~ 2GB * modes * vars...
def compute_pca(time_series):
    """
    # type: (TimeSeries)  -> PrincipalComponents
    Compute the temporal covariance between nodes in the time_series.

    Parameters
    __________
    time_series : TimeSeries
    The timeseries to which the PCA is to be applied.
    """

    ts_shape = time_series.data.shape

    # Need more measurements than variables
    if ts_shape[0] < ts_shape[2]:
        msg = "PCA requires a longer timeseries (tpts > number of nodes)."
        log.error(msg)
        raise Exception(msg)

    # (nodes, nodes, state-variables, modes)
    weights_shape = (ts_shape[2], ts_shape[2], ts_shape[1], ts_shape[3])
    log.info("weights shape will be: %s" % str(weights_shape))

    fractions_shape = (ts_shape[2], ts_shape[1], ts_shape[3])
    log.info("fractions shape will be: %s" % str(fractions_shape))

    weights = numpy.zeros(weights_shape)
    fractions = numpy.zeros(fractions_shape)

    # One inter-node temporal covariance matrix for each state-var & mode.
    for mode in range(ts_shape[3]):
        for var in range(ts_shape[1]):
            data = time_series.data[:, var, :, mode]

            fracts, w = _compute_weights_and_fractions(data)
            fractions[:, var, mode] = fracts
            weights[:, :, var, mode] = w

    log.debug("fractions")
    log.debug(narray_describe(fractions))
    log.debug("weights")
    log.debug(narray_describe(weights))

    pca_result = mode_decompositions.PrincipalComponents(
        source=time_series,
        fractions=fractions,
        weights=weights,
        norm_source=numpy.array([]),
        component_time_series=numpy.array([]),
        normalised_component_time_series=numpy.array([]))

    return pca_result
