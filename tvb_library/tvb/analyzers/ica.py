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
Perform Independent Component Analysis on a TimeSeries Object and returns an
IndependentComponents datatype.

.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>

"""

import numpy
import tvb.datatypes.mode_decompositions as mode_decompositions
from tvb.analyzers.ica_algorithm import fastica


"""
Takes a TimeSeries datatype (x) and returns the unmixed temporal sources (S) 
and the estimated mixing matrix (A).
    
:math: x = A S
    
ICA takes time-points as observations and nodes as variables.
    
It uses the FastICA algorithm implemented in the scikit-learn toolkit, and
its intended usage is as a `blind source separation` method.
    
See also: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.fastica.html#sklearn.decomposition.fastica

"""


def compute_ica_decomposition(time_series, n_components):
    """
    # type: (TimeSeries, int) -> IndependentComponents
    Run FastICA on the given time series data.

    Parameters
    __________

    time_series : TimeSeries
    The timeseries to which the ICA is to be applied.

    n_components : int
    Number of principal components to unmix.
    """

    # problem dimensions
    data = time_series.data
    n_time, n_svar, n_node, n_mode = data.shape
    n_components = n_comp = n_components or n_node

    if n_time < n_comp:
        msg = ("ICA requires more time points (received %d) than number of components (received %d)."
               " Please run a longer simulation, use a higher sampling frequency or specify a lower"
               " number of components to extract.")
        msg %= n_time, n_comp
        raise ValueError(msg)

    # ICA operates on matrices, here we perform for all state variables and modes
    W = numpy.zeros((n_comp, n_comp, n_svar, n_mode))  # unmixing
    K = numpy.zeros((n_comp, n_node, n_svar, n_mode))  # whitening matrix
    src = numpy.zeros((n_time, n_comp, n_svar, n_mode))  # component time series

    for mode in range(n_mode):
        for var in range(n_svar):
            sl = Ellipsis, var, mode
            K[sl], W[sl], src[sl] = fastica(data[:, var, :, mode], n_components)

    return mode_decompositions.IndependentComponents(source=time_series, component_time_series=src,
                                                     prewhitening_matrix=K, unmixing_matrix=W, n_components=n_comp)
