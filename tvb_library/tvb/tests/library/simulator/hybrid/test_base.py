# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
Shared fixtures and helper utilities for the hybrid-simulator test suite.

This module provides :class:`BaseHybridTest`, which every hybrid test class
inherits from.  It wires up a realistic two-subnetwork topology:

* **cortex** — `JansenRit` model, nodes assigned by random partition of the
  default 76-region TVB connectivity.
* **thalamus** — `ReducedSetFitzHughNagumo` (FHN) model, remaining nodes.

The two subnetworks are coupled via three :class:`InterProjection` edges that
cover both cross-model cvar combinations (JR→FHN and FHN→JR), using sparse
weight and length matrices extracted from the global connectivity.
"""

import numpy as np
import scipy.sparse
from tvb.tests.library.base_testcase import BaseTestCase
from tvb.simulator.models import JansenRit, ReducedSetFitzHughNagumo
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.hybrid import NetworkSet, Subnetwork, InterProjection
from tvb.datatypes.connectivity import Connectivity


class BaseHybridTest(BaseTestCase):
    """
    Base class for hybrid module tests.

    Provides a realistic two-subnetwork topology wired from the default TVB
    connectivity and helper methods for generating random state/cvar arrays of
    the correct shape.

    Subnetwork convention
    ---------------------
    * **cortex** — :class:`~tvb.simulator.models.JansenRit` (6 state vars,
      2 coupling vars), nodes where ``ix == AtlasPart.CORTEX``.
    * **thalamus** — :class:`~tvb.simulator.models.ReducedSetFitzHughNagumo`
      (4 state vars, 2 coupling vars, 3 modes), nodes where
      ``ix == AtlasPart.THALAMUS``.

    The random node-to-subnetwork assignment is seeded (``np.random.seed(42)``)
    so results are reproducible across test runs.
    """
    
    def setup(self, nvois=None, jrmon=None):
        """
        Build and return the canonical cortex-thalamus hybrid network.

        Constructs two configured :class:`~tvb.simulator.hybrid.Subnetwork`
        objects and three :class:`~tvb.simulator.hybrid.InterProjection` edges
        (cortex→thalamus  × 2, thalamus→cortex × 1) assembled into a
        :class:`~tvb.simulator.hybrid.NetworkSet`.  Projection buffers are
        fully configured on return.

        Parameters
        ----------
        nvois : int, optional
            When given, restricts both models to their first *nvois* variables
            of interest.  Useful for tests that check monitor output shapes.
        jrmon : Monitor, optional
            If provided, the monitor is added to the cortex subnetwork via
            :meth:`~tvb.simulator.hybrid.Subnetwork.add_monitor` before the
            :class:`~tvb.simulator.hybrid.NetworkSet` is assembled.

        Returns
        -------
        conn : Connectivity
            The full 76-region TVB connectivity used to derive weights.
        ix : ndarray of int, shape (n_regions,)
            Random 0/1 partition assigning each region to cortex (0) or
            thalamus (1).
        cortex : Subnetwork
            Configured JansenRit subnetwork.
        thalamus : Subnetwork
            Configured ReducedSetFitzHughNagumo subnetwork.
        AtlasPart : type
            Simple namespace with ``CORTEX = 0`` and ``THALAMUS = 1``
            constants used to index into *ix*.
        nets : NetworkSet
            Fully assembled and buffer-configured network with all three
            inter-subnet projections.
        """
        conn = Connectivity.from_file()
        conn.configure()
        np.random.seed(42)

        # Define brain regions
        class AtlasPart:
            CORTEX = 0
            THALAMUS = 1

        ix = np.random.randint(
            low=0, high=2, size=conn.number_of_regions)

        scheme = HeunDeterministic(dt=0.1)

        jrkwargs = {}
        fhnkwargs = {}
        if nvois:
            jrkwargs['variables_of_interest'] = JansenRit.variables_of_interest.default[:nvois]
            fhnkwargs['variables_of_interest'] = ReducedSetFitzHughNagumo.variables_of_interest.default[:nvois]

        # Create subnetworks with just their size and behavior
        cortex = Subnetwork(
            name='cortex',
            model=JansenRit(**jrkwargs),
            scheme=scheme,
            nnodes=(ix == AtlasPart.CORTEX).sum(),
        ).configure()

        if jrmon:
            cortex.add_monitor(jrmon)

        thalamus = Subnetwork(
            name='thalamus',
            model=ReducedSetFitzHughNagumo(**fhnkwargs),
            scheme=scheme,
            nnodes=(ix == AtlasPart.THALAMUS).sum(),
        ).configure()

        # Create projections with explicit weights from global connectivity
        cortex_indices = np.where(ix == AtlasPart.CORTEX)[0]
        thalamus_indices = np.where(ix == AtlasPart.THALAMUS)[0]

        # Use valid coupling variable indices for each model
        # JansenRit has 2 coupling variables (0,1)
        # ReducedSetFitzHughNagumo has 2 coupling variables (0,1)

        # Prepare sparse weights and dummy lengths/params for projections
        weights_c_t_dense = conn.weights[thalamus_indices][:, cortex_indices]
        weights_c_t_sparse = scipy.sparse.csr_matrix(weights_c_t_dense)
        weights_t_c_dense = conn.weights[cortex_indices][:, thalamus_indices]
        weights_t_c_sparse = scipy.sparse.csr_matrix(weights_t_c_dense)

        # Create sparse lengths with same sparsity pattern as weights
        # Dummy lengths data [0, 10)
        lengths_c_t_data = np.random.rand(weights_c_t_sparse.nnz) * 10.0
        lengths_c_t_sparse = scipy.sparse.csr_matrix(
            (lengths_c_t_data, weights_c_t_sparse.indices, weights_c_t_sparse.indptr),
            shape=weights_c_t_sparse.shape)
        # Dummy lengths data [0, 10)
        lengths_t_c_data = np.random.rand(weights_t_c_sparse.nnz) * 10.0
        lengths_t_c_sparse = scipy.sparse.csr_matrix(
            (lengths_t_c_data, weights_t_c_sparse.indices, weights_t_c_sparse.indptr),
            shape=weights_t_c_sparse.shape)

        # Dummy delay parameters (can be overridden in specific tests if needed)
        default_cv = 3.0
        default_dt = scheme.dt

        nets = NetworkSet(
            subnets=[cortex, thalamus],
            projections=[
                InterProjection(
                    source=cortex, target=thalamus,
                    source_cvar=np.r_[0], target_cvar=np.r_[1],
                    weights=weights_c_t_sparse,
                    scale=1e-4,
                    lengths=lengths_c_t_sparse, cv=default_cv, dt=default_dt, # Use sparse lengths
                ),
                InterProjection(
                    source=cortex, target=thalamus,
                    source_cvar=np.r_[1], target_cvar=np.r_[0],
                    weights=weights_c_t_sparse,
                    scale=1e-4,
                     # Reusing weights, so must reuse sparse lengths for consistency
                    lengths=lengths_c_t_sparse, cv=default_cv, dt=default_dt, # Use sparse lengths
                ),
                InterProjection(
                    source=thalamus, target=cortex,
                    source_cvar=np.r_[0], target_cvar=np.r_[1],
                    scale=1e-4,
                    weights=weights_t_c_sparse,
                    lengths=lengths_t_c_sparse, cv=default_cv, dt=default_dt, # Use sparse lengths
                ),
            ]
        )

        # Configure buffers for all projections in the NetworkSet
        for p in nets.projections:
            p.configure_buffer(
                n_vars_src=p.source.model.nvar,
                n_nodes_src=p.source.nnodes,
                n_modes_src=p.source.model.number_of_modes
            )

        return conn, ix, cortex, thalamus, AtlasPart, nets

    def _randn_like_states(self, states):
        """
        Return a new state container filled with independent standard normals.

        The returned object has the same type and per-array shapes as *states*,
        making it suitable as a randomised initial condition or perturbation
        when the exact values are not under test.

        Parameters
        ----------
        states : namedtuple or similar container of ndarray
            Template container whose element shapes are replicated.  Typically
            obtained from :meth:`~tvb.simulator.hybrid.NetworkSet.zero_states`.

        Returns
        -------
        same type as states
            Container of i.i.d. ``N(0, 1)`` arrays with identical shapes.
        """
        return states.__class__(*(np.random.randn(*x.shape) for x in states))

    def _randn_like_cvars(self, cvars):
        """
        Return a new cvar container filled with independent standard normals.

        Mirrors :meth:`_randn_like_states` but operates on coupling-variable
        containers of shape ``(ncvars, nnodes, nmodes)`` per subnetwork.

        Parameters
        ----------
        cvars : namedtuple or similar container of ndarray
            Template container whose element shapes are replicated.  Typically
            obtained from :meth:`~tvb.simulator.hybrid.NetworkSet.zero_cvars`.

        Returns
        -------
        same type as cvars
            Container of i.i.d. ``N(0, 1)`` arrays with identical shapes.
        """
        return cvars.__class__(*(np.random.randn(*x.shape) for x in cvars)) 
