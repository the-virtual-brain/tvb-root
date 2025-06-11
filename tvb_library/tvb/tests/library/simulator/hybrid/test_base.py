"""
Base test utilities for hybrid module tests.
"""

import numpy as np
import scipy.sparse
from tvb.tests.library.base_testcase import BaseTestCase
from tvb.simulator.models import JansenRit, ReducedSetFitzHughNagumo
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.hybrid import NetworkSet, Subnetwork, InterProjection
from tvb.datatypes.connectivity import Connectivity


class BaseHybridTest(BaseTestCase):
    """Base class for hybrid module tests with common setup and utilities."""
    
    def setup(self, nvois=None, jrmon=None):
        """Setup test environment with cortex-thalamus network.
        
        Parameters
        ----------
        nvois : int, optional
            Number of variables of interest to use
        jrmon : Monitor, optional
            Monitor to attach to Jansen-Rit model
        use_zero_lengths : bool, optional
            If True, set up projections with zero lengths for minimal delay.
            
        Returns
        -------
        tuple
            (conn, ix, cortex, thalamus, AtlasPart, nets)
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
        """Generate random states with same structure as input states.
        
        Parameters
        ----------
        states : NetworkSet.States
            Template states to match structure of
            
        Returns
        -------
        NetworkSet.States
            New states filled with random numbers
        """
        return states.__class__(*(np.random.randn(*x.shape) for x in states))

    def _randn_like_cvars(self, cvars):
        """Generate random coupling variables with same structure as input cvars.
        
        Parameters
        ----------
        cvars : NetworkSet.States
            Template coupling variables to match structure of
            
        Returns
        -------
        NetworkSet.States
            New coupling variables filled with random numbers
        """
        return cvars.__class__(*(np.random.randn(*x.shape) for x in cvars)) 
