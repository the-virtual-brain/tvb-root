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
Unit tests for :mod:`tvb.simulator.hybrid.projection_utils`.

Covers the factory functions that build projection objects from high-level
descriptions:

* :func:`~tvb.simulator.hybrid.projection_utils.extract_connectivity_subset` —
  slices a global ``Connectivity`` into sparse weight/length sub-matrices for
  a pair of node subsets.
* :func:`~tvb.simulator.hybrid.projection_utils.create_all_to_all_weights` —
  returns a sparse identity matrix representing uniform all-to-all coupling.
* :func:`~tvb.simulator.hybrid.projection_utils.create_inter_projection` —
  builds an :class:`~tvb.simulator.hybrid.InterProjection` from explicit
  weights or from a global ``Connectivity``, accepting both string and integer
  cvar specifications.
* :func:`~tvb.simulator.hybrid.projection_utils.create_intra_projection` —
  builds an :class:`~tvb.simulator.hybrid.IntraProjection` for within-subnet
  coupling.

Also covers ``NetworkSet.add_projection`` and
``NetworkSet.add_projection_from_connectivity`` convenience methods.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from tvb.simulator.models import JansenRit
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.hybrid import (
    Subnetwork,
    InterProjection,
    NetworkSet,
    IntraProjection,
)
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.hybrid.projection_utils import (
    create_inter_projection,
    create_intra_projection,
    create_all_to_all_weights,
    extract_connectivity_subset,
)


def _create_test_connectivity(n_regions=4):
    """Helper function to create a test connectivity object."""
    weights = np.random.rand(n_regions, n_regions) * 0.1
    np.fill_diagonal(weights, 0.0)
    tract_lengths = np.random.rand(n_regions, n_regions) * 50.0
    np.fill_diagonal(tract_lengths, 0.0)
    centres = np.random.rand(n_regions, 3) * 100.0
    region_labels = np.array([f"Region{i}" for i in range(n_regions)], dtype="U128")

    conn = Connectivity(
        weights=weights,
        tract_lengths=tract_lengths,
        centres=centres,
        region_labels=region_labels,
        speed=np.array([3.0]),
    )
    conn.configure()
    return conn


class TestProjectionUtils:
    """
    Tests for the low-level projection factory functions in
    :mod:`tvb.simulator.hybrid.projection_utils`.
    """

    def test_extract_connectivity_subset(self):
        """
        ``extract_connectivity_subset`` returns sparse weight and length matrices
        of the correct shape for the requested source/target index subsets.

        Given a 4-region connectivity, selecting source indices ``[0, 1]`` and
        target indices ``[2, 3]`` should yield ``(2, 2)`` CSR matrices for both
        ``"weights"`` and ``"lengths"``.  The test also confirms that the
        returned dictionary contains exactly these two keys.
        """
        # Create a simple connectivity
        connectivity = _create_test_connectivity(4)

        # Extract subset
        result = extract_connectivity_subset(
            connectivity,
            source_indices=[0, 1],
            target_indices=[2, 3],
            use_weights=True,
            use_lengths=True,
        )

        # Extract subset
        result = extract_connectivity_subset(
            connectivity,
            source_indices=[0, 1],
            target_indices=[2, 3],
            use_weights=True,
            use_lengths=True,
        )

        # Check weights
        assert "weights" in result
        assert isinstance(result["weights"], sp.csr_matrix)
        assert result["weights"].shape == (2, 2)

        # Check lengths
        assert "lengths" in result
        assert isinstance(result["lengths"], sp.csr_matrix)
        assert result["lengths"].shape == (2, 2)

    def test_create_all_to_all_weights(self):
        """
        ``create_all_to_all_weights(n)`` returns a sparse identity matrix of
        shape ``(n, n)``.

        An identity weight matrix represents uniform self-to-self (all-to-all
        equal) coupling; this is the default for intra-subnet projections that
        do not derive weights from a global connectivity object.
        """
        weights = create_all_to_all_weights(3)
        assert isinstance(weights, sp.csr_matrix)
        assert weights.shape == (3, 3)
        expected = np.eye(3)
        np.testing.assert_array_equal(weights.toarray(), expected)

    def test_create_inter_projection_with_names(self):
        """
        ``create_inter_projection`` resolves string cvar names to integer indices.

        Passing ``source_cvar="y0"`` and ``target_cvar="y1"`` for JansenRit
        subnetworks should produce a projection with
        ``source_cvar == [0]`` and ``target_cvar == [1]``.  The weight matrix
        is preserved exactly through the factory.
        """
        # Create subnetworks
        source = Subnetwork(
            name="source", model=JansenRit(), scheme=HeunDeterministic(dt=0.1), nnodes=3
        ).configure()

        target = Subnetwork(
            name="target", model=JansenRit(), scheme=HeunDeterministic(dt=0.1), nnodes=2
        ).configure()

        # Create projection with named cvars
        weights = np.array([[0.1, 0.2], [0.3, 0.4]])
        weights_sparse = sp.csr_matrix(weights)

        proj = create_inter_projection(
            source_subnet=source,
            target_subnet=target,
            source_cvar="y0",  # Named cvar!
            target_cvar="y1",  # Named cvar!
            weights=weights_sparse,
            scale=1.0,
        )

        # Verify cvars were resolved to indices
        np.testing.assert_array_equal(proj.source_cvar, np.array([0]))
        np.testing.assert_array_equal(proj.target_cvar, np.array([1]))

        # Verify weights are correct (allow for floating point precision)
        np.testing.assert_allclose(proj.weights.toarray(), weights)

    def test_create_inter_projection_with_connectivity(self):
        """
        ``create_inter_projection`` can derive weights and lengths from a global
        ``Connectivity`` object given source and target regional indices.

        For a 5-region connectivity with source indices ``[0, 1, 2]`` and target
        indices ``[3, 4]``, the resulting projection has 6 non-zero entries in
        both its weight and length matrices (3 × 2 fully-connected block), and
        ``cv`` / ``dt`` are forwarded correctly.
        """
        # Create global connectivity
        connectivity = _create_test_connectivity(5)

        # Create subnetworks (first 3 nodes are source, last 2 are target)
        source = Subnetwork(
            name="source", model=JansenRit(), scheme=HeunDeterministic(dt=0.1), nnodes=3
        ).configure()

        target = Subnetwork(
            name="target", model=JansenRit(), scheme=HeunDeterministic(dt=0.1), nnodes=2
        ).configure()

        # Create projection using connectivity
        proj = create_inter_projection(
            source_subnet=source,
            target_subnet=target,
            source_cvar="y0",
            target_cvar="y1",
            connectivity=connectivity,
            source_indices=[0, 1, 2],
            target_indices=[3, 4],
            scale=1e-4,
            cv=3.0,
            dt=0.1,
        )

        # Verify projection was created correctly
        assert proj.weights.nnz == 6  # 3x2 connections
        assert proj.lengths.nnz == 6

        # Verify cv and dt were set from parameters
        assert proj.cv == 3.0
        assert proj.dt == 0.1

    def test_create_inter_projection_multiple_cvars(self):
        """
        Multiple source cvars can be mapped to a single target cvar.

        Passing ``source_cvar=["y0", "y1"]`` and ``target_cvar="y0"`` creates
        a projection with ``source_cvar == [0, 1]`` and ``target_cvar == [0]``.
        This pattern is used when several source variables jointly drive one
        target coupling channel.
        """
        source = Subnetwork(
            name="source", model=JansenRit(), scheme=HeunDeterministic(dt=0.1), nnodes=3
        ).configure()

        target = Subnetwork(
            name="target", model=JansenRit(), scheme=HeunDeterministic(dt=0.1), nnodes=2
        ).configure()

        weights = np.array([[0.1, 0.2], [0.3, 0.4]])
        weights_sparse = sp.csr_matrix(weights)

        # Multiple source cvars to single target cvar
        proj = create_inter_projection(
            source_subnet=source,
            target_subnet=target,
            source_cvar=["y0", "y1"],  # Multiple source!
            target_cvar="y0",  # Single target!
            weights=weights_sparse,
            scale=1.0,
        )

        np.testing.assert_array_equal(proj.source_cvar, np.array([0, 1]))
        np.testing.assert_array_equal(proj.target_cvar, np.array([0]))

    def test_create_inter_projection_repeated_cvars(self):
        """
        The same source cvar may be repeated to drive multiple distinct target
        cvars.

        ``source_cvar=["y0", "y0", "y0"]`` with ``target_cvar=[0, 1, 2]``
        (mixed string/int) should resolve without error, yielding
        ``source_cvar == [0, 0, 0]`` and ``target_cvar == [1, 2, 3]``.
        """
        source = Subnetwork(
            name="source", model=JansenRit(), scheme=HeunDeterministic(dt=0.1), nnodes=3
        ).configure()

        target = Subnetwork(
            name="target", model=JansenRit(), scheme=HeunDeterministic(dt=0.1), nnodes=3
        ).configure()

        weights = np.eye(3)
        weights_sparse = sp.csr_matrix(weights)

        # Repeated source cvars to multiple target cvars (M=N with repetition)
        proj = create_inter_projection(
            source_subnet=source,
            target_subnet=target,
            source_cvar=["y0", "y0", "y0"],  # Same source repeated
            target_cvar=[0, 1, 2],  # Different targets
            weights=weights_sparse,
            scale=1.0,
        )

        # Verify cvars were correctly resolved
        np.testing.assert_array_equal(proj.source_cvar, np.array([0, 0, 0]))
        np.testing.assert_array_equal(proj.target_cvar, np.array([0, 1, 2]))

    def test_create_inter_projection_repeated_cvars_with_names(self):
        """Test creating inter-projection with repeated named cvars."""
        source = Subnetwork(
            name="source", model=JansenRit(), scheme=HeunDeterministic(dt=0.1), nnodes=3
        ).configure()

        target = Subnetwork(
            name="target", model=JansenRit(), scheme=HeunDeterministic(dt=0.1), nnodes=3
        ).configure()

        weights = np.eye(3)
        weights_sparse = sp.csr_matrix(weights)

        # Repeated source cvars with string names, mixed target cvars
        proj = create_inter_projection(
            source_subnet=source,
            target_subnet=target,
            source_cvar=["y0", "y0", "y0"],  # Repeated named source
            target_cvar=["y1", "y2", "y3"],  # Named targets
            weights=weights_sparse,
            scale=1.0,
        )

        # Verify cvars were correctly resolved
        np.testing.assert_array_equal(proj.source_cvar, np.array([0, 0, 0]))
        np.testing.assert_array_equal(proj.target_cvar, np.array([1, 2, 3]))

    def test_create_intra_projection(self):
        """
        ``create_intra_projection`` builds an :class:`IntraProjection` for
        within-subnet self-coupling.

        An identity weight matrix is passed for a 3-node subnet; the factory
        should preserve the non-zero count (potentially augmented by an epsilon
        regularisation) and correctly resolve the named cvars ``"y0"`` → 0 and
        ``"y1"`` → 1.
        """
        subnet = Subnetwork(
            name="subnet", model=JansenRit(), scheme=HeunDeterministic(dt=0.1), nnodes=3
        ).configure()

        weights = np.eye(3)
        weights_sparse = sp.csr_matrix(weights)

        proj = create_intra_projection(
            subnet=subnet,
            source_cvar="y0",
            target_cvar="y1",
            weights=weights_sparse,
            scale=1.0,
        )

        assert (
            proj.weights.nnz == 5
        )  # 3 diagonal + 2 epsilon additions (first column, excluding overlap at (0,0))
        np.testing.assert_array_equal(proj.source_cvar, np.array([0]))
        np.testing.assert_array_equal(proj.target_cvar, np.array([1]))


class TestNetworkSetHelpers:
    """
    Tests for the :class:`~tvb.simulator.hybrid.NetworkSet` convenience methods
    :meth:`~tvb.simulator.hybrid.NetworkSet.add_projection` and
    :meth:`~tvb.simulator.hybrid.NetworkSet.add_projection_from_connectivity`.
    """

    def test_add_projection_by_name(self):
        """
        ``NetworkSet.add_projection`` creates and registers a projection
        identified by subnet name, resolving string cvar specifiers.

        After the call the ``NetworkSet.projections`` list should contain
        exactly one entry, the returned projection object, with
        ``source_cvar == [0]`` and ``target_cvar == [1]``.
        """
        # Create two subnetworks
        source = Subnetwork(
            name="cortex", model=JansenRit(), scheme=HeunDeterministic(dt=0.1), nnodes=3
        ).configure()

        target = Subnetwork(
            name="thalamus",
            model=JansenRit(),
            scheme=HeunDeterministic(dt=0.1),
            nnodes=2,
        ).configure()

        nets = NetworkSet(subnets=[source, target])

        # Add projection by name (with named cvars!)
        weights = np.array([[0.1, 0.2], [0.3, 0.4]])
        weights_sparse = sp.csr_matrix(weights)

        proj = nets.add_projection(
            source_name="cortex",
            target_name="thalamus",
            source_cvar="y0",  # Named cvar!
            target_cvar="y1",  # Named cvar!
            weights=weights_sparse,
            scale=1e-4,
        )

        # Verify projection was created and added
        assert len(nets.projections) == 1
        assert nets.projections[0] is proj

        # Verify cvars were resolved
        np.testing.assert_array_equal(proj.source_cvar, np.array([0]))
        np.testing.assert_array_equal(proj.target_cvar, np.array([1]))

    def test_add_projection_from_connectivity(self):
        """
        ``NetworkSet.add_projection_from_connectivity`` derives weights and
        lengths from a global ``Connectivity`` and registers the resulting
        :class:`InterProjection` on the network set.

        The ``cv`` and ``dt`` timing parameters supplied by the caller must be
        forwarded intact to the created projection.
        """
        # Create global connectivity
        connectivity = _create_test_connectivity(5)

        # Create subnetworks (first 3 are cortex, last 2 are thalamus)
        cortex = Subnetwork(
            name="cortex", model=JansenRit(), scheme=HeunDeterministic(dt=0.1), nnodes=3
        ).configure()

        thalamus = Subnetwork(
            name="thalamus",
            model=JansenRit(),
            scheme=HeunDeterministic(dt=0.1),
            nnodes=2,
        ).configure()

        nets = NetworkSet(subnets=[cortex, thalamus])

        # Add projection from connectivity (with named cvars!)
        proj = nets.add_projection_from_connectivity(
            source_name="cortex",
            target_name="thalamus",
            connectivity=connectivity,
            source_indices=[0, 1, 2],
            target_indices=[3, 4],
            source_cvar="y0",  # Named cvar!
            target_cvar="y1",  # Named cvar!
            scale=1e-4,
            cv=3.0,
            dt=0.1,
        )

        # Verify projection was created and added
        assert len(nets.projections) == 1
        assert nets.projections[0] is proj

        # Verify cv and dt were set from parameters
        assert proj.cv == 3.0
        assert proj.dt == 0.1
