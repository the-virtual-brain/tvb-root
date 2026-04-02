"""
Tests for projection_utils module.
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
    """Tests for projection utility functions."""

    def test_extract_connectivity_subset(self):
        """Test extracting connectivity subset."""
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
        """Test creating all-to-all weights matrix."""
        weights = create_all_to_all_weights(3)
        assert isinstance(weights, sp.csr_matrix)
        assert weights.shape == (3, 3)
        expected = np.eye(3)
        np.testing.assert_array_equal(weights.toarray(), expected)

    def test_create_inter_projection_with_names(self):
        """Test creating inter-projection with named cvars."""
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
        """Test creating inter-projection from connectivity."""
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
        """Test creating inter-projection with multiple cvars."""
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
        """Test creating inter-projection with repeated source cvars."""
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
        """Test creating intra-projection."""
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
    """Tests for NetworkSet convenience methods."""

    def test_add_projection_by_name(self):
        """Test adding projection by subnet name."""
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
        """Test adding projection from global connectivity."""
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
