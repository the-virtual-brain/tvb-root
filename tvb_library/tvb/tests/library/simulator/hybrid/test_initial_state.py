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
Tests for hybrid simulator initial-state generation.

Validates that:

* :meth:`~tvb.simulator.hybrid.Subnetwork.random_states` draws values from
  each model's ``state_variable_range``.
* :meth:`~tvb.simulator.hybrid.NetworkSet.random_states` propagates that
  behaviour across all subnetworks.
* Projection history buffers pre-filled via
  :meth:`~tvb.simulator.hybrid.NetworkSet.init_projection_buffers` contain
  only values within the state variable ranges.
* :meth:`~tvb.simulator.hybrid.Simulator.run` accepts a ``random_state``
  argument for reproducibility, and two calls with the same integer seed
  produce identical results.
* Explicit ``initial_conditions`` passed to ``run`` override the random
  default (backward-compatibility).

Models under test: Generic2dOscillator, JansenRit, ReducedSetFitzHughNagumo.
"""

import numpy as np
import scipy.sparse as sp
import pytest

from tvb.simulator.models import JansenRit, ReducedSetFitzHughNagumo, Generic2dOscillator
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.monitors import TemporalAverage
from tvb.simulator.hybrid import (
    NetworkSet,
    Simulator,
    Subnetwork,
    IntraProjection,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODELS = [Generic2dOscillator, JansenRit, ReducedSetFitzHughNagumo]
SINGLE_MODE_MODELS = [Generic2dOscillator, JansenRit]  # ReducedSetFitzHughNagumo uses 3 modes
NNODES = 8
DT = 0.1


def _make_subnetwork(model_cls) -> Subnetwork:
    """Return a configured Subnetwork for *model_cls*."""
    model = model_cls()
    scheme = HeunDeterministic(dt=DT)
    sn = Subnetwork(name="subnet", model=model, scheme=scheme, nnodes=NNODES)
    sn.configure()
    return sn


def _make_network(model_cls) -> NetworkSet:
    """Return a NetworkSet with a single Subnetwork plus an IntraProjection."""
    sn = _make_subnetwork(model_cls)
    # Minimal identity-weight, minimal-delay intra-projection so history
    # buffers are actually allocated and can be inspected.
    weights = sp.eye(NNODES, format="csr")
    lengths = sp.eye(NNODES, format="csr")
    proj = IntraProjection(
        source_cvar=sn.model.cvar[:1],
        target_cvar=sn.model.cvar[:1],
        weights=weights,
        lengths=lengths,
        cv=1.0,
        dt=DT,
    )
    sn2 = Subnetwork(
        name="subnet",
        model=model_cls(),
        scheme=HeunDeterministic(dt=DT),
        nnodes=NNODES,
        projections=[proj],
    )
    sn2.configure()
    nets = NetworkSet(subnets=[sn2], projections=[])
    return nets, sn2


def _state_variable_bounds(model):
    """Return list of (lo, hi) per state variable."""
    svr = model.state_variable_range
    return [svr[sv] for sv in model.state_variables]


# ---------------------------------------------------------------------------
# Tests: Subnetwork.random_states
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_cls", MODELS)
def test_subnetwork_random_states_shape(model_cls):
    """random_states() returns shape ``(nvar, nnodes, modes)``."""
    sn = _make_subnetwork(model_cls)
    x = sn.random_states()
    assert x.shape == (sn.model.nvar, NNODES, sn.model.number_of_modes)


@pytest.mark.parametrize("model_cls", MODELS)
def test_subnetwork_random_states_within_range(model_cls):
    """Every state variable value lies within the model's state_variable_range."""
    sn = _make_subnetwork(model_cls)
    x = sn.random_states(rng=np.random.RandomState(0))
    bounds = _state_variable_bounds(sn.model)
    for i, (lo, hi) in enumerate(bounds):
        assert np.all(x[i] >= lo), (
            f"{model_cls.__name__} sv[{i}]: min {x[i].min():.4f} < lo {lo}"
        )
        assert np.all(x[i] <= hi), (
            f"{model_cls.__name__} sv[{i}]: max {x[i].max():.4f} > hi {hi}"
        )


@pytest.mark.parametrize("model_cls", MODELS)
def test_subnetwork_random_states_not_all_zero(model_cls):
    """random_states() must not return all-zeros (unlike zero_states)."""
    sn = _make_subnetwork(model_cls)
    x = sn.random_states(rng=np.random.RandomState(1))
    # At least one element should differ from zero (extremely unlikely to be
    # all-zero given random sampling over [lo, hi]).
    assert not np.all(x == 0.0)


@pytest.mark.parametrize("model_cls", MODELS)
def test_subnetwork_random_states_reproducible(model_cls):
    """Same RandomState seed → identical arrays."""
    sn = _make_subnetwork(model_cls)
    x1 = sn.random_states(rng=np.random.RandomState(42))
    x2 = sn.random_states(rng=np.random.RandomState(42))
    np.testing.assert_array_equal(x1, x2)


@pytest.mark.parametrize("model_cls", MODELS)
def test_subnetwork_random_states_different_seeds(model_cls):
    """Different seeds → different arrays (with overwhelming probability)."""
    sn = _make_subnetwork(model_cls)
    x1 = sn.random_states(rng=np.random.RandomState(0))
    x2 = sn.random_states(rng=np.random.RandomState(1))
    assert not np.array_equal(x1, x2)


# ---------------------------------------------------------------------------
# Tests: NetworkSet.random_states
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_cls", MODELS)
def test_network_random_states_within_range(model_cls):
    """NetworkSet.random_states() propagates range bounds to each subnetwork."""
    sn = _make_subnetwork(model_cls)
    nets = NetworkSet(subnets=[sn], projections=[])
    xs = nets.random_states(rng=np.random.RandomState(7))
    x = xs.subnet
    bounds = _state_variable_bounds(sn.model)
    for i, (lo, hi) in enumerate(bounds):
        assert np.all(x[i] >= lo)
        assert np.all(x[i] <= hi)


# ---------------------------------------------------------------------------
# Tests: history buffer content after init_projection_buffers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_cls", MODELS)
def test_history_buffers_filled_from_initial_conditions(model_cls):
    """After init_projection_buffers with random ICs, all buffer slots match the IC values.

    The buffer stores the full source state at every horizon slot.  When ICs
    are provided, every slot should contain exactly those IC values (not zeros).
    """
    nets, sn = _make_network(model_cls)
    rng = np.random.RandomState(99)
    xs = nets.random_states(rng)
    nets.init_projection_buffers(xs)

    proj = sn.projections[0]
    # Buffer shape: (nvar, nnodes, modes, horizon)
    buf = proj._history_buffer
    assert np.all(np.isfinite(buf)), "Non-finite values in history buffer"
    # Every time slot should contain the IC state (all slots filled identically)
    ic = xs.subnet  # shape (nvar, nnodes, modes)
    for slot in range(proj._horizon):
        # Buffer is stored as float32; compare with matching tolerance
        np.testing.assert_allclose(
            buf[..., slot], ic.astype(np.float32),
            rtol=1e-5, atol=1e-5,
            err_msg=f"Buffer slot {slot} does not match IC for {model_cls.__name__}",
        )
    # Also verify the buffer is not all zeros (ICs come from random_states, not zeros)
    assert not np.all(buf == 0.0), "Buffer contains only zeros — IC was not applied"


# ---------------------------------------------------------------------------
# Tests: Simulator.run random_state API
# ---------------------------------------------------------------------------


def _make_simulator(model_cls):
    """Return a minimally-configured Simulator with a TemporalAverage monitor (2 ms).

    Only single-mode models are supported here; multi-mode models (e.g.
    ReducedSetFitzHughNagumo with 3 modes) conflict with the Simulator-level
    monitor stock allocation which hardcodes ``modes=1``.
    """
    sn = _make_subnetwork(model_cls)
    nets = NetworkSet(subnets=[sn], projections=[])
    tavg = TemporalAverage(period=DT)
    sim = Simulator(nets=nets, simulation_length=2.0, monitors=[tavg])
    sim.configure()
    return sim


@pytest.mark.parametrize("model_cls", SINGLE_MODE_MODELS)
def test_simulator_run_random_state_reproducibility(model_cls):
    """run(random_state=42) twice → identical outputs."""
    ((_, y1),) = _make_simulator(model_cls).run(random_state=42)
    ((_, y2),) = _make_simulator(model_cls).run(random_state=42)
    np.testing.assert_array_equal(y1, y2)


@pytest.mark.parametrize("model_cls", SINGLE_MODE_MODELS)
def test_simulator_run_different_seeds_differ(model_cls):
    """run(random_state=0) vs run(random_state=1) → different outputs."""
    ((_, y1),) = _make_simulator(model_cls).run(random_state=0)
    ((_, y2),) = _make_simulator(model_cls).run(random_state=1)
    assert not np.array_equal(y1, y2)


@pytest.mark.parametrize("model_cls", SINGLE_MODE_MODELS)
def test_simulator_run_explicit_initial_conditions_override(model_cls):
    """Explicit zero initial_conditions → output is identical to another zero-IC run.

    Explicit ICs must override the random default, so two runs with the same
    zero IC array produce identical output regardless of random_state.
    """
    sn = _make_subnetwork(model_cls)
    zero_ic = [np.zeros((sn.model.nvar, NNODES, sn.model.number_of_modes))]

    def _run_with_zeros():
        sn_ = _make_subnetwork(model_cls)
        nets_ = NetworkSet(subnets=[sn_], projections=[])
        tavg = TemporalAverage(period=DT)
        sim = Simulator(nets=nets_, simulation_length=1.0, monitors=[tavg])
        sim.configure()
        ((_, y),) = sim.run(initial_conditions=zero_ic)
        return y

    np.testing.assert_array_equal(_run_with_zeros(), _run_with_zeros())


@pytest.mark.parametrize("model_cls", SINGLE_MODE_MODELS)
def test_simulator_default_run_starts_from_ranges(model_cls):
    """Default run() produces finite, non-zero output, confirming random IC from ranges.

    The first recorded sample should differ from zeros, confirming that
    the default initialisation draws from state_variable_range, not zeros.
    """
    ((_, y),) = _make_simulator(model_cls).run(random_state=0)

    assert np.all(np.isfinite(y)), "Non-finite values in default-run output"
    assert not np.all(y == 0.0), "Default run produced all-zero output (expected random ICs)"
