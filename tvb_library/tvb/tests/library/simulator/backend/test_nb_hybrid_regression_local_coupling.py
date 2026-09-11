"""Regression coverage for model ``local_coupling`` in the hybrid backend."""

import numpy as np
import pytest

from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.integrators import EulerDeterministic
from tvb.simulator.models.linear import Linear


DT = 0.01
NSTEPS = 5
INITIAL_STATE = np.array([[[0.25], [0.75]]], dtype=np.float64)


class _EulerWithModelLocalCoupling(EulerDeterministic):
    """Expose the model argument that the hybrid Subnetwork API omits."""

    def __init__(self, model_local_coupling):
        super().__init__(dt=DT)
        self.model_local_coupling = float(model_local_coupling)

    def scheme(self, state, dfun, coupling, local_coupling, stimulus):
        return super().scheme(
            state,
            dfun,
            coupling,
            self.model_local_coupling,
            stimulus,
        )


def _network(local_coupling):
    model = Linear(gamma=np.array([-1.0]))
    subnet = Subnetwork(
        name="linear",
        model=model,
        scheme=_EulerWithModelLocalCoupling(local_coupling),
        nnodes=2,
    )
    subnet.configure()
    network = NetworkSet(subnets=[subnet], projections=[])
    network.configure()
    return network


def _run_python(network):
    state = network.zero_states(initial_states=[INITIAL_STATE.copy()])
    network.init_projection_buffers(state)
    trajectory = []
    for step in range(1, NSTEPS + 1):
        state = network.step(step, state)
        trajectory.append(state.linear.copy())
    return np.stack(trajectory)


def _run_numba(network):
    return NbHybridBackend().run_network(
        network,
        nstep=NSTEPS,
        chunk_size=1,
        initial_states=[INITIAL_STATE.copy()],
    )[0][1]


def test_zero_model_local_coupling_python_numba_control():
    """The supported zero case remains numerically identical in both paths."""
    python = _run_python(_network(0.0))
    numba = _run_numba(_network(0.0))

    np.testing.assert_allclose(numba, python, rtol=1e-6, atol=1e-7)


def test_nonzero_model_local_coupling_is_not_silently_dropped():
    """Reject nonzero model local coupling until Numba can represent it.

    The Python path can pass this standard ``dfun`` argument through an
    integrator, but the generated Numba expressions currently have no slot for
    it.  An explicit compatibility error is therefore the required contract;
    compiling successfully with zero-coupling dynamics would silently change
    the model.
    """
    zero = _run_python(_network(0.0))
    nonzero_network = _network(0.5)
    nonzero = _run_python(nonzero_network)

    assert not np.allclose(nonzero, zero, rtol=1e-6, atol=1e-7), (
        "Linear.dfun must make this fixture sensitive to local_coupling"
    )
    with pytest.raises(NotImplementedError, match=r"(?i)local[_ ]coupling"):
        _run_numba(nonzero_network)
