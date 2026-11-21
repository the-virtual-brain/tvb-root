"""The public C++ simulator must infer a chunk compatible with all monitors."""

import numpy as np
import pytest

from tvb.simulator.hybrid import NetworkSet, Simulator, Subnetwork
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.models.linear import Linear
from tvb.simulator.monitors import Raw, RawVoi, SubSample, TemporalAverage


@pytest.mark.parametrize("monitor, stride", [(Raw, 1), (RawVoi, 1), (SubSample, 2)])
def test_cpp_public_automatic_chunk_preserves_monitor_cadence(monitor, stride):
    pytest.importorskip("tvb.simulator.backend.cpp_hybrid")
    dt, nstep = 0.125, 16
    initial = np.array([[[1.0], [-0.5]]])
    subnet = Subnetwork(name="linear", model=Linear(gamma=np.array([-0.5])),
                        scheme=HeunDeterministic(dt=dt), nnodes=2,
                        node_indices=np.arange(2))
    fast = SubSample(period=stride * dt) if monitor is SubSample else monitor()
    sim = Simulator(nets=NetworkSet(subnets=[subnet]), backend="cpp",
                    simulation_length=nstep * dt,
                    monitors=[fast, TemporalAverage(period=4 * dt)])
    sim.configure()
    (t_fast, x_fast), (t_avg, x_avg) = sim.run(initial_conditions=[initial])

    steps = np.arange(1, nstep + 1)
    z = -0.5 * dt
    trajectory = (1 + z + z*z/2) ** steps[:, None, None, None] * initial
    np.testing.assert_array_equal(t_fast, dt * steps[stride-1::stride])
    np.testing.assert_array_equal(t_avg, 4 * dt * (np.arange(nstep // 4) + 0.5))
    np.testing.assert_allclose(x_fast, trajectory[stride-1::stride], rtol=1e-6)
    np.testing.assert_allclose(x_avg, trajectory.reshape(-1, 4, 1, 2, 1).mean(1), rtol=1e-6)
