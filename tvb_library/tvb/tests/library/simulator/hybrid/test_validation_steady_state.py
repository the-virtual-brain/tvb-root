"""
Validate steady state convergence for hybrid models.
"""

import numpy as np
from tvb.tests.library.simulator.hybrid.test_validation_base import ValidationTestBase
from tvb.simulator.models import Generic2dOscillator, JansenRit, ReducedWongWang
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.hybrid import Subnetwork, NetworkSet, Simulator


class TestValidationSteadyState(ValidationTestBase):
    """Validate steady state convergence for hybrid models."""

    def _test_model_steady_state(self, model_class, fp, dt=0.01, n_step=2000):
        """Test that model converges to given fixed point.

        Parameters
        ----------
        model_class : Model
            Model class to test
        fp : ndarray
            Expected fixed point
        dt : float, default=0.01
            Integration time step
        n_step : int, default=2000
            Number of simulation steps
        """
        scheme = HeunDeterministic(dt=dt)
        subnet = Subnetwork(
            name="test_model", model=model_class(), scheme=scheme, nnodes=1
        ).configure()

        model = subnet.model
        svr = model.state_variable_range
        sv_names = list(svr.keys())

        init = np.zeros((model._nvar, 1, model.number_of_modes))
        for i, sv_name in enumerate(sv_names):
            lo, hi = svr[sv_name]
            init[i, :, :] = (hi + lo) / 2.0

        nets = NetworkSet(subnets=[subnet], projections=[])
        sim = Simulator(nets=nets, simulation_length=n_step * dt)
        sim.initial_conditions = init
        sim.configure()
        ((t, y),) = sim.run()

        rfp = np.sqrt(np.sum(np.sum((y[-1] - y) ** 2, axis=1), axis=1))
        dr = rfp[-20:] / (rfp[-21:-1] + 1e-9)
        self.assertTrue((dr < 1.0).all(), "Model did not converge to steady state")

        np.testing.assert_allclose(y[-1, :, 0, 0], fp, rtol=1e-6, atol=1e-3)

    def test_generic_2d_oscillator_steady_state(self):
        """Test Generic2dOscillator converges to fixed point."""
        model = Generic2dOscillator(a=np.array([-1.0]), b=np.array([0.0]))
        fp = np.array([0.0, 0.0])
        self._test_model_steady_state(model, fp)

    def test_generic_2d_oscillator_excitable(self):
        """Test Generic2dOscillator excitable regime steady state."""
        model = Generic2dOscillator(
            a=np.array([1.05]),
            b=np.array([-1.00]),
            c=np.array([0.0]),
            d=np.array([0.1]),
            I=np.array([0.0]),
            alpha=np.array([1.0]),
            beta=np.array([0.2]),
            gamma=np.array([-1.0]),
            e=np.array([0.0]),
            g=np.array([1.0]),
            f=np.array([1.0 / 3.0]),
            tau=np.array([1.25]),
        )
        fp = np.array([0.0, 0.0])
        self._test_model_steady_state(model, fp)

    def test_jansen_rit_steady_state(self):
        """Test JansenRit converges to steady state."""
        model = JansenRit(
            A=np.array([3.25]),
            B=np.array([22.0]),
            a=np.array([0.1]),
            b=np.array([0.05]),
            v0=np.array([6.0]),
            e0=np.array([2.5]),
            r=np.array([0.56]),
        )
        fp = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self._test_model_steady_state(model, fp, dt=0.05, n_step=3000)

    def test_reduced_wong_wang_steady_state(self):
        """Test ReducedWongWang converges to steady state."""
        model = ReducedWongWang(
            a=np.array([0.270]),
            b=np.array([0.108]),
            d=np.array([154.0]),
            gamma=np.array([0.641]),
            tau_s=np.array([100.0]),
            w=np.array([0.9]),
            J_N=np.array([0.2609]),
            I_o=np.array([0.33]),
        )
        fp = np.array([0.0])
        self._test_model_steady_state(model, fp, dt=0.1, n_step=5000)

    def test_stability_small_perturbation(self):
        """Test that small perturbations return to steady state."""
        scheme = HeunDeterministic(dt=0.01)
        model = Generic2dOscillator(a=np.array([-1.0]), b=np.array([0.0]))
        subnet = Subnetwork(
            name="stable_model", model=model, scheme=scheme, nnodes=2
        ).configure()

        svr = model.state_variable_range
        init = np.zeros((model._nvar, 2, model.number_of_modes))
        init[0, 0, :] = 0.01
        init[1, 0, :] = 0.01
        init[0, 1, :] = -0.01
        init[1, 1, :] = -0.01

        nets = NetworkSet(subnets=[subnet], projections=[])
        sim = Simulator(nets=nets, simulation_length=500.0)
        sim.initial_conditions = init
        sim.configure()
        ((t, y),) = sim.run()

        final_states = y[-1, :, :, 0]
        for node_idx in range(2):
            np.testing.assert_allclose(
                final_states[:, node_idx],
                [0.0, 0.0],
                rtol=1e-4,
                atol=1e-4,
                err_msg=f"Node {node_idx} did not return to steady state",
            )
