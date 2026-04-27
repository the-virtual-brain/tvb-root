# -*- coding: utf-8 -*-
"""
Tests for UX improvement features (TDD — written before implementation).

Three feature areas tested here:
1.  constant_stim / pulse_stim / sinusoid_stim convenience factories
2.  StimuliRegion.from_weights() class method
3.  Simulator(backend="numba") parameter

Each test class starts with a comment identifying the proposal it covers
and will FAIL until the corresponding production code is implemented.
"""

import unittest

import numpy as np

from tvb.datatypes import equations as eqs
from tvb.datatypes.patterns import StimuliRegion
from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.hybrid import NetworkSet, Subnetwork, Stim
from tvb.simulator.hybrid.simulator import Simulator
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.monitors import TemporalAverage

DT = 0.1


def _make_subnet(name="ctx", n=4):
    model = MontbrioPazoRoxin()
    model.configure()
    scheme = HeunDeterministic(dt=DT)
    sn = Subnetwork(name=name, model=model, scheme=scheme, nnodes=n)
    sn.configure()
    return sn


# ======================================================================
# Proposal 1: constant_stim / pulse_stim / sinusoid_stim factories
# ======================================================================


class TestConstantStimFactory(unittest.TestCase):
    """Tests for stimulus_utils.constant_stim()."""

    def _import(self):
        from tvb.simulator.hybrid.stimulus_utils import constant_stim
        return constant_stim

    def test_returns_stim_object(self):
        constant_stim = self._import()
        sn = _make_subnet(n=5)
        stim = constant_stim(sn, amplitude=0.05, simulation_length=10.0)
        self.assertIsInstance(stim, Stim)

    def test_targets_correct_subnet(self):
        constant_stim = self._import()
        sn = _make_subnet(n=5)
        stim = constant_stim(sn, amplitude=0.05, simulation_length=10.0)
        self.assertIs(stim.target, sn)

    def test_default_targets_cvar_zero(self):
        constant_stim = self._import()
        sn = _make_subnet(n=5)
        stim = constant_stim(sn, amplitude=0.05, simulation_length=10.0)
        np.testing.assert_array_equal(stim.target_cvar, [0])

    def test_custom_target_cvar(self):
        constant_stim = self._import()
        sn = _make_subnet(n=5)
        stim = constant_stim(sn, amplitude=0.05, target_cvar=1,
                             simulation_length=10.0)
        np.testing.assert_array_equal(stim.target_cvar, [1])

    def test_default_targets_single_node(self):
        """By default, only node 0 is stimulated."""
        constant_stim = self._import()
        sn = _make_subnet(n=5)
        stim = constant_stim(sn, amplitude=0.05, simulation_length=10.0)
        # The stimulus weight vector should be [1, 0, 0, 0, 0]
        w = stim.stimulus.weight
        self.assertEqual(w[0], 1.0)
        np.testing.assert_array_equal(w[1:], 0.0)

    def test_target_node_none_stimulates_all(self):
        """target_node=None → all nodes get uniform weight."""
        constant_stim = self._import()
        sn = _make_subnet(n=5)
        stim = constant_stim(sn, amplitude=0.05, target_node=None,
                             simulation_length=10.0)
        w = stim.stimulus.weight
        np.testing.assert_array_equal(w, np.ones(5))

    def test_is_configured_when_simulation_length_given(self):
        """If simulation_length is provided, stim should be pre-configured."""
        constant_stim = self._import()
        sn = _make_subnet(n=4)
        stim = constant_stim(sn, amplitude=0.1, simulation_length=10.0)
        # A configured stim has a time array
        self.assertTrue(hasattr(stim, 'time'))
        self.assertIsNotNone(stim.time)

    def test_constant_amplitude_value(self):
        """The stimulus should produce the specified constant amplitude."""
        constant_stim = self._import()
        sn = _make_subnet(n=4)
        amp = 0.123
        stim = constant_stim(sn, amplitude=amp, simulation_length=10.0)
        # Evaluate the temporal equation at a mid-point
        t = np.linspace(0, 10.0, 100)
        temporal = stim.stimulus.temporal
        values = temporal(t)
        # Linear(a=0, b=amp) → constant = amp
        np.testing.assert_allclose(values, amp, rtol=1e-10)


class TestPulseStimFactory(unittest.TestCase):
    """Tests for stimulus_utils.pulse_stim()."""

    def _import(self):
        from tvb.simulator.hybrid.stimulus_utils import pulse_stim
        return pulse_stim

    def test_returns_stim_object(self):
        pulse_stim = self._import()
        sn = _make_subnet(n=4)
        stim = pulse_stim(sn, amplitude=0.1, onset=1.0, period=5.0,
                          pulse_width=0.5, simulation_length=20.0)
        self.assertIsInstance(stim, Stim)

    def test_uses_pulsetrain_equation(self):
        pulse_stim = self._import()
        sn = _make_subnet(n=4)
        stim = pulse_stim(sn, amplitude=0.1, onset=1.0, period=5.0,
                          pulse_width=0.5, simulation_length=20.0)
        self.assertIsInstance(stim.stimulus.temporal, eqs.PulseTrain)

    def test_pulsetrain_parameters_set(self):
        pulse_stim = self._import()
        sn = _make_subnet(n=4)
        stim = pulse_stim(sn, amplitude=0.1, onset=1.0, period=5.0,
                          pulse_width=0.5, simulation_length=20.0)
        params = stim.stimulus.temporal.parameters
        self.assertAlmostEqual(params["T"], 5.0)
        self.assertAlmostEqual(params["tau"], 0.5)
        self.assertAlmostEqual(params["onset"], 1.0)

    def test_custom_target_cvar(self):
        pulse_stim = self._import()
        sn = _make_subnet(n=4)
        stim = pulse_stim(sn, amplitude=0.1, onset=1.0, period=5.0,
                          pulse_width=0.5, target_cvar=1,
                          simulation_length=20.0)
        np.testing.assert_array_equal(stim.target_cvar, [1])


class TestSinusoidStimFactory(unittest.TestCase):
    """Tests for stimulus_utils.sinusoid_stim()."""

    def _import(self):
        from tvb.simulator.hybrid.stimulus_utils import sinusoid_stim
        return sinusoid_stim

    def test_returns_stim_object(self):
        sinusoid_stim = self._import()
        sn = _make_subnet(n=4)
        stim = sinusoid_stim(sn, amplitude=0.05, frequency=0.1,
                             simulation_length=10.0)
        self.assertIsInstance(stim, Stim)

    def test_uses_sinusoid_equation(self):
        sinusoid_stim = self._import()
        sn = _make_subnet(n=4)
        stim = sinusoid_stim(sn, amplitude=0.05, frequency=0.1,
                             simulation_length=10.0)
        self.assertIsInstance(stim.stimulus.temporal, eqs.Sinusoid)

    def test_sinusoid_parameters_set(self):
        sinusoid_stim = self._import()
        sn = _make_subnet(n=4)
        stim = sinusoid_stim(sn, amplitude=0.05, frequency=0.1,
                             simulation_length=10.0)
        params = stim.stimulus.temporal.parameters
        self.assertAlmostEqual(params["amp"], 0.05)
        self.assertAlmostEqual(params["frequency"], 0.1)

    def test_custom_target_node(self):
        sinusoid_stim = self._import()
        sn = _make_subnet(n=4)
        stim = sinusoid_stim(sn, amplitude=0.05, frequency=0.1,
                             target_node=2, simulation_length=10.0)
        w = stim.stimulus.weight
        self.assertEqual(w[2], 1.0)
        np.testing.assert_array_equal(np.delete(w, 2), 0.0)


# ======================================================================
# Proposal 2: Simulator(backend="numba")
# ======================================================================


class TestSimulatorBackend(unittest.TestCase):
    """Tests for Simulator backend parameter."""

    def test_default_backend_is_python(self):
        sn = _make_subnet(n=4)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        sim = Simulator(nets=ns, simulation_length=1.0)
        self.assertEqual(sim.backend, "python")

    def test_backend_numba_accepted(self):
        sn = _make_subnet(n=4)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        sim = Simulator(nets=ns, simulation_length=1.0, backend="numba")
        self.assertEqual(sim.backend, "numba")

    def test_backend_invalid_raises(self):
        sn = _make_subnet(n=4)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        with self.assertRaises((ValueError, AssertionError)):
            Simulator(nets=ns, simulation_length=1.0, backend="julia")

    def test_python_backend_runs_python_loop(self):
        """backend='python' should use NetworkSet.step internally."""
        sn = _make_subnet(n=4)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        sim = Simulator(nets=ns, simulation_length=1.0, backend="python")
        sim.configure()
        result = sim.run(random_state=42)
        # With no monitors, returns empty list
        self.assertEqual(result, [])

    def test_numba_backend_produces_output(self):
        """backend='numba' should delegate to NbHybridBackend."""
        sn = _make_subnet(n=4)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        tavg = TemporalAverage(period=1.0)
        sim = Simulator(nets=ns, simulation_length=10.0, monitors=[tavg],
                        backend="numba")
        sim.configure()
        result = sim.run(random_state=42)
        # Should return list of (times, data) tuples
        self.assertEqual(len(result), 1)
        t, y = result[0]
        self.assertEqual(len(t), 10)  # 10 steps / period=1.0
        self.assertEqual(y.shape[0], 10)

    def test_numba_and_python_agree(self):
        """Python and Numba backends produce numerically similar results."""
        sn = _make_subnet(n=4)
        ns_py = NetworkSet(subnets=[_make_subnet(n=4)], projections=[], stimuli=[])
        ns_nb = NetworkSet(subnets=[_make_subnet(n=4)], projections=[], stimuli=[])

        tavg_py = TemporalAverage(period=1.0)
        tavg_nb = TemporalAverage(period=1.0)

        sim_py = Simulator(nets=ns_py, simulation_length=5.0,
                           monitors=[tavg_py], backend="python")
        sim_nb = Simulator(nets=ns_nb, simulation_length=5.0,
                           monitors=[tavg_nb], backend="numba")
        sim_py.configure()
        sim_nb.configure()

        (_, y_py), = sim_py.run(random_state=42)
        (_, y_nb), = sim_nb.run(random_state=42)

        np.testing.assert_allclose(y_nb, y_py, rtol=1e-3, atol=1e-4)

    def test_numba_with_stimulus(self):
        """backend='numba' should work with stimuli attached."""
        from tvb.datatypes.connectivity import Connectivity
        sn = _make_subnet(n=4)
        # Build a minimal stimulus
        conn = Connectivity(
            centres=np.zeros((4, 3)), weights=np.zeros((4, 4)),
            tract_lengths=np.zeros((4, 4)),
            region_labels=np.array([str(i) for i in range(4)]),
            speed=np.array([1.0]),
        )
        conn.configure()
        temporal = eqs.Linear()
        temporal.parameters["a"] = 0.0
        temporal.parameters["b"] = 0.05
        weight = np.zeros(4); weight[0] = 1.0
        stim_pattern = StimuliRegion(
            temporal=temporal, connectivity=conn, weight=weight,
        )
        stim = Stim(target=sn, stimulus=stim_pattern,
                    target_cvar=np.array([0], dtype=np.int_))
        stim.configure(simulation_length=10.0)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[stim])
        tavg = TemporalAverage(period=1.0)
        sim = Simulator(nets=ns, simulation_length=10.0,
                        monitors=[tavg], backend="numba")
        sim.configure()
        result = sim.run(random_state=42)
        self.assertEqual(len(result), 1)


# ======================================================================
# Proposal 3: StimuliRegion.from_weights()
# ======================================================================


class TestStimuliRegionFromWeights(unittest.TestCase):
    """Tests for StimuliRegion.from_weights() class method."""

    def test_returns_stimuli_region(self):
        temporal = eqs.Linear()
        temporal.parameters["a"] = 0.0
        temporal.parameters["b"] = 1.0
        sr = StimuliRegion.from_weights(
            weight=np.array([1.0, 0.0, 0.0]), temporal=temporal,
        )
        self.assertIsInstance(sr, StimuliRegion)

    def test_weight_preserved(self):
        temporal = eqs.Linear()
        temporal.parameters["a"] = 0.0
        temporal.parameters["b"] = 1.0
        w = np.array([1.0, 0.5, 0.0, 0.3])
        sr = StimuliRegion.from_weights(weight=w, temporal=temporal)
        np.testing.assert_array_equal(sr.weight, w)

    def test_temporal_preserved(self):
        temporal = eqs.Sinusoid()
        temporal.parameters["amp"] = 0.1
        temporal.parameters["frequency"] = 0.5
        sr = StimuliRegion.from_weights(
            weight=np.array([1.0, 0.0]), temporal=temporal,
        )
        self.assertIs(sr.temporal, temporal)

    def test_connectivity_created_automatically(self):
        temporal = eqs.Linear()
        temporal.parameters["a"] = 0.0
        temporal.parameters["b"] = 1.0
        w = np.array([1.0, 0.0, 0.0, 0.0])
        sr = StimuliRegion.from_weights(weight=w, temporal=temporal)
        self.assertIsNotNone(sr.connectivity)
        self.assertEqual(sr.connectivity.weights.shape, (4, 4))

    def test_connectivity_is_zero_weight(self):
        temporal = eqs.Linear()
        temporal.parameters["a"] = 0.0
        temporal.parameters["b"] = 1.0
        w = np.array([1.0, 0.0])
        sr = StimuliRegion.from_weights(weight=w, temporal=temporal)
        np.testing.assert_array_equal(sr.connectivity.weights, np.zeros((2, 2)))

    def test_single_node(self):
        temporal = eqs.Linear()
        temporal.parameters["a"] = 0.0
        temporal.parameters["b"] = 1.0
        sr = StimuliRegion.from_weights(
            weight=np.array([1.0]), temporal=temporal,
        )
        self.assertEqual(sr.connectivity.weights.shape, (1, 1))

    def test_works_in_full_stimulus_pipeline(self):
        """from_weights output should be usable as a Stim stimulus."""
        sn = _make_subnet(n=4)
        temporal = eqs.Linear()
        temporal.parameters["a"] = 0.0
        temporal.parameters["b"] = 0.05
        w = np.zeros(4); w[0] = 1.0
        sr = StimuliRegion.from_weights(weight=w, temporal=temporal)
        stim = Stim(target=sn, stimulus=sr,
                    target_cvar=np.array([0], dtype=np.int_),
                    projection_scale=1.0)
        stim.configure(simulation_length=10.0)
        # Should not raise
        coupling = stim.get_coupling(step=5)
        self.assertEqual(coupling.shape[1], 4)


if __name__ == "__main__":
    unittest.main()
