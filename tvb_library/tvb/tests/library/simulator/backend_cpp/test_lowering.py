# -*- coding: utf-8 -*-

import os
import tempfile
import unittest

import numpy as np

os.environ.setdefault("TVB_USER_HOME", os.path.join(tempfile.gettempdir(), "tvb-user"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

from tvb.simulator.backend_cpp.lowering import SpecLoweringResult, lower_network_set
from tvb.simulator.backend_cpp.spec import IntegratorSpec, SimulationSpec, SubnetworkSpec
from tvb.simulator.hybrid import NetworkSet, Subnetwork
from tvb.simulator.integrators import (
    EulerDeterministic,
    EulerStochastic,
    HeunDeterministic,
    HeunStochastic,
)
from tvb.simulator.noise import Additive
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.monitors import TemporalAverage


DT = 0.1


def _make_mpr_subnet(
    name: str,
    n_nodes: int,
    scheme=None,
    **model_kwargs,
) -> Subnetwork:
    model = MontbrioPazoRoxin(**model_kwargs)
    model.configure()
    subnet = Subnetwork(
        name=name,
        model=model,
        scheme=scheme or HeunDeterministic(dt=DT),
        nnodes=n_nodes,
    ).configure()
    subnet.node_indices = np.arange(n_nodes)
    return subnet


def _make_network(*subnets: Subnetwork) -> NetworkSet:
    network = NetworkSet(subnets=list(subnets), projections=[], stimuli=[])
    network.configure()
    return network


def _lower(n_nodes: int = 3, monitors=None, **model_kwargs) -> SpecLoweringResult:
    if monitors is None:
        monitors = [TemporalAverage(period=0.2)]
    subnet = _make_mpr_subnet("sn", n_nodes, **model_kwargs)
    return lower_network_set(_make_network(subnet), monitors=monitors)


class TestLoweringResult(unittest.TestCase):
    def test_returns_spec_lowering_result(self):
        result = _lower()
        self.assertIsInstance(result, SpecLoweringResult)
        self.assertIsInstance(result.spec, SimulationSpec)

    def test_analysis_attached(self):
        result = _lower()
        self.assertIsNotNone(result.analysis)


class TestSimulationSpec(unittest.TestCase):
    def test_dt_matches_integrator(self):
        result = _lower()
        self.assertAlmostEqual(result.spec.dt, DT)

    def test_backend_version_is_nonempty_string(self):
        result = _lower()
        self.assertIsInstance(result.spec.backend_version, str)
        self.assertGreater(len(result.spec.backend_version), 0)

    def test_no_projections_for_simple_network(self):
        result = _lower()
        self.assertEqual(len(result.spec.inter_projections), 0)
        self.assertEqual(len(result.spec.intra_projections), 0)

    def test_no_stimuli_for_simple_network(self):
        result = _lower()
        self.assertEqual(len(result.spec.stimuli), 0)

    def test_monitor_spec_type_name(self):
        result = _lower(monitors=[TemporalAverage(period=0.2)])
        self.assertEqual(len(result.spec.monitors), 1)
        self.assertEqual(result.spec.monitors[0].type_name, "TemporalAverage")

    def test_monitor_spec_period(self):
        result = _lower(monitors=[TemporalAverage(period=0.4)])
        self.assertAlmostEqual(float(result.spec.monitors[0].period), 0.4)

    def test_no_monitors_when_none_passed(self):
        subnet = _make_mpr_subnet("sn", 3)
        result = lower_network_set(_make_network(subnet), monitors=None)
        self.assertEqual(len(result.spec.monitors), 0)


class TestSubnetworkSpec(unittest.TestCase):
    def _subnet(self, **kwargs) -> SubnetworkSpec:
        return _lower(**kwargs).spec.subnetworks[0]

    def test_single_subnetwork(self):
        result = _lower()
        self.assertEqual(len(result.spec.subnetworks), 1)

    def test_model_type(self):
        self.assertEqual(self._subnet().model_type, "MontbrioPazoRoxin")

    def test_n_nodes(self):
        self.assertEqual(self._subnet(n_nodes=5).n_nodes, 5)

    def test_n_modes(self):
        self.assertEqual(self._subnet().n_modes, 1)

    def test_n_state_vars(self):
        self.assertEqual(self._subnet().n_state_vars, 2)

    def test_n_coupling_vars(self):
        self.assertEqual(self._subnet().n_coupling_vars, 2)

    def test_state_variables(self):
        sn = self._subnet()
        self.assertIn("r", sn.state_variables)
        self.assertIn("V", sn.state_variables)
        self.assertEqual(len(sn.state_variables), 2)

    def test_variables_of_interest(self):
        sn = self._subnet()
        self.assertGreater(len(sn.variables_of_interest), 0)
        for voi in sn.variables_of_interest:
            self.assertIn(voi, sn.state_variables)

    def test_initial_state_shape(self):
        sn = self._subnet(n_nodes=7)
        self.assertEqual(sn.initial_state_shape, (2, 7, 1))

    def test_no_stimulus(self):
        self.assertFalse(self._subnet().has_stimulus)


class TestIntegratorSpec(unittest.TestCase):
    def _integrator(self) -> IntegratorSpec:
        return _lower().spec.subnetworks[0].integrator

    def test_type_name(self):
        self.assertEqual(self._integrator().type_name, "HeunDeterministic")

    def test_dt(self):
        self.assertAlmostEqual(self._integrator().dt, DT)

    def test_not_stochastic(self):
        self.assertFalse(self._integrator().is_stochastic)

    def test_no_noise_nsig(self):
        self.assertIsNone(self._integrator().noise_nsig)


class TestParameterValues(unittest.TestCase):
    def _params(self, **model_kwargs):
        return _lower(**model_kwargs).spec.subnetworks[0].parameter_values

    def test_all_mpr_parameters_present(self):
        params = self._params()
        for name in ("tau", "Delta", "eta", "J", "I", "cr", "cv"):
            self.assertIn(name, params)

    def test_parameter_values_are_float64_arrays(self):
        for name, values in self._params().items():
            self.assertIsInstance(values, np.ndarray, msg=f"param {name}")
            self.assertEqual(values.ndim, 1, msg=f"param {name}")
            self.assertEqual(values.dtype, np.float64, msg=f"param {name}")

    def test_parameter_values_are_contiguous(self):
        for name, values in self._params().items():
            self.assertTrue(values.flags["C_CONTIGUOUS"], msg=f"param {name}")

    def test_I_reflects_configuration(self):
        params = self._params(I=np.array([3.5]))
        np.testing.assert_array_equal(params["I"], np.array([3.5]))

    def test_eta_reflects_configuration(self):
        params = self._params(eta=np.array([-2.0]))
        np.testing.assert_array_equal(params["eta"], np.array([-2.0]))


class TestSpecHoldsNoPythonObjects(unittest.TestCase):
    def test_model_type_is_string_not_object(self):
        sn = _lower().spec.subnetworks[0]
        self.assertIsInstance(sn.model_type, str)
        self.assertNotIsInstance(sn.model_type, MontbrioPazoRoxin)

    def test_integrator_is_spec_not_tvb_object(self):
        sn = _lower().spec.subnetworks[0]
        self.assertIsInstance(sn.integrator, IntegratorSpec)
        self.assertNotIsInstance(sn.integrator, HeunDeterministic)


class TestCacheKey(unittest.TestCase):
    def test_cache_key_is_64_char_hex(self):
        key = _lower().spec.cache_key()
        self.assertIsInstance(key, str)
        self.assertEqual(len(key), 64)
        int(key, 16)  # raises ValueError if not valid hex

    def test_cache_key_determinism(self):
        key1 = _lower(n_nodes=3).spec.cache_key()
        key2 = _lower(n_nodes=3).spec.cache_key()
        self.assertEqual(key1, key2)

    def test_cache_key_changes_with_n_nodes(self):
        key1 = _lower(n_nodes=2).spec.cache_key()
        key2 = _lower(n_nodes=4).spec.cache_key()
        self.assertNotEqual(key1, key2)

    def test_cache_key_changes_with_parameter(self):
        key1 = _lower(I=np.array([1.0])).spec.cache_key()
        key2 = _lower(I=np.array([2.0])).spec.cache_key()
        self.assertNotEqual(key1, key2)

    def test_cache_key_changes_with_dt(self):
        subnet1 = _make_mpr_subnet("sn", 3)
        subnet2 = Subnetwork(
            name="sn",
            model=MontbrioPazoRoxin(),
            scheme=HeunDeterministic(dt=0.05),
            nnodes=3,
        ).configure()
        subnet2.model.configure()
        subnet2.node_indices = np.arange(3)
        result1 = lower_network_set(_make_network(subnet1))
        result2 = lower_network_set(_make_network(subnet2))
        self.assertNotEqual(result1.spec.cache_key(), result2.spec.cache_key())


class TestCompatibilityGate(unittest.TestCase):
    def test_euler_deterministic_is_supported(self):
        subnet = _make_mpr_subnet("sn", 3, scheme=EulerDeterministic(dt=DT))
        result = lower_network_set(_make_network(subnet))
        self.assertEqual(
            result.spec.subnetworks[0].integrator.type_name,
            "EulerDeterministic",
        )
        self.assertFalse(result.spec.subnetworks[0].integrator.is_stochastic)

    def test_euler_stochastic_is_supported(self):
        subnet = _make_mpr_subnet(
            "sn",
            3,
            scheme=EulerStochastic(
                dt=DT,
                noise=Additive(nsig=np.array([0.01, 0.02]), noise_seed=123),
            ),
        )
        result = lower_network_set(_make_network(subnet))
        integrator = result.spec.subnetworks[0].integrator
        self.assertEqual(integrator.type_name, "EulerStochastic")
        self.assertTrue(integrator.is_stochastic)
        np.testing.assert_array_equal(
            integrator.noise_nsig,
            np.array([0.01, 0.02], dtype=np.float64),
        )

    def test_heun_stochastic_is_supported(self):
        subnet = _make_mpr_subnet(
            "sn",
            3,
            scheme=HeunStochastic(
                dt=DT,
                noise=Additive(nsig=np.array([0.0, 0.0]), noise_seed=123),
            ),
        )
        result = lower_network_set(_make_network(subnet))
        self.assertEqual(
            result.spec.subnetworks[0].integrator.type_name,
            "HeunStochastic",
        )

    def test_empty_network_raises(self):
        network = NetworkSet(subnets=[], projections=[], stimuli=[])
        network.configure()
        with self.assertRaises(ValueError):
            lower_network_set(network)

    def test_mismatched_dt_raises(self):
        sn1 = _make_mpr_subnet("sn1", 3)
        model2 = MontbrioPazoRoxin()
        model2.configure()
        sn2 = Subnetwork(
            name="sn2",
            model=model2,
            scheme=HeunDeterministic(dt=0.05),
            nnodes=3,
        ).configure()
        sn2.node_indices = np.arange(3)
        network = NetworkSet(subnets=[sn1, sn2], projections=[], stimuli=[])
        network.configure()
        with self.assertRaises(ValueError):
            lower_network_set(network)
