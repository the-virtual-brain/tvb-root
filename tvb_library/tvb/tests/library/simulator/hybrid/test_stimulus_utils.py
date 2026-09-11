# -*- coding: utf-8 -*-
"""Unit tests for :mod:`tvb.simulator.hybrid.stimulus_utils`."""
import numpy as np
import pytest
from tvb.simulator.models import JansenRit
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.hybrid import Subnetwork
from tvb.simulator.hybrid.stimulus_utils import (
    constant_stim, pulse_stim, sinusoid_stim,
)


@pytest.fixture
def subnet():
    return Subnetwork(
        name="test", model=JansenRit(), scheme=HeunDeterministic(dt=0.1), nnodes=3
    ).configure()


class TestConstantStim:
    def test_amplitude_on_target(self, subnet):
        stim = constant_stim(subnet, amplitude=1.5, target_cvar=0,
                             target_node=1, simulation_length=10.0)
        stim.configure(simulation_length=10.0)
        c = stim.get_coupling(step=0)
        assert c[0, 1, 0] == pytest.approx(1.5, rel=0.2)

    def test_shape(self, subnet):
        stim = constant_stim(subnet, amplitude=2.0, target_cvar=0,
                             simulation_length=10.0)
        stim.configure(simulation_length=10.0)
        c = stim.get_coupling(step=0)
        assert c.shape[1] == subnet.nnodes

    def test_time_invariant(self, subnet):
        stim = constant_stim(subnet, amplitude=3.0, target_cvar=0,
                             target_node=0, simulation_length=10.0)
        stim.configure(simulation_length=10.0)
        np.testing.assert_allclose(
            stim.get_coupling(step=0), stim.get_coupling(step=5))


class TestPulseStim:
    def test_has_nonzero_values(self, subnet):
        stim = pulse_stim(subnet, amplitude=5.0, target_cvar=0,
                          onset=0.0, period=20.0, pulse_width=10.0,
                          target_node=0, simulation_length=20.0)
        stim.configure(simulation_length=20.0)
        values = [stim.get_coupling(step=s).max() for s in range(50)]
        assert max(values) > 1.0, "pulse should have nonzero amplitude"

    def test_has_zero_period(self, subnet):
        stim = pulse_stim(subnet, amplitude=5.0, target_cvar=0,
                          onset=0.0, period=20.0, pulse_width=2.0,
                          target_node=0, simulation_length=20.0)
        stim.configure(simulation_length=20.0)
        values = [stim.get_coupling(step=s).max() for s in range(100)]
        assert min(values) == pytest.approx(0.0, abs=0.1), \
            "between-pulse period should be zero"


class TestSinusoidStim:
    def test_zero_at_origin(self, subnet):
        stim = sinusoid_stim(subnet, amplitude=3.0, target_cvar=0,
                             frequency=1.0, simulation_length=10.0)
        stim.configure(simulation_length=10.0)
        c = stim.get_coupling(step=0)
        assert c[0, 0, 0] == pytest.approx(0.0, abs=0.2)

    def test_has_positive_and_negative(self, subnet):
        stim = sinusoid_stim(subnet, amplitude=3.0, target_cvar=0,
                             frequency=1.0, simulation_length=10.0)
        stim.configure(simulation_length=10.0)
        values = [stim.get_coupling(step=s)[0, 0, 0] for s in range(100)]
        assert max(values) > 1.0
        assert min(values) < -1.0
