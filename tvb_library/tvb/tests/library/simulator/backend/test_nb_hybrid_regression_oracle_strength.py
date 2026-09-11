"""Sensitivity guards for hybrid-backend parity regression oracles.

These tests do not repeat backend parity coverage.  They verify that the
fixtures used by that coverage can reject the specific wrong implementations
described in the backend review handoffs.
"""

import numpy as np

from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.monitors import TemporalAverage

from . import test_nb_hybrid_regression_cfun_packing as packing
from . import test_nb_hybrid_regression_reducedset_modes as reduced_modes
from . import test_nb_hybrid_regression_resume_state as resume_state
from . import test_nb_hybrid_regression_sjr_classic as sjr


RTOL = 2e-5
ATOL = 2e-6


def _direct_sjr_cmin_run(value, nstep):
    """Fresh object graph and run, independent of either sweep path."""
    network = packing._network(packing._sjr(True, cmin=value))
    return NbHybridBackend().run_network(
        network,
        nstep=nstep,
        chunk_size=1,
        initial_states=[packing._initial_state()],
    )[0]


def test_long_moderately_coupled_sjr_oracle_rejects_two_x_coupling():
    """The classic-SJR fixture must not reproduce the old weak 15-step test."""
    nstep = 101
    state = sjr._initial_state()
    correct_network = sjr._network(a=1.0)
    doubled_network = sjr._network(a=2.0)
    projection = correct_network.subnets[0].projections[0]

    assert projection.scale >= 0.1
    np.testing.assert_array_equal(projection.source_cvar, sjr.TWO_SOURCE_CVARS)

    correct_coupling = sjr._classic_reference(correct_network, state, a=1.0)
    wrong_two_x = 2.0 * correct_coupling
    assert np.max(np.abs(correct_coupling)) > 0.1
    assert not np.allclose(correct_coupling, wrong_two_x, rtol=RTOL, atol=ATOL)

    correct = sjr._python_trajectory(correct_network, state, nstep)
    doubled = sjr._python_trajectory(doubled_network, state, nstep)
    per_step_error = np.max(np.abs(correct - doubled), axis=(1, 2, 3))

    assert per_step_error[-1] > 0.02
    assert per_step_error[-1] > 20.0 * per_step_error[14]
    assert not np.allclose(correct, doubled, rtol=1e-3, atol=1e-3)


def test_sweep_rows_match_fresh_full_time_oracles_and_reject_wrong_slot():
    """Sweep parity cannot pass through shared packing or time averaging bugs."""
    nstep = 32
    values = np.array([-0.2, 0.2], dtype=np.float32)
    initial = packing._initial_state()
    base_cfun = packing._sjr(True)
    packed_parameters = np.array(
        [base_cfun.a[0], base_cfun.cmin[0], base_cfun.cmax[0],
         base_cfun.r[0], base_cfun.midpoint[0]]
    )
    assert np.unique(packed_parameters).size == packed_parameters.size

    sequential = NbHybridBackend().sweep(
        packing._network(packing._sjr(True)),
        params={"ctx.intra.cmin": values},
        nstep=nstep,
        backend="cpu",
        n_workers=1,
        initial_states=[initial],
    )
    direct = [_direct_sjr_cmin_run(value, nstep) for value in values]

    assert sequential.times.shape == (nstep,)
    for row, (times, states, coupling) in enumerate(direct):
        np.testing.assert_allclose(sequential.times, times, rtol=0.0, atol=1e-8)
        np.testing.assert_allclose(
            sequential.tavg["ctx"][row], states, rtol=RTOL, atol=ATOL
        )
        np.testing.assert_allclose(
            sequential.ctavg["ctx"][row], coupling, rtol=RTOL, atol=ATOL
        )

    assert not np.allclose(
        direct[0][1], direct[1][1], rtol=RTOL, atol=ATOL
    ), "full-time state oracle is insensitive to the swept cmin values"
    assert not np.allclose(
        direct[0][2], direct[1][2], rtol=RTOL, atol=ATOL
    ), "full-time coupling oracle is insensitive to the swept cmin values"

    # Reproduce the historical descriptor/packed-slot confusion: put the row's
    # cmin value into midpoint while leaving cmin at its base value.
    wrong_network = packing._network(packing._sjr(True, midpoint=float(values[1])))
    wrong_slot = NbHybridBackend().run_network(
        wrong_network,
        nstep=nstep,
        chunk_size=1,
        initial_states=[initial],
    )[0]
    assert not np.allclose(
        direct[1][1], wrong_slot[1], rtol=RTOL, atol=ATOL
    ), "full-time state oracle does not reject wrong-slot serialization"
    assert not np.allclose(
        direct[1][2], wrong_slot[2], rtol=RTOL, atol=ATOL
    ), "full-time coupling oracle does not reject wrong-slot serialization"
    assert not np.allclose(
        direct[1][1].mean(axis=0), wrong_slot[1].mean(axis=0),
        rtol=RTOL, atol=ATOL,
    )


def test_reduced_set_oracle_rejects_mode_zero_only_alternative():
    """Keep all three unequal modes visible before any observation collapse."""
    network, initial = reduced_modes._build_case(
        reduced_modes.ReducedSetFitzHughNagumo, "intra"
    )
    coupling = reduced_modes._total_initial_coupling(network, initial)[0]
    wrong_mode_zero_only = np.zeros_like(coupling)
    wrong_mode_zero_only[..., 0] = coupling[..., 0]

    assert coupling.shape[-1] == 3
    assert np.max(np.ptp(initial[0], axis=-1)) > 0.5
    assert np.max(np.abs(coupling[..., 1:])) > 0.1
    assert not np.allclose(coupling, wrong_mode_zero_only, rtol=RTOL, atol=ATOL)


def test_temporal_average_oracle_rejects_half_step_timestamp_shift():
    """Check absolute midpoint values, not merely spacing or averaged data."""
    period_steps = 4
    nstep = 12
    monitor = TemporalAverage(period=period_steps * resume_state.DT)
    result = resume_state._compiled().run(
        nstep,
        initial_states=resume_state._initial_state(),
        monitors=[monitor],
    )
    times, _data = result[0][0]
    expected = (
        np.arange(nstep // period_steps) * period_steps
        + period_steps / 2.0
    ) * resume_state.DT
    wrong_shifted = expected + 0.5 * resume_state.DT

    np.testing.assert_allclose(times, expected, rtol=0.0, atol=3e-8)
    assert not np.allclose(times, wrong_shifted, rtol=0.0, atol=3e-8)
