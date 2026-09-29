# -*- coding: utf-8 -*-
"""Monitor parity tests: each monitor supported by the hybrid path must
produce identical sample streams from CppHybridBackend and NbHybridBackend.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.hybrid.inter_projection import InterProjection
from tvb.simulator.hybrid.coupling import Linear
from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.backend.cpp_hybrid.backend import CppHybridBackend

DT = 0.1
NSTEP = 60


def _network():
    def mk(name, n, offset):
        model = MontbrioPazoRoxin()
        model.configure()
        sn = Subnetwork(name=name, model=model,
                        scheme=HeunDeterministic(dt=DT), nnodes=n,
                        node_indices=np.arange(offset, offset + n))
        sn.configure()
        return sn
    A, B = mk("A", 3, 0), mk("B", 4, 3)
    rng = np.random.RandomState(7)
    W = np.zeros((4, 3), np.float32)
    mask = rng.rand(4, 3) < 0.6
    W[mask] = rng.rand(mask.sum())
    proj = InterProjection(
        source=A, target=B,
        source_cvar=np.array([0], dtype=np.int_),
        target_cvar=np.array([0], dtype=np.int_),
        weights=sp.csr_matrix(W), lengths=sp.csr_matrix(W * 5.0),
        cfun=Linear(), scale=1.0, cv=1.0, dt=DT)
    ns = NetworkSet(subnets=[A, B], projections=[proj])
    ns.configure()
    return ns


def _run_pair(monitors, chunk_size=None):
    nb_out = NbHybridBackend().compile(_network(), eager=True).run(
        NSTEP, chunk_size=chunk_size, monitors=monitors)
    cpp_out = CppHybridBackend().run_network(_network(), NSTEP,
                                             chunk_size=chunk_size,
                                             monitors=monitors)
    return nb_out, cpp_out


def _assert_streams(nb_out, cpp_out, tag):
    assert len(nb_out) == len(cpp_out), tag
    for mi, (per_nb, per_cpp) in enumerate(zip(nb_out, cpp_out)):
        assert len(per_nb) == len(per_cpp), (tag, mi)
        for si, (t_nb, d_nb) in enumerate(per_nb):
            t_cp, d_cp = per_cpp[si]
            np.testing.assert_allclose(
                d_cp, d_nb, rtol=1e-3, atol=1e-4,
                err_msg=f"{tag} monitor#{mi} subnet#{si}")


def test_raw():
    from tvb.simulator.monitors import Raw
    nb, cpp = _run_pair([Raw()])
    _assert_streams(nb, cpp, "Raw")


def test_temporal_average():
    from tvb.simulator.monitors import TemporalAverage
    nb, cpp = _run_pair([TemporalAverage(period=0.5)])
    _assert_streams(nb, cpp, "TemporalAverage")


def test_subsample():
    from tvb.simulator.monitors import SubSample
    nb, cpp = _run_pair([SubSample(period=1.0)], chunk_size=1)
    _assert_streams(nb, cpp, "SubSample")


def test_global_average():
    from tvb.simulator.monitors import GlobalAverage
    nb, cpp = _run_pair([GlobalAverage(period=0.3)])
    _assert_streams(nb, cpp, "GlobalAverage")


def test_afferent_coupling():
    from tvb.simulator.monitors import AfferentCoupling
    nb, cpp = _run_pair([AfferentCoupling()])
    _assert_streams(nb, cpp, "AfferentCoupling")


def test_afferent_coupling_temporal_average():
    from tvb.simulator.monitors import AfferentCouplingTemporalAverage
    nb, cpp = _run_pair([AfferentCouplingTemporalAverage(period=0.5)])
    _assert_streams(nb, cpp, "AfferentCouplingTemporalAverage")


def test_spatial_average():
    from tvb.simulator.monitors import SpatialAverage
    m = SpatialAverage(period=0.5)
    m.configure()  # default weights (identity / uniform)
    nb, cpp = _run_pair([m])
    _assert_streams(nb, cpp, "SpatialAverage")


def test_raw_voi():
    from tvb.simulator.monitors import RawVoi
    nb, cpp = _run_pair([RawVoi()])
    _assert_streams(nb, cpp, "RawVoi")


def test_bold():
    from tvb.simulator.monitors import Bold
    nb, cpp = _run_pair([Bold(period=1.0)])
    _assert_streams(nb, cpp, "Bold")


def test_projection():
    from tvb.simulator.monitors import Projection
    from tvb.simulator.monitors import Projection as _P
    m = _P(period=0.5)
    ns = _network()
    n_node = sum(sn.nnodes for sn in ns.subnets)
    rng = np.random.RandomState(3)
    m.gain = rng.rand(2, n_node).astype(np.float32)
    nb, cpp = _run_pair([m])
    _assert_streams(nb, cpp, "Projection")
