"""Regression coverage for unsupported hybrid coupling functions."""

import numpy as np
import pytest
import scipy.sparse as sp

from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.hybrid.coupling import Coupling, Linear
from tvb.simulator.hybrid.intra_projection import IntraProjection
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.integrators import EulerDeterministic
from tvb.simulator.models.linear import Linear as LinearModel


class _OffsetCoupling(Coupling):
    def post(self, x):
        return x + 3.0


def _network(cfun):
    model = LinearModel()
    model.configure()
    subnet = Subnetwork(
        name="linear",
        model=model,
        scheme=EulerDeterministic(dt=0.1),
        nnodes=1,
    )
    subnet.projections = [
        IntraProjection(
            source_cvar=np.array([0], dtype=np.int_),
            target_cvar=np.array([0], dtype=np.int_),
            weights=sp.csr_matrix([[1.0]]),
            lengths=sp.csr_matrix((1, 1), dtype=np.float64),
            cv=1.0,
            dt=0.1,
            cfun=cfun,
        )
    ]
    subnet.configure()
    network = NetworkSet(subnets=[subnet], projections=[])
    network.configure()
    return network


def test_linear_coupling_remains_supported():
    compiled = NbHybridBackend().compile(
        _network(Linear(a=np.array([2.0]), b=np.array([1.0]))),
        eager=False,
    )

    assert compiled is not None


def test_custom_coupling_is_rejected_instead_of_compiled_as_identity():
    coupling = _OffsetCoupling()
    sample = np.array([1.0, 2.0])
    assert not np.array_equal(coupling.post(sample), sample)

    with pytest.raises(NotImplementedError) as exc_info:
        NbHybridBackend().compile(_network(coupling), eager=False)

    message = str(exc_info.value)
    assert "coupling" in message.lower()
    assert type(coupling).__name__ in message
