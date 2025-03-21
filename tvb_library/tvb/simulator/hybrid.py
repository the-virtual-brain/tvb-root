# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2024, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
This modules defines a set of classes for hybrid-model simulation.

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import collections
import numpy as np
import tvb.basic.neotraits.api as t
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.models import Model


class Subnetwork(t.HasTraits):
    "Represents a subnetwork of a connectome."

    conn: Connectivity = t.Attr(Connectivity, required=True)
    mask: np.ndarray = t.NArray(dtype=np.bool_, required=True)
    name: str = t.Attr(str)
    model: Model = t.Attr(Model, required=True)

    def zero_states(self) -> np.ndarray:
        return np.zeros((self.model.nvar, self.mask.sum()))

    def zero_cvars(self) -> np.ndarray:
        return np.zeros((self.model.cvar.size, self.mask.sum()))


class Projection(t.HasTraits):
    """Represents a projection from a source subnetwork to a target
    subnetwork, specifying indices of coupling variables."""

    source: Subnetwork = t.Attr(Subnetwork, required=True)
    target: Subnetwork = t.Attr(Subnetwork, required=True)
    source_cvar: int = t.Int(required=True)
    target_cvar: int = t.Int(required=True)
    scale: float = t.Float(default=1.0)

    # if not provided, default to source.conn
    conn: Connectivity = t.Attr(Connectivity)

    # cfun = t.Attr(tvb.coupling.Coupling)  # TODO

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert (self.source.mask * self.target.mask).sum() == 0, 'overlap'
        assert self.source.conn.gid == self.target.conn.gid
        try:
            self.conn
        except:
            self.conn = self.source.conn
        self._weights = self.conn.weights[self.target.mask][:, self.source.mask]
        # TODO time delays

    def apply(self, tgt, src):
        tgt[self.target_cvar] += self.scale * src[self.source_cvar] @ self._weights.T


class NetworkSet(t.HasTraits):
    """Collects subnetworks & projections, with methods for evaluating
    full network coupling and time stepping.
    """
    subnets: [Subnetwork] = t.List(of=Subnetwork)
    projections: [Subnetwork] = t.List(of=Projection)

    # NOTE dynamically generated namedtuple based on subnetworks
    States: collections.namedtuple = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.States = collections.namedtuple(
            'States',
            ' '.join([_.name for _ in self.subnets]))
        self.States.shape = property(lambda self: [_.shape for _ in self])

    def zero_states(self) -> States:
        return self.States(*[_.zero_states() for _ in self.subnets])

    def zero_cvars(self) -> States:
        return self.States(*[_.zero_cvars() for _ in self.subnets])

    def cfun(self, eff: States) -> States:
        "applies all projections to compute total coupling."
        aff = self.zero_cvars()
        for p in self.projections:
            tgt = getattr(aff, p.target.name)
            src = getattr(eff, p.source.name)
            p.apply(tgt, src)
        return aff

    def step(self, scheme, xs: States) -> States:
        cs = self.cfun(xs)
        nxs = self.zero_states()
        for sn, nx, x, c in zip(self.subnets, nxs, xs, cs):
            nx[:] = scheme(x[..., None],
                           sn.model.dfun,
                           c[..., None], 0, 0)[:, :, 0]
        return nxs

