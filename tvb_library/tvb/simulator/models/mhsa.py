# -*- coding: utf-8 -*-
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

import numpy as np
from tvb.simulator.models.base import Model
import tvb.basic.neotraits as t


class MHSA(Model):
    """
    A neural mass model implementing linear self-attention dynamics.
    The model treats each node as representing a query that attends to all other nodes (keys/values).

    """

    n_head: int = t.api.Int(default=4)
    head_size: int = t.api.Int(default=8)

    state_variables = ('vt',)
    variables_of_interest = ('vt',)
    _nvar = 1
    cvar = np.r_[0]

    def __init__(self, **kwargs):
        self.number_of_modes = self.n_head * self.head_size

    def phi(self, x):
        return 1/(1 + np.exp(-x))

    def _pt_ref(self, x):
        "PyTorch ref impl from github.com/maedoc/nanogpt"
        B, T, C = x.size()
        # z-scored activity in thalamus
        z = (x - x.mean(axis=-1)[..., None])/x.std(dim=-1)[..., None]
        # projection to cortex
        sh = B, T, 3, self.n_head, C//self.n_head
        q, k, v = self.c_attn(z).view(*sh).permute(2, 0, 3, 1, 4)
        # l23 salience
        s = self.phi(q) @ self.phi(k).transpose(-2, -1)
        # l5 transformed values
        vt = (s * self.bias[:, :, :T, :T]) @ v
        # project back to thalamus
        y = vt.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        # thalamus has skip connection
        return x + y

    def dfun(self, _, coupling, local_coupling=0.0):
        # assume afferents here from thalamus as in _pt_ref above
        T, _, C = coupling.shape
        assert C == self.number_of_modes
        q, k, v = coupling.reshape(
            T, 3, self.n_head, self.head_size).transpose(1, 0, 2, 3)
        # l23 salience, XXX how to mask?
        s = np.einsum('Tmd,tmd->Ttm', self.phi(q), self.phi(k))
        # l5 transformed values
        vt = np.einsum('Ttm,tmd->Tmd', s, v)
        return vt.reshape(T, self.number_of_modes)



