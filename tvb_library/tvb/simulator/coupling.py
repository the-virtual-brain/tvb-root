# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
Coupling functions

The activity (state-variables) that have been propagated over the long-range
Connectivity pass through these functions before entering the equations
(Model.dfun()) describing the local dynamics.

The state-variable vector for the $k$-th node or region in the network can be expressed as:
Derivative = Noise + Local dynamics + Coupling(time delays).

More formally:

.. math::

         \\dot{\\Psi}_{k} = - \\Lambda\\left(\\Psi_{k}\\right) + Z \\left(\\Xi_{k} + \\sum_{j=1}^{l} u_{kj} \\Gamma_{v=2}[\\left(\\Psi_{k}(t),  \\Psi_{j}(t-\\tau_{kj}\\right)]\\right).


Here we compute the term Coupling(time delays) or 
:math:`\\sum_{j=1}^{l} u_{kj} \\Gamma_{v=2}[\\left(\\Psi_{k}(t),  \\Psi_{j}(t-\\tau_{kj}\\right)]`, 
where :math:`u_{kj}` are the elements of the weights matrix from a Connectivity datatype.

This term is equivalent to the dot product between the weights matrix (on the
left) and the delayed state vector. This order is important in the case 
case of an asymmetric connectivity matrix, where the
convention to distinguish target ($k$) and source ($j$) nodes is the
following:


.. math::

    \\left(\\begin{matrix} a & b \\
        c & d \\end{matrix}\\right)

         C_{kj}  &= \\left(\\begin{matrix} ^\\mathrm{To}/_\\mathrm{from} & 0 & 1 & 2 & \\cdots & l \\
                                                           0         & 1  & 1  &  0 & 1  &  0 \\
                                                           1         & 1  & 1  &  0 & 1  &  0 \\
                                                           2         & 1  & 0  &  0 & 1  &  0 \\
                                                     \\vdots          & 1  & 0  &  1 & 0  &  1 \\
                                                           l         & 0  & 0  &  0 & 0  &  0 \\
                                                           \\end{matrix}\\right)

.. NOTE: Our convention is the inverse of the BCT toolbox. Furthermore, this
         convention is consistent with the notation used in Physics and in our
         equation, ie, :math:`u_{kj}` matches row-column indexing and describes the
         connection strength from node j to node k


.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Noelia Montejo <Noelia@tvb.invalid>
.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>
.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>

"""

import numpy

from tvb.basic.neotraits.api import HasTraits, NArray, Attr, Range
from .history import SparseHistory
from .common import simple_gen_astr


class Coupling(HasTraits):
    r"""
    The base class for Coupling functions.

    Instances of the Coupling class are called by the simulator in the
    following way:

    .. math::

        k_i = coupling(g_ij, x_i, x_j)

    where g_ij is the connectivity weight matrix, x_i is the current state,
    x_j is the delayed state of the coupling variables chosen for the
    simulation, and k_i is the input to the ith node due to the coupling
    between the nodes.

    Coupling functions can all be defined as a combination of a
    pre-"synaptic" or pre-summation function, the summation over weighted
    afferents and the post-"synaptic" or post-summation function. Therefore,
    a Coupling subclass should not define the `__call__` method directly but
    rather appropriate `pre` and `post` methods, which are used by 
    `Coupling.__call__` to compute the coupling correctly.
    
    Default implementations of `pre` and `post` are provided, which simply
    apply the connectivity to afferent activity, without scaling or other changes.

    .. automethod:: PreSigmoidal.__call__

    """

    def __call__(self, step, history):
        g_ij = history.es_weights
        x_i, x_j = history.query(step)
        x_i = x_i[numpy.newaxis].transpose((2, 1, 0, 3))  # (to, ncv, from, m)
        pre = self.pre(x_i, x_j)
        sum = (g_ij * pre).sum(axis=2)  # (to, ncv, m)
        return self.post(sum).transpose((1, 0, 2))  # (ncv, to, m)

    def pre(self, x_i, x_j):
        return x_j

    def post(self, gx):
        return gx


class SparseCoupling(Coupling):
    """
    A coupling implementation which takes advantage of a sparse weights structure to reduce the
    number of coupling terms evaluated.

    """

    def _lri(self, nnz_row_el_idx):
        "Flat array of indices afferent, non-zero-weight connections."
        if not hasattr(self, '_cached_lri'):
            rows = numpy.r_[-1, nnz_row_el_idx]
            self._cached_lri, = numpy.argwhere(numpy.diff(rows)).T
            self._cached_nzr = numpy.unique(nnz_row_el_idx)
            self.log.debug('lri.size %d nzr.size %d', self._cached_lri.size, self._cached_nzr.size)
        return self._cached_lri, self._cached_nzr

    def __call__(self, step, history):
        h = history # type: SparseHistory
        x_i, x_j = h.query_sparse(step)
        assert x_i.shape == (h.n_cvar, h.n_node, h.n_mode)
        assert x_j.shape == (h.n_cvar, h.n_nnzw, h.n_mode)
        #                              ^ from (columns)

        sum = numpy.zeros_like(x_i)
        x_i = x_i[:, h.nnz_row_el_idx]
        assert x_i.shape == (h.n_cvar, h.n_nnzw, h.n_mode)
        #                              ^ to (rows)

        pre = self.pre(x_i, x_j)
        assert pre.shape == (h.n_cvar, h.n_nnzw, h.n_mode)

        weights_col = h.nnz_weights.reshape((h.n_nnzw, 1))
        lri, nzr = self._lri(h.nnz_row_el_idx)
        sum[:, nzr] = numpy.add.reduceat(weights_col * pre, lri, axis=1)
        return self.post(sum)


class Linear(SparseCoupling):
    r"""
    Provides a linear coupling function of the following form

    .. math::
        a x + b

    """

    a = NArray(
        label=":math:`a`",
        default=numpy.array([0.00390625,]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="Rescales the connection strength while maintaining the ratio "
            "between different values.")

    b = NArray(
        label=":math:`b`",
        default=numpy.array([0.0]),
        doc="Shifts the base of the connection strength while maintaining "
            "the absolute difference between different values.")

    parameter_names = 'a b'.split()
    pre_expr = 'x_j'
    post_expr = 'a * gx + b'

    def post(self, gx):
        return self.a * gx + self.b

    def __str__(self):
        return simple_gen_astr(self, 'a b')


class Scaling(SparseCoupling):
    r"""
    Provides a simple scaling of the connectivity of the form

    .. math::
        a x

    """

    a = NArray(
        label="Scaling factor",
        default=numpy.array([0.00390625]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="Rescales the connection strength while maintaining "
            "the ratio between different values."
    )

    def post(self, gx):
        return self.a * gx

    def __str__(self):
        return simple_gen_astr(self, 'a')


class HyperbolicTangent(SparseCoupling):
    r"""
    Provides a sigmoidal coupling function of the form

    .. math::
        a * (1 + tanh((x - midpoint)/sigma))

    NB: This coupling function is applied pre-summation. For a post-summation
        sigmoidal, see `Sigmoidal`.

    """

    a = NArray(
        label=":math:`a`",
        default=numpy.array([1.0]),
        domain=Range(lo=-1000.0, hi=1000.0, step=10.0),
        doc="Minimum of the sigmoid function")

    b = NArray(
        label=":math:`b`",
        default=numpy.array([1.0]),
        domain=Range(lo=-1.0, hi=1.0, step=10.0),
        doc="Scaling factor for the variable")

    midpoint = NArray(
        label="midpoint",
        default=numpy.array([0.0,]),
        domain=Range(lo=-1000.0, hi=1000.0, step=10.0),
        doc="Midpoint of the linear portion of the sigmoid")

    sigma = NArray(
        label=r":math:`\sigma`",
        default=numpy.array([1.0,]),
        domain=Range(lo=0.01, hi=1000.0, step=10.0),
        doc="Standard deviation of the coupling")

    def pre(self, x_i, x_j):
        return self.a * (1 +  numpy.tanh((self.b * x_j - self.midpoint) / self.sigma))

    def __str__(self):
        return simple_gen_astr(self, 'a b midpoint sigma')


class Sigmoidal(Coupling):
    r"""
    Provides a sigmoidal coupling function of the form

    .. math::
        c_{min} + (c_{max} - c_{min}) / (1.0 + \exp(-a(x-midpoint)/\sigma))

    NB: using a = numpy.pi / numpy.sqrt(3.0) and the default parameter 
        produces something close to the current default for
        Linear (a=0.00390625, b=0) over the linear portion of the sigmoid,
        with saturation at -1 and 1.

    """

    cmin = NArray(
        label=":math:`c_{min}`",
        default=numpy.array([-1.0,]),
        domain=Range(lo=-1000.0, hi=1000.0, step=10.0),
        doc="""Minimum of the sigmoid function""",)

    cmax = NArray(
        label=":math:`c_{max}`",
        default=numpy.array([1.0,]),
        domain=Range(lo=-1000.0, hi=1000.0, step=10.0),
        doc="""Maximum of the sigmoid function""",)

    midpoint = NArray(
        label="midpoint",
        default=numpy.array([0.0,]),
        domain=Range(lo=-1000.0, hi=1000.0, step=10.0),
        doc="Midpoint of the linear portion of the sigmoid",)

    a = NArray(
        label=r":math:`a`",
        default=numpy.array([1.0,]),
        domain=Range(lo=0.01, hi=1000.0, step=10.0),
        doc="Scaling of sigmoidal",)

    sigma = NArray(
        label=r":math:`\sigma`",
        default=numpy.array([230.0,]),
        domain=Range(lo=0.01, hi=1000.0, step=10.0),
        doc="Standard deviation of the sigmoidal",)

    parameter_names = 'cmin cmax midpoint a sigma'.split()
    pre_expr = 'x_j'
    post_expr = 'cmin + ((cmax - cmin) / (1.0 + exp(-a *((gx - midpoint) / sigma))))'

    def __str__(self):
        return simple_gen_astr(self, 'cmin cmax midpoint a sigma')

    def post(self, gx):
        return self.cmin + ((self.cmax - self.cmin) / (1.0 + numpy.exp(-self.a *((gx - self.midpoint) / self.sigma))))


class SigmoidalJansenRit(Coupling):
    r"""
    Provides a sigmoidal coupling function as described in the 
    Jansen and Rit model, of the following form

    .. math::
        c_{min} + (c_{max} - c_{min}) / (1.0 + \exp(-r(x-midpoint)/\sigma))

    Assumes that x has have two state variables.

    """

    cmin = NArray(
        label=":math:`c_{min}`",
        default=numpy.array([0.0,]),
        domain=Range(lo=-1000.0, hi=1000.0, step=10.0),
        doc="Minimum of the sigmoid function",)

    cmax = NArray(
        label=":math:`c_{max}`",
        default=numpy.array([2.0 * 0.0025,]),
        domain=Range(lo=-1000.0, hi=1000.0, step=10.0),
        doc="Maximum of the sigmoid function",)

    midpoint = NArray(
        label="midpoint",
        default=numpy.array([6.0,]),
        domain=Range(lo=-1000.0, hi=1000.0, step=10.0),
        doc="Midpoint of the linear portion of the sigmoid",)

    r = NArray(
        label=r":math:`r`",
        default=numpy.array([0.56,]),
        domain=Range(lo=0.01, hi=1000.0, step=10.0),
        doc="the steepness of the sigmoidal transformation",)

    a = NArray(
        label=r":math:`a`",
        default=numpy.array([1.0,]),
        domain=Range(lo=0.01, hi=1000.0, step=10.0),
        doc="Scaling of the coupling term",)

    def __str__(self):
        return simple_gen_astr(self, 'cmin cmax midpoint a r')

    def pre(self, x_i, x_j):
        pre = self.cmin + \
              (self.cmax - self.cmin) / (1.0 + numpy.exp(self.r * (self.midpoint - (x_j[:, 0] - x_j[:, 1]))))
        return pre[:, numpy.newaxis]

    def post(self, gx):
        return self.a * gx


class PreSigmoidal(Coupling):
    r"""
    Provides a pre-summation sigmoidal coupling function with a static or dynamic
    and local or global threshold.

    .. math::
        H * (Q + \tanh(G * (P*x - \theta)))

    The dynamic threshold as state variable given by the second state variable.
    With the coupling term, returns the direct node output for the dynamic threshold.

    """

    H = NArray(
        label="H",
        default=numpy.array([0.5,]),
        domain=Range(lo=-100.0, hi=100.0, step=1.0),
        doc="Global Factor.",)

    Q = NArray(
        label="Q",
        default=numpy.array([1.,]),
        domain=Range(lo=-100.0, hi=100.0, step=1.0),
        doc="Average.",)

    G = NArray(
        label="G",
        default=numpy.array([60.,]),
        domain=Range(lo=-1000.0, hi=1000.0, step=1.),
        doc="Gain.",)

    P = NArray(
        label="P",
        default=numpy.array([1.,]),
        domain=Range(lo=-100.0, hi=100.0, step=0.01),
        doc="Excitation-Inhibition ratio.",)

    theta = NArray(
        label=":math:`\\theta`",
        default=numpy.array([0.5,]),
        domain=Range(lo=-100.0, hi=100.0, step=0.01),
        doc="Threshold.",)

    dynamic = Attr(
        field_type=bool,
        label="Dynamic",
        default=True,
        doc="Use dynamic threshold (otherwise static).",)

    globalT = Attr(
        field_type=bool,
        label=":math:`global_{\\theta}`",
        default=False,
        doc="Use global threshold (otherwise local).",)

    def __str__(self):
        return simple_gen_astr(self, 'H Q G P theta dynamic globalT')

    def configure(self):
        """Set the right indirect call."""
        super(PreSigmoidal, self).configure()
        self.sliceT = 0 if self.globalT else slice(None)

    # override __call__ directly simpler than pre/post form
    # TODO check use of arrays dims here
    def __call__(self, step, history, na=numpy.newaxis):
        g_ij = history.es_weights
        x_i, x_j = history.query(step)
        if self.dynamic:
            _ = (self.P * x_j[:,0] - x_j[:,1,self.sliceT])[:,na]
        else:
            _ = self.P * x_j - self.theta[self.sliceT,na]
        A_j = self.H * (self.Q + numpy.tanh(self.G * _))
        if self.dynamic:
            c_0 = (g_ij[:,0] * A_j[:,0]).sum(axis=0)
            c_1 = numpy.diag(A_j[:,0,:,0])[:, na]
            if self.globalT:
                c_1[:] = c_1.mean()
            return numpy.array([c_0, c_1])
        else: # static threshold
            return (g_ij.transpose((2, 1, 0, 3)) * A_j).sum(axis=0)


class Difference(SparseCoupling):
    r"""
    Provides a difference coupling function, between pre and post synaptic
    activity of the form

    .. math::

        a G_ij (x_j - x_i)

    """

    a = NArray(
        label=":math:`a`",
        default=numpy.array([0.1,]),
        domain=Range(lo=0.0, hi=10., step=0.1),
        doc="Rescales the connection strength.",)

    def __str__(self):
        return simple_gen_astr(self, 'a')

    def pre(self, x_i, x_j):
        return x_j - x_i

    def post(self, gx):
        return self.a * gx


class Kuramoto(SparseCoupling):
    r"""
    Provides a Kuramoto-style coupling, a periodic difference of the form
    
    .. math::
        a / N G_ij sin(x_j - x_i)
    
    """
    a = NArray(
        label=":math:`a`",
        default=numpy.array([1.0,]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="Rescales the connection strength.",)

    def __str__(self):
        return simple_gen_astr(self, 'a')

    def pre(self, x_i, x_j):
        return numpy.sin(x_j - x_i)

    def post(self, gx):
        return self.a / gx.shape[0] * gx
