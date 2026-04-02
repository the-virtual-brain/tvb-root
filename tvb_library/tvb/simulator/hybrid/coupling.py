# -*- coding: utf-8 -*-
#
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
#

"""
Coupling functions for hybrid model

Coupling functions transform the afferent activity that has been propagated
over connectivity before it enters the local dynamics equations.

In the hybrid model framework, coupling functions operate on the weighted
afferent activity computed by projections.  Both hooks are applied *after*
the weighted sum over source nodes has been formed.

Mathematical Framework
-----------------------

For a projection from source subnetwork to target subnetwork, the
``BaseProjection.apply()`` pipeline is:

1. Extract delayed states from source: x_j(t - τ_ij)
2. Multiply by sparse weights: w_ij * x_j
3. Sum afferents per target node: s_i = Σ_j w_ij * x_j
4. Apply pre-scaling coupling:  s_i ← cfun.pre(s_i)   [optional]
5. Apply projection scale:      s_i ← scale * s_i
6. Apply post-scaling coupling: s_i ← cfun.post(s_i)  [optional]

The labels ``pre`` and ``post`` therefore refer to whether the
transformation is applied *before* or *after* the scalar ``scale``
factor, not relative to the weighted summation.  All coupling classes
currently implement their main logic in ``post()``; ``pre()`` is the
identity in every concrete subclass and is provided as an extension point.

The base Coupling class provides pre() and post() methods that can be
overridden to implement specific coupling behaviors.

"""

import numpy as np
import tvb.basic.neotraits.api as t


class Coupling(t.HasTraits):
    """Base class for coupling functions in hybrid model.

    Coupling functions transform afferent activity in projections via two
    hooks that ``BaseProjection.apply()`` calls around the scalar ``scale``
    factor:

    * ``pre(x)``  — called on the *already-summed* weighted afferent, before
      multiplication by ``scale``.
    * ``post(x)`` — called on the scaled afferent, immediately before the
      result is accumulated into the target coupling array.

    The default implementation returns the input unchanged (identity) for both
    hooks.  Subclasses should override ``post()`` (and optionally ``pre()``) to
    implement specific behaviours.

    Methods
    -------
    pre(x) : ndarray
        Apply pre-synaptic coupling transformation to afferent activity.
        Default: returns input unchanged (identity).

    post(x) : ndarray
        Apply post-synaptic coupling transformation to summed afferent activity.
        Default: returns input unchanged (identity).

    Examples
    --------
    >>> # Create a linear coupling: a * x + b
    >>> linear = Linear(a=0.5, b=0.1)
    >>> result = linear.post(np.array([1.0, 2.0, 3.0]))
    >>> # result = 0.5 * [1.0, 2.0, 3.0] + 0.1 = [0.6, 1.1, 1.6]

    >>> # Use in projection (via factory function)
    >>> proj = create_inter_projection(
    ...     source_subnet=cortex,
    ...     target_subnet=thalamus,
    ...     source_cvar='y0',
    ...     target_cvar='V',
    ...     weights=weights,
    ...     coupling=linear
    ... )

    Notes
    -----
    In the hybrid model framework, coupling functions are simpler than in
    classic TVB because they operate on already-summed afferent activity.
    The projection's apply() method handles:
    1. Delayed state extraction
    2. Weighted summation
    3. Coupling transformation (pre and/or post)
    4. Scaling

    See Also
    --------
    tvb.simulator.coupling : Classic TVB coupling functions for reference
    """

    def pre(self, x):
        """Apply pre-synaptic coupling transformation.

        Parameters
        ----------
        x : ndarray
            Afferent activity from source nodes. Shape depends on projection.

        Returns
        -------
        ndarray
            Transformed afferent activity. Same shape as input.
        """
        return x

    def post(self, x):
        """Apply post-synaptic coupling transformation.

        Parameters
        ----------
        x : ndarray
            Summed weighted afferent activity. Shape depends on projection.

        Returns
        -------
        ndarray
            Transformed summed activity. Same shape as input.
        """
        return x


class Linear(Coupling):
    """Linear coupling function: a * x + b.

    Applies a linear transformation with scaling (a) and offset (b).
    This is the most commonly used coupling function in TVB simulations.

    The transformation is applied post-summation:

    .. math::
        y = a \\cdot x + b

    Parameters
    ----------
    a : float or ndarray, default=1.0
        Scaling factor. Multiplies the summed afferent activity.
    b : float or ndarray, default=0.0
        Offset term. Added after scaling.

    Attributes
    ----------
    a : NArray
        Scaling factor with domain [0.0, 1.0] and default 1.0.
    b : NArray
        Offset with default 0.0.

    Examples
    --------
    >>> # Default: identity (a=1, b=0)
    >>> linear = Linear()
    >>> linear.post(np.array([1.0, 2.0]))
    array([1., 2.])

    >>> # Scale down by factor 0.5
    >>> linear = Linear(a=0.5, b=0.0)
    >>> linear.post(np.array([1.0, 2.0]))
    array([0.5, 1.])

    >>> # Scale and add offset
    >>> linear = Linear(a=0.5, b=0.1)
    >>> linear.post(np.array([1.0, 2.0]))
    array([0.6, 1.1])

    See Also
    --------
    Scaling : Simplified linear coupling with only scaling
    Sigmoidal : Nonlinear sigmoidal coupling
    """

    a = t.NArray(
        label=":math:`a`",
        default=np.array([1.0]),
        domain=t.Range(lo=0.0, hi=10.0, step=0.01),
        doc="Scaling factor for the coupling.",
    )

    b = t.NArray(
        label=":math:`b`",
        default=np.array([0.0]),
        doc="Offset added to the scaled coupling.",
    )

    def __init__(self, a=None, b=None, **kwargs):
        """Initialize Linear coupling with scalar or array parameters."""
        converted = {}
        for name, val in [('a', a), ('b', b)]:
            if val is not None:
                if not isinstance(val, np.ndarray):
                    val = np.array([float(val)])
                converted[name] = val
        super().__init__(**converted, **kwargs)

    def post(self, x):
        """Apply linear transformation: a * x + b.

        Parameters
        ----------
        x : ndarray
            Input summed afferent activity.

        Returns
        -------
        ndarray
            Transformed activity: a * x + b.
        """
        return self.a * x + self.b


class Scaling(Coupling):
    """Simple scaling coupling function: a * x.

    A simplified variant of linear coupling that applies only scaling
    without offset. Useful when you just need to adjust connection strength.

    The transformation is applied post-summation:

    .. math::
        y = a \\cdot x

    Parameters
    ----------
    a : float or ndarray, default=1.0
        Scaling factor. Multiplies the summed afferent activity.

    Attributes
    ----------
    a : NArray
        Scaling factor with domain [0.0, 1.0] and default 1.0.

    Examples
    --------
    >>> # Default: identity
    >>> scaling = Scaling()
    >>> scaling.post(np.array([1.0, 2.0]))
    array([1., 2.])

    >>> # Scale down
    >>> scaling = Scaling(a=0.5)
    >>> scaling.post(np.array([1.0, 2.0]))
    array([0.5, 1.])

    >>> # Scale up
    >>> scaling = Scaling(a=2.0)
    >>> scaling.post(np.array([1.0, 2.0]))
    array([2., 4.])

    Notes
    -----
    This is equivalent to Linear with b=0, but more explicit about intent.

    See Also
    --------
    Linear : Linear coupling with scaling and offset
    """

    a = t.NArray(
        label="Scaling factor",
        default=np.array([1.0]),
        domain=t.Range(lo=0.0, hi=10.0, step=0.01),
        doc="Rescales the connection strength.",
    )

    def __init__(self, a=None, **kwargs):
        """Initialize Scaling coupling with scalar or array parameter."""
        converted = {}
        if a is not None:
            if not isinstance(a, np.ndarray):
                a = np.array([float(a)])
            converted['a'] = a
        super().__init__(**converted, **kwargs)

    def post(self, x):
        """Apply scaling transformation: a * x.

        Parameters
        ----------
        x : ndarray
            Input summed afferent activity.

        Returns
        -------
        ndarray
            Scaled activity: a * x.
        """
        return self.a * x


class Sigmoidal(Coupling):
    """Sigmoidal coupling function with configurable bounds.

    Provides a sigmoidal (S-shaped) coupling that saturates at configurable
    minimum and maximum values. This is useful for modeling saturating
    synaptic transmission or threshold effects.

    The transformation is applied post-summation:

    .. math::
        y = c_{min} + \\frac{c_{max} - c_{min}}{1 + \\exp(-a \\cdot (x - midpoint) / \\sigma)}

    Parameters
    ----------
    cmin : float, default=-1.0
        Minimum value of the sigmoid (saturation lower bound).
    cmax : float, default=1.0
        Maximum value of the sigmoid (saturation upper bound).
    a : float, default=1.0
        Steepness/slope of the sigmoid curve.
    midpoint : float, default=0.0
        Midpoint of the linear portion of the sigmoid (inflection point).
    sigma : float, default=1.0
        Scale/width parameter controlling the steepness of the transition.

    Attributes
    ----------
    cmin : NArray
        Minimum value, domain [-1000.0, 1000.0], default -1.0.
    cmax : NArray
        Maximum value, domain [-1000.0, 1000.0], default 1.0.
    a : NArray
        Steepness parameter, domain [0.01, 1000.0], default 1.0.
    midpoint : NArray
        Inflection point, domain [-1000.0, 1000.0], default 0.0.
    sigma : NArray
        Width parameter, domain [0.01, 1000.0], default 1.0.

    Examples
    --------
    >>> # Default sigmoid centered at 0, bounds [-1, 1]
    >>> sigmoid = Sigmoidal()
    >>> sigmoid.post(np.array([-2.0, 0.0, 2.0]))  # Large negative, center, large positive
    array([-0.999...,  0.,  0.999...])

    >>> # Shift midpoint and adjust bounds
    >>> sigmoid = Sigmoidal(cmin=0.0, cmax=2.0, midpoint=1.0)
    >>> sigmoid.post(np.array([0.0, 1.0, 2.0]))
    array([0.001..., 1., 1.998...])

    Notes
    -----
    - For a=π/√3≈1.81, sigma=230, midpoint=0, cmin=-1, cmax=1,
      the linear region approximates Linear(a=0.004, b=0).
    - This is a post-synaptic coupling function (applied after summation).

    See Also
    --------
    Linear : Linear coupling for comparison
    tvb.simulator.coupling.Sigmoidal : Classic TVB implementation
    """

    cmin = t.NArray(
        label=":math:`c_{min}`",
        default=np.array([-1.0]),
        domain=t.Range(lo=-1000.0, hi=1000.0, step=10.0),
        doc="Minimum of the sigmoid function.",
    )

    cmax = t.NArray(
        label=":math:`c_{max}`",
        default=np.array([1.0]),
        domain=t.Range(lo=-1000.0, hi=1000.0, step=10.0),
        doc="Maximum of the sigmoid function.",
    )

    midpoint = t.NArray(
        label="midpoint",
        default=np.array([0.0]),
        domain=t.Range(lo=-1000.0, hi=1000.0, step=10.0),
        doc="Midpoint of the linear portion of the sigmoid.",
    )

    a = t.NArray(
        label=r":math:`a`",
        default=np.array([1.0]),
        domain=t.Range(lo=0.01, hi=1000.0, step=10.0),
        doc="Steepness of the sigmoid.",
    )

    sigma = t.NArray(
        label=r":math:`\sigma`",
        default=np.array([1.0]),
        domain=t.Range(lo=0.01, hi=1000.0, step=10.0),
        doc="Width of the sigmoid transition.",
    )

    def __init__(self, cmin=None, cmax=None, midpoint=None, a=None, sigma=None, **kwargs):
        """Initialize Sigmoidal coupling with scalar or array parameters."""
        # Convert scalar parameters to arrays, only pass if explicitly provided
        # so neotraits defaults are used when args are omitted
        converted = {}
        for name, val in [('cmin', cmin), ('cmax', cmax), ('midpoint', midpoint),
                          ('a', a), ('sigma', sigma)]:
            if val is not None:
                if not isinstance(val, np.ndarray):
                    val = np.array([float(val)])
                converted[name] = val
        super().__init__(**converted, **kwargs)

    def post(self, x):
        """Apply sigmoidal transformation.

        Parameters
        ----------
        x : ndarray
            Input summed afferent activity.

        Returns
        -------
        ndarray
            Transformed activity: cmin + (cmax-cmin) / (1 + exp(-a*(x-midpoint)/sigma)).
        """
        return self.cmin + (
            (self.cmax - self.cmin)
            / (1.0 + np.exp(-self.a * ((x - self.midpoint) / self.sigma)))
        )


class Kuramoto(Coupling):
    """Kuramoto (phase) coupling function.

    Implements the classic phase coupling used in Kuramoto models.
    The sine function provides a smooth, periodic interaction that tends to
    synchronize oscillators.

    The transformation is applied post-summation:

    .. math::
        y = \\frac{a}{N} \\cdot \\sin(x)

    where N is the number of modes (typically 1 for hybrid models).

    Parameters
    ----------
    a : float or ndarray, default=1.0
        Coupling strength. Multiplies the sine of the summed afferent activity.

    Attributes
    ----------
    a : NArray
        Coupling strength with domain [0.0, 10.0] and default 1.0.

    Examples
    --------
    >>> # Default coupling
    >>> kuramoto = Kuramoto()
    >>> kuramoto.post(np.array([0.0, np.pi/2, np.pi]))
    array([0., 1., 0.])

    >>> # Custom coupling strength
    >>> kuramoto = Kuramoto(a=np.array([0.5]))
    >>> kuramoto.post(np.array([np.pi/2]))
    array([0.5])

    Notes
    -----
    - This is a post-synaptic coupling function (applied after summation).
    - The Kuramoto coupling is widely used in synchronization studies.
    - For N modes > 1, the coupling is normalized by 1/N.

    References
    ----------
    Kuramoto, Y. (1984). Chemical oscillations, waves, and turbulence.
    Springer Series in Synergetics. Springer Berlin Heidelberg.

    See Also
    --------
    Linear : Linear coupling for comparison
    Difference : Diffusive coupling for comparison
    """

    a = t.NArray(
        label=r":math:`a`",
        default=np.array([1.0]),
        domain=t.Range(lo=0.0, hi=10.0, step=0.01),
        doc="Coupling strength for the sine function.",
    )

    def pre(self, x):
        """Apply pre-synaptic coupling (identity for Kuramoto).

        Parameters
        ----------
        x : ndarray
            Input afferent activity.

        Returns
        -------
        ndarray
            Unchanged input.
        """
        return x

    def post(self, x, mode=0):
        """Apply post-synaptic Kuramoto coupling: (a/N) * sin(x).

        Parameters
        ----------
        x : ndarray
            Input summed afferent activity.
        mode : int, default=0
            Number of modes for normalization. If mode > 1, coupling is
            normalized by 1/mode.

        Returns
        -------
        ndarray
            Transformed activity: (a / mode) * sin(x).
        """
        if mode is not None and mode > 1:
            return self.a[:, None] * np.sin(x) / mode
        return self.a[:, None] * np.sin(x)


class Difference(Coupling):
    """Diffusive (difference) coupling function.

    Implements the standard diffusive coupling used in physical systems.
    It tends to equalize values across nodes, leading to diffusion-like
    dynamics.

    The transformation is applied post-summation:

    .. math::
        y = a \\cdot (x_j - x_i) = a \\cdot x

    where in the hybrid framework, x represents the weighted sum of
    differences from incoming connections.

    Parameters
    ----------
    a : float or ndarray, default=1.0
        Coupling strength. Multiplies the summed afferent activity.

    Attributes
    ----------
    a : NArray
        Coupling strength with domain [0.0, 10.0] and default 1.0.

    Examples
    --------
    >>> # Default coupling (identity-like)
    >>> diff = Difference()
    >>> diff.post(np.array([1.0, 2.0, 3.0]))
    array([1., 2., 3.])

    >>> # Custom coupling strength
    >>> diff = Difference(a=np.array([0.5]))
    >>> diff.post(np.array([1.0, 2.0, 3.0]))
    array([0.5, 1. , 1.5])

    Notes
    -----
    - This is a post-synaptic coupling function (applied after summation).
    - Diffusive coupling is widely used in wave propagation studies.
    - In the hybrid framework, the weighted sum already represents
      the difference between incoming afferent and local state.

    See Also
    --------
    Linear : Linear coupling for comparison
    Kuramoto : Phase coupling for comparison
    """

    a = t.NArray(
        label=r":math:`a`",
        default=np.array([1.0]),
        domain=t.Range(lo=0.0, hi=10.0, step=0.01),
        doc="Coupling strength for diffusive coupling.",
    )

    def pre(self, x):
        """Apply pre-synaptic coupling (identity for Difference).

        Parameters
        ----------
        x : ndarray
            Input afferent activity.

        Returns
        -------
        ndarray
            Unchanged input.
        """
        return x

    def post(self, x, mode=0):
        """Apply post-synaptic diffusive coupling: a * x.

        Parameters
        ----------
        x : ndarray
            Input summed afferent activity (weighted sum of differences).
        mode : int, default=0
            Number of modes (not used in diffusive coupling).

        Returns
        -------
        ndarray
            Transformed activity: a * x.
        """
        return self.a[:, None] * x


class HyperbolicTangent(Coupling):
    """Hyperbolic tangent coupling function (pre-summation).

    Provides a smooth, asymmetric sigmoidal response centered at
    'midpoint'. The hyperbolic tangent ranges from -1 to +1, so this
    coupling function ranges from 0 to 2*a.

    The transformation is applied pre-summation:

    .. math::
        y = a \\cdot \\left(1 + \\tanh\\left(\\frac{x - \\text{midpoint}}{\\sigma}\\right)\\right)

    Parameters
    ----------
    a : float or ndarray, default=1.0
        Amplitude. Multiplies the tanh function.
    midpoint : float or ndarray, default=0.0
        Center of the tanh function (inflection point).
    sigma : float or ndarray, default=1.0
        Slope parameter (width of transition).

    Attributes
    ----------
    a : NArray
        Amplitude with domain [0.0, 10.0] and default 1.0.
    midpoint : NArray
        Center of the tanh, domain [-1000.0, 1000.0], default 0.0.
    sigma : NArray
        Width parameter, domain [0.01, 1000.0], default 1.0.

    Examples
    --------
    >>> # Default parameters
    >>> tanh_cfun = HyperbolicTangent()
    >>> tanh_cfun.pre(np.array([0.0]))
    array([1.])

    >>> # At midpoint, tanh(0) = 0, so output = a * (1 + 0) = 1
    >>> tanh_cfun.pre(np.array([0.0]))
    array([1.])

    >>> # At large positive, tanh(+inf) = 1, so output = a * (1 + 1) = 2
    >>> tanh_cfun.pre(np.array([10.0]))
    array([1.999...])

    >>> # At large negative, tanh(-inf) = -1, so output = a * (1 - 1) = 0
    >>> tanh_cfun.pre(np.array([-10.0]))
    array([0.000...])

    Notes
    -----
    - This is a pre-synaptic coupling function (applied before summation).
    - The output ranges from 0 to 2*a, providing a smooth transition.
    - The tanh function is smoother than the logistic sigmoid.

    See Also
    --------
    Sigmoidal : Logistic sigmoid coupling for comparison
    PreSigmoidal : Generalized pre-summation sigmoidal coupling
    """

    a = t.NArray(
        label=r":math:`a`",
        default=np.array([1.0]),
        domain=t.Range(lo=0.0, hi=10.0, step=0.01),
        doc="Amplitude of the tanh function.",
    )

    midpoint = t.NArray(
        label="midpoint",
        default=np.array([0.0]),
        domain=t.Range(lo=-1000.0, hi=1000.0, step=10.0),
        doc="Center of the tanh function.",
    )

    sigma = t.NArray(
        label=r":math:`\sigma`",
        default=np.array([1.0]),
        domain=t.Range(lo=0.01, hi=1000.0, step=10.0),
        doc="Width/slope of the tanh function.",
    )

    def pre(self, x, mode=0):
        """Apply pre-synaptic tanh coupling.

        Parameters
        ----------
        x : ndarray
            Input afferent activity.
        mode : int, default=0
            Number of modes (not used in pre-synaptic coupling).

        Returns
        -------
        ndarray
            Transformed activity: a * (1 + tanh((x - midpoint) / sigma)).
        """
        return self.a[:, None] * (
            1.0 + np.tanh((x - self.midpoint[:, None]) / self.sigma[:, None])
        )

    def post(self, x, mode=0):
        """Apply post-synaptic coupling (identity for pre-summation).

        Parameters
        ----------
        x : ndarray
            Input summed afferent activity.
        mode : int, default=0
            Number of modes (not used).

        Returns
        -------
        ndarray
            Unchanged input.
        """
        return x


class SigmoidalJansenRit(Coupling):
    """Sigmoidal coupling function for JansenRit model.

    Specialized coupling function designed for the JansenRit neural
    mass model. It applies sigmoidal transformation to inputs from
    excitatory pyramidal populations.

    Formula (pre-summation):

    .. math::
        y = a \\cdot \\frac{2 e_0}{1 + \\exp(r \\cdot (v_0 - x))}

    where e0, r, v0 are JansenRit model parameters.

    Parameters
    ----------
    a : float or ndarray, default=1.0
        Amplitude. Multiplies the sigmoidal function.
    e0 : float or ndarray, default=2.5
        Maximum firing rate (Hz).
    r : float or ndarray, default=0.56
        Slope of sigmoid.
    v0 : float or ndarray, default=6.0
        Firing threshold (mV).

    Attributes
    ----------
    a : NArray
        Amplitude with domain [0.0, 10.0] and default 1.0.
    e0 : NArray
        Maximum firing rate, domain [0.0, 100.0], default 2.5.
    r : NArray
        Slope parameter, domain [0.01, 10.0], default 0.56.
    v0 : NArray
        Threshold parameter, domain [-100.0, 100.0], default 6.0.

    Examples
    --------
    >>> # Default JansenRit parameters
    >>> jr_sigmoid = SigmoidalJansenRit()
    >>> jr_sigmoid.pre(np.array([6.0]))
    array([2.5])

    >>> # At threshold (x = v0), output = a * e0 = 2.5
    >>> jr_sigmoid.pre(np.array([6.0]))
    array([2.5])

    >>> # At high input, output approaches 2*a*e0 = 5.0
    >>> jr_sigmoid.pre(np.array([20.0]))
    array([4.99...])

    >>> # At low input, output approaches 0
    >>> jr_sigmoid.pre(np.array([-10.0]))
    array([0.00...])

    Notes
    -----
    - This is a pre-synaptic coupling function (applied before summation).
    - Model-specific: should only be used with JansenRit or compatible models.
    - Typically used for coupling between pyramidal populations (y0, y5).
    - The sigmoidal function models the firing rate transformation.

    References
    ----------
    Jansen, B. H., & Rit, V. G. (1995). Electroencephalogram and visual
    evoked potential generation in a mathematical model of coupled cortical
    columns. Biological Cybernetics, 73(4), 357-366.

    See Also
    --------
    Sigmoidal : General sigmoidal coupling
    HyperbolicTangent : Tanh-based coupling
    """

    a = t.NArray(
        label=r":math:`a`",
        default=np.array([1.0]),
        domain=t.Range(lo=0.0, hi=10.0, step=0.01),
        doc="Amplitude of the sigmoidal function.",
    )

    e0 = t.NArray(
        label=r":math:`e_0`",
        default=np.array([2.5]),
        domain=t.Range(lo=0.0, hi=100.0, step=0.1),
        doc="Maximum firing rate (Hz).",
    )

    r = t.NArray(
        label=r":math:`r`",
        default=np.array([0.56]),
        domain=t.Range(lo=0.01, hi=10.0, step=0.01),
        doc="Slope of the sigmoid.",
    )

    v0 = t.NArray(
        label=r":math:`v_0`",
        default=np.array([6.0]),
        domain=t.Range(lo=-100.0, hi=100.0, step=1.0),
        doc="Firing threshold (mV).",
    )

    def pre(self, x, mode=0):
        """Apply pre-synaptic JansenRit sigmoidal coupling.

        Parameters
        ----------
        x : ndarray
            Input afferent activity (membrane potential).
        mode : int, default=0
            Number of modes (not used in pre-synaptic coupling).

        Returns
        -------
        ndarray
            Transformed activity: a * (2*e0) / (1 + exp(r * (v0 - x))).
        """
        return (
            self.a[:, None]
            * (2 * self.e0[:, None])
            / (1.0 + np.exp(self.r[:, None] * (self.v0[:, None] - x)))
        )

    def post(self, x, mode=0):
        """Apply post-synaptic coupling (identity for pre-summation).

        Parameters
        ----------
        x : ndarray
            Input summed afferent activity.
        mode : int, default=0
            Number of modes (not used).

        Returns
        -------
        ndarray
            Unchanged input.
        """
        return x


class PreSigmoidal(Coupling):
    """Pre-summation sigmoidal coupling with dynamic threshold.

    Implements a more general sigmoidal coupling function that allows
    for dynamic thresholds and input projections.

    Formula (pre-summation):

    .. math::
        y = H \\cdot \\left(Q + \\tanh(G \\cdot (P \\cdot x - \\theta))\\right)

    Parameters
    ----------
    H : float or ndarray, default=1.0
        Amplitude scale. Multiplies the entire expression.
    Q : float or ndarray, default=0.0
        Baseline offset. Added to the tanh output.
    G : float or ndarray, default=1.0
        Input sensitivity (gain). Controls the steepness of tanh.
    P : float or ndarray, default=1.0
        Input projection. Scales the input before thresholding.
    theta : float or ndarray, default=0.0
        Activation threshold. Subtracted from scaled input.
    dynamic : bool, default=False
        Whether threshold is dynamic (not currently implemented).

    Attributes
    ----------
    H : NArray
        Amplitude with domain [0.0, 10.0] and default 1.0.
    Q : NArray
        Baseline offset, domain [-10.0, 10.0], default 0.0.
    G : NArray
        Input sensitivity, domain [0.01, 100.0], default 1.0.
    P : NArray
        Input projection, domain [-10.0, 10.0], default 1.0.
    theta : NArray
        Threshold, domain [-100.0, 100.0], default 0.0.
    dynamic : Bool
        Whether threshold is dynamic.

    Examples
    --------
    >>> # Default parameters: reduces to tanh(x)
    >>> pre_sigmoid = PreSigmoidal()
    >>> pre_sigmoid.pre(np.array([0.0]))
    array([0.])

    >>> # At threshold (x = theta = 0), tanh(0) = 0
    >>> pre_sigmoid.pre(np.array([0.0]))
    array([0.])

    >>> # At large positive, tanh(+inf) = 1, output = H * (Q + 1) = 1
    >>> pre_sigmoid.pre(np.array([10.0]))
    array([1.])

    >>> # With baseline Q = 1, output shifts upward
    >>> pre_sigmoid = PreSigmoidal(Q=np.array([1.0]))
    >>> pre_sigmoid.pre(np.array([10.0]))
    array([2.])

    Notes
    -----
    - This is a pre-synaptic coupling function (applied before summation).
    - The dynamic parameter is reserved for future implementation.
    - Provides more flexibility than simple sigmoidal functions.

    See Also
    --------
    Sigmoidal : Post-synaptic sigmoidal coupling
    HyperbolicTangent : Simpler tanh-based coupling
    """

    H = t.NArray(
        label=r":math:`H`",
        default=np.array([1.0]),
        domain=t.Range(lo=0.0, hi=10.0, step=0.01),
        doc="Amplitude scale.",
    )

    Q = t.NArray(
        label=r":math:`Q`",
        default=np.array([0.0]),
        domain=t.Range(lo=-10.0, hi=10.0, step=0.1),
        doc="Baseline offset.",
    )

    G = t.NArray(
        label=r":math:`G`",
        default=np.array([1.0]),
        domain=t.Range(lo=0.01, hi=100.0, step=0.1),
        doc="Input sensitivity (gain).",
    )

    P = t.NArray(
        label=r":math:`P`",
        default=np.array([1.0]),
        domain=t.Range(lo=-10.0, hi=10.0, step=0.1),
        doc="Input projection.",
    )

    theta = t.NArray(
        label=r":math:`\theta`",
        default=np.array([0.0]),
        domain=t.Range(lo=-100.0, hi=100.0, step=1.0),
        doc="Activation threshold.",
    )

    dynamic = t.NArray(
        label="dynamic",
        dtype=bool,
        default=np.array([False]),
        doc="Whether threshold is dynamic (reserved for future use).",
    )

    def pre(self, x, mode=0):
        """Apply pre-synaptic sigmoidal coupling.

        Parameters
        ----------
        x : ndarray
            Input afferent activity.
        mode : int, default=0
            Number of modes (not used in pre-synaptic coupling).

        Returns
        -------
        ndarray
            Transformed activity: H * (Q + tanh(G * (P * x - theta))).
        """
        return self.H[:, None] * (
            self.Q[:, None]
            + np.tanh(self.G[:, None] * (self.P[:, None] * x - self.theta[:, None]))
        )

    def post(self, x, mode=0):
        """Apply post-synaptic coupling (identity for pre-summation).

        Parameters
        ----------
        x : ndarray
            Input summed afferent activity.
        mode : int, default=0
            Number of modes (not used).

        Returns
        -------
        ndarray
            Unchanged input.
        """
        return x
