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
2. Apply pre-scaling coupling:       cfun.pre(x_j, x_i) [optional]
   - x_i is the current target state (when available)
   - For Difference/Kuramoto, pre computes x_j - x_i / sin(x_j - x_i)
   - When x_i is unavailable, pre falls back to identity / sin(x_j)
3. Multiply by sparse weights:       w_ij * pre(x_j, x_i)
4. Sum afferents per target node:    s_i = Σ_j w_ij * pre(x_j, x_i)
5. Apply projection scale:           s_i ← scale * s_i
6. Apply post-scaling coupling:      s_i ← cfun.post(s_i)  [optional]

The labels ``pre`` and ``post`` therefore refer to whether the
transformation is applied *before* or *after* the scalar ``scale``
factor, not relative to the weighted summation.  ``post()`` is the
main hook for most coupling classes.  ``pre()`` is used by coupling
functions that require access to both source and target activity
(e.g., Difference, Kuramoto) to compute relative quantities.

The base ``Coupling`` class provides default ``pre(x_j, x_i=None)``
and ``post(x)`` methods.  ``pre()`` accepts an optional second argument
``x_i`` for coupling functions that depend on both source and target
states.  When ``x_i`` is not provided, ``pre()`` returns ``x_j``
unchanged.

"""

import numpy as np
import warnings
import tvb.basic.neotraits.api as t


class Coupling(t.HasTraits):
    """Base class for coupling functions in hybrid model.

    Coupling functions transform afferent activity in projections via two
    hooks that ``BaseProjection.apply()`` calls around the scalar ``scale``
    factor:

    * ``pre(x_j, x_i=None)`` — called on the source (afferent) activity,
      optionally using the target state to compute relative quantities.
      Applied before multiplication by ``scale``.
    * ``post(x)`` — called on the scaled afferent, immediately before the
      result is accumulated into the target coupling array.

    The default implementation returns the input unchanged (identity) for both
    hooks.  Subclasses should override ``post()`` (and optionally ``pre()``) to
    implement specific behaviours.

    Methods
    -------
    pre(x_j, x_i=None) : ndarray
        Apply pre-synaptic coupling transformation to afferent activity.
        When ``x_i`` is provided, coupling functions like Difference and
        Kuramoto can compute relative quantities (e.g., ``x_j - x_i``,
        ``sin(x_j - x_i)``).  Default: returns ``x_j`` unchanged.
    
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

    def pre(self, x_j, x_i=None):
        """Apply pre-synaptic coupling transformation.

        Parameters
        ----------
        x_j : ndarray
            Afferent (source) activity. Shape depends on projection.
        x_i : ndarray, optional
            Target (local) activity. Required for Difference and Kuramoto
            coupling where the transformation depends on both source and
            target states.

        Returns
        -------
        ndarray
            Transformed afferent activity. Same shape as input.
        """
        return x_j

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

    In the classic TVB framework, the pre-synaptic transformation computes
    the sine of the phase difference between source and target:

    .. math::
        \\mathrm{pre}(x_j, x_i) = \\sin(x_j - x_i)

    After weighted summation, the post-synaptic transformation applies scaling:

    .. math::
        \\mathrm{post}(gx) = a \\cdot gx

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
    >>> # Default coupling
    >>> kuramoto = Kuramoto()
    >>> kuramoto.post(np.array([0.0, 0.5]))
    array([0., 0.5])

    >>> # Custom coupling strength
    >>> kuramoto = Kuramoto(a=np.array([0.5]))
    >>> kuramoto.post(np.array([0.0, 0.5]))
    array([0. , 0.25])

    >>> # With both source and target activity
    >>> kuramoto = Kuramoto()
    >>> kuramoto.pre(np.array([np.pi/2, np.pi]), np.array([0.0, 0.0]))
    array([1., 0.])

    Notes
    -----
    - The ``pre()`` method requires both ``x_j`` (source) and ``x_i`` (target)
      to compute the phase difference.  When only ``x_j`` is provided,
      ``pre()`` returns ``sin(x_j)``.
    - The ``post()`` method applies ``a[:, None] / N * gx`` with
      ``N = gx.shape[0]``, matching classic TVB's normalization by the
      number of coupling variables.

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

    def pre(self, x_j, x_i=None):
        """Compute the sine of the phase difference between source and target.

        Parameters
        ----------
        x_j : ndarray
            Afferent (source) activity.
        x_i : ndarray, optional
            Target (local) activity.  When provided, returns
            ``sin(x_j - x_i)``.  When *None*, returns ``sin(x_j)``.

        Returns
        -------
        ndarray
            ``sin(x_j - x_i)`` when ``x_i`` is provided, otherwise
            ``sin(x_j)``.
        """
        if x_i is None:
            return np.sin(x_j)
        return np.sin(x_j - x_i)

    def post(self, gx):
        """Apply post-synaptic Kuramoto coupling: a / N * gx.

        Parameters
        ----------
        gx : ndarray
            Input summed afferent activity.  Shape
            ``(n_cvar, n_nodes, n_modes)`` where ``n_cvar`` is the
            number of coupling variables.

        Returns
        -------
        ndarray
            Transformed activity: ``a[:, None] / N * gx`` where
            ``N = gx.shape[0]`` is the number of coupling variables.

        Notes
        -----
        This matches classic TVB's ``Kuramoto.post(gx)`` which
        normalizes by ``gx.shape[0]`` (the number of coupling
        variables).  In the common case ``N = 1`` so the normalization
        has no effect, but for multi-mode or multi-cvar projections
        it ensures the coupling strength is correctly distributed.
        """
        N = gx.shape[0]
        return self.a[:, None] / N * gx


class Difference(Coupling):
    """Diffusive (difference) coupling function.

    Implements the standard diffusive coupling used in physical systems.
    It tends to equalize values across nodes, leading to diffusion-like
    dynamics.

    In the classic TVB framework, the pre-synaptic transformation computes
    the difference between source and target activity:

    .. math::
        \\mathrm{pre}(x_j, x_i) = x_j - x_i

    After weighted summation, the post-synaptic transformation applies scaling:

    .. math::
        \\mathrm{post}(gx) = a \\cdot gx

    .. versionchanged:: 2.8
        The default coupling strength ``a`` changed from ``1.0`` to ``0.1``
        to match the classic TVB ``Difference`` coupling.

    Parameters
    ----------
    a : float or ndarray, default=0.1
        Coupling strength. Multiplies the summed afferent activity.

    Attributes
    ----------
    a : NArray
        Coupling strength with domain [0.0, 10.0] and default 0.1.

    Examples
    --------
    >>> # Default coupling
    >>> diff = Difference()
    >>> diff.post(np.array([1.0, 2.0, 3.0]))
    array([0.1, 0.2, 0.3])

    >>> # Custom coupling strength
    >>> diff = Difference(a=np.array([0.5]))
    >>> diff.post(np.array([1.0, 2.0, 3.0]))
    array([0.5, 1. , 1.5])

    >>> # With both source and target activity
    >>> diff = Difference()
    >>> diff.pre(np.array([2.0, 3.0]), np.array([1.0, 1.0]))
    array([1., 2.])

    Notes
    -----
    - The ``pre()`` method requires both ``x_j`` (source) and ``x_i`` (target)
      to compute the difference.  When only ``x_j`` is provided, ``pre()``
      returns ``x_j`` unchanged (identity), which is only correct if the
      weighted summation already encodes the difference.
    - This matches the classic TVB ``Difference`` coupling semantics where
      ``pre(x_i, x_j) = x_j - x_i`` followed by ``post(gx) = a * gx``.

    See Also
    --------
    Linear : Linear coupling for comparison
    Kuramoto : Phase coupling for comparison
    """

    a = t.NArray(
        label=r":math:`a`",
        default=np.array([0.1]),
        domain=t.Range(lo=0.0, hi=10.0, step=0.01),
        doc="Coupling strength for diffusive coupling.",
    )

    def pre(self, x_j, x_i=None):
        """Compute the difference between source and target activity.

        Parameters
        ----------
        x_j : ndarray
            Afferent (source) activity.
        x_i : ndarray, optional
            Target (local) activity.  When provided, returns ``x_j - x_i``.
            When *None*, returns ``x_j`` unchanged (identity).

        Returns
        -------
        ndarray
            ``x_j - x_i`` when ``x_i`` is provided, otherwise ``x_j``.
        """
        if x_i is None:
            return x_j
        return x_j - x_i

    def post(self, gx):
        """Apply post-synaptic diffusive coupling: a * gx.

        Parameters
        ----------
        gx : ndarray
            Input summed afferent activity (weighted sum of differences).

        Returns
        -------
        ndarray
            Transformed activity: a[:, None] * gx.
        """
        return self.a[:, None] * gx


class HyperbolicTangent(Coupling):
    """Hyperbolic tangent coupling function (pre-summation).

    Provides a smooth, asymmetric sigmoidal response centered at
    'midpoint'. The hyperbolic tangent ranges from -1 to +1, so this
    coupling function ranges from 0 to 2*a.

    The transformation is applied pre-summation:

    .. math::
        y = a \\cdot \\left(1 + \\tanh\\left(\\frac{b \\cdot x - \\text{midpoint}}{\\sigma}\\right)\\right)

    Parameters
    ----------
    a : float or ndarray, default=1.0
        Amplitude. Multiplies the tanh function.
    b : float or ndarray, default=1.0
        Input scaling factor. Scales the input before applying tanh.
    midpoint : float or ndarray, default=0.0
        Center of the tanh function (inflection point).
    sigma : float or ndarray, default=1.0
        Slope parameter (width of transition).

    Attributes
    ----------
    a : NArray
        Amplitude with domain [0.0, 10.0] and default 1.0.
    b : NArray
        Input scaling factor with domain [-1.0, 10.0] and default 1.0.
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

    b = t.NArray(
        label=r":math:`b`",
        default=np.array([1.0]),
        domain=t.Range(lo=-1.0, hi=10.0, step=0.01),
        doc="Input scaling factor.",
    )

    def __init__(self, a=None, b=None, midpoint=None, sigma=None, **kwargs):
        """Initialize HyperbolicTangent coupling with scalar or array parameters."""
        converted = {}
        for name, val in [('a', a), ('b', b), ('midpoint', midpoint), ('sigma', sigma)]:
            if val is not None:
                if not isinstance(val, np.ndarray):
                    val = np.array([float(val)])
                converted[name] = val
        super().__init__(**converted, **kwargs)

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
            Transformed activity: a * (1 + tanh((b*x - midpoint) / sigma)).
        """
        return self.a[:, None] * (
            1.0 + np.tanh((self.b[:, None] * x - self.midpoint[:, None]) / self.sigma[:, None])
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

    When ``use_classic`` is ``1`` (default), this class matches the
    classic TVB ``SigmoidalJansenRit`` coupling: it expects two source
    state variables, computes their difference, and applies a logistic
    sigmoid using ``cmin``, ``cmax``, and ``midpoint``.

    Formula (classic mode, pre-summation):

    .. math::
        y = c_{min} + \\frac{c_{max} - c_{min}}{1 + \\exp(r \\cdot (midpoint - (x_0 - x_1)))}

    Post-summation, the result is multiplied by amplitude ``a``.

    When ``use_classic`` is ``0``, the legacy hybrid formula is used:

    .. math::
        y = a \\cdot \\frac{2 e_0}{1 + \\exp(r \\cdot (v_0 - x))}

    Parameters
    ----------
    a : float or ndarray, default=1.0
        Amplitude. In classic mode this is applied in post(); in legacy
        mode it is folded into pre().
    e0 : float or ndarray, default=2.5
        Maximum firing rate (Hz). **Deprecated**: only used when
        ``use_classic=0``.
    r : float or ndarray, default=0.56
        Slope of sigmoid.
    v0 : float or ndarray, default=6.0
        Firing threshold (mV). **Deprecated**: only used when
        ``use_classic=0``.
    cmin : float or ndarray, default=0.0
        Minimum value of the sigmoid (classic mode).
    cmax : float or ndarray, default=0.005
        Maximum value of the sigmoid (classic mode).
    midpoint : float or ndarray, default=6.0
        Inflection point of the sigmoid (classic mode).
    use_classic : int, default=1
        If 1, use the classic TVB formula with cmin/cmax/midpoint.
        If 0, use the legacy e0/v0 formula.

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
    cmin : NArray
        Minimum sigmoid value, domain [0.0, 1.0], default 0.0.
    cmax : NArray
        Maximum sigmoid value, domain [0.0, 0.1], default 0.005.
    midpoint : NArray
        Inflection point, domain [0.0, 100.0], default 6.0.
    use_classic : Int
        Flag selecting classic (1) or legacy (0) behaviour.

    Examples
    --------
    >>> # Classic mode (default) — expects 2 source variables
    >>> jr = SigmoidalJansenRit()
    >>> x = np.zeros((2, 5, 1))
    >>> x[0] = 12.0
    >>> jr.pre(x).shape
    (1, 5, 1)

    >>> # Legacy mode — single source variable
    >>> jr_legacy = SigmoidalJansenRit(use_classic=0)
    >>> jr_legacy.pre(np.array([[6.0]])).round(2)
    array([[2.5]])

    Notes
    -----
    - Classic mode collapses 2 source cvars into 1 coupling term.
    - In classic mode ``post()`` multiplies by ``a``; in legacy mode
      ``post()`` is the identity because ``a`` is already in ``pre()``.
    - Set ``use_classic=0`` only for backward compatibility with older
      hybrid scripts.

    See Also
    --------
    Sigmoidal : General sigmoidal coupling
    HyperbolicTangent : Tanh-based coupling
    tvb.simulator.coupling.SigmoidalJansenRit : Classic TVB reference
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
        doc="Maximum firing rate (Hz). Deprecated: only used in legacy mode (use_classic=0).",
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
        doc="Firing threshold (mV). Deprecated: only used in legacy mode (use_classic=0).",
    )

    cmin = t.NArray(
        label=r":math:`c_{min}`",
        default=np.array([0.0]),
        domain=t.Range(lo=-1000.0, hi=1000.0, step=10.0),
        doc="Minimum value of the sigmoid (classic mode).",
    )

    cmax = t.NArray(
        label=r":math:`c_{max}`",
        default=np.array([0.005]),
        domain=t.Range(lo=-1000.0, hi=1000.0, step=10.0),
        doc="Maximum value of the sigmoid (classic mode).",
    )

    midpoint = t.NArray(
        label=r":math:`midpoint`",
        default=np.array([6.0]),
        domain=t.Range(lo=0.0, hi=100.0, step=1.0),
        doc="Inflection point of the sigmoid (classic mode).",
    )

    use_classic = t.Int(
        default=1,
        doc="If 1 (default), use the classic TVB formula with cmin/cmax/midpoint. If 0, use the legacy e0/v0 formula.",
        label="use_classic",
    )

    @property
    def n_cvar_in(self):
        """Number of source state variables required by this coupling."""
        return 2 if self.use_classic else 1

    def _is_classic(self, x_j):
        """Determine whether the input should be treated as classic 2-cvar.

        In classic mode, when ``x_j`` has a leading dimension of 2 (i.e.
        shape ``(2, nnz, modes)``), we compute the difference
        ``x_j[0] - x_j[1]`` before applying the sigmoid.

        When ``use_classic=1`` but the data shape disagrees, a warning
        is emitted and we fall back to the legacy formula so the
        simulation can continue, but the results will NOT match classic
        TVB.
        """
        if not self.use_classic:
            return False
        shape_matches = getattr(x_j, 'ndim', 0) >= 1 and x_j.shape[0] == 2
        if not shape_matches:
            warnings.warn(
                f"{type(self).__name__}: use_classic=1 but input has "
                f"shape {getattr(x_j, 'shape', '?')} (leading dim != 2). "
                f"Falling back to legacy single-cvar formula. "
                f"Results will NOT match classic TVB.",
                RuntimeWarning, stacklevel=2,
            )
        return shape_matches

    def pre(self, x_j, x_i=None, mode=0):
        """Apply pre-synaptic JansenRit sigmoidal coupling.

        Parameters
        ----------
        x_j : ndarray
            Input afferent activity.  In classic mode with two source
            state variables this has shape ``(2, nnz, modes)``; otherwise
            it is a 2-D or 3-D array of membrane potentials.
        x_i : ndarray, optional
            Target (local) activity. Not used by this coupling.

        Returns
        -------
        ndarray
            Transformed activity.  In classic 2-cvar mode the returned
            array has shape ``(1, nnz, modes)``.  In legacy mode the
            shape is the same as ``x_j``.
        """
        if self._is_classic(x_j):
            # x_j shape: (2, nnz, modes)
            diff = x_j[0] - x_j[1]   # shape: (nnz, modes)
            sig = (
                self.cmin[:, None]
                + (self.cmax[:, None] - self.cmin[:, None])
                / (1.0 + np.exp(self.r[:, None] * (self.midpoint[:, None] - diff)))
            )
            # Return shape: (1, nnz, modes)
            return sig[np.newaxis, ...]
        # Legacy mode — single input, a folded into pre()
        return (
            self.a[:, None]
            * (2 * self.e0[:, None])
            / (1.0 + np.exp(self.r[:, None] * (self.v0[:, None] - x_j)))
        )

    def post(self, x, mode=0):
        """Apply post-synaptic coupling.

        Parameters
        ----------
        x : ndarray
            Input summed afferent activity.
        mode : int, default=0
            Number of modes (not used).

        Returns
        -------
        ndarray
            In classic mode: ``a * x``.  In legacy mode: identity.
        """
        if self.use_classic:
            return self.a[:, None] * x
        return x


class PreSigmoidal(Coupling):
    """Pre-summation sigmoidal coupling with static or dynamic threshold.

    Implements a sigmoidal coupling function using ``tanh`` that supports
    both static and dynamic threshold modes, matching the classic TVB
    ``PreSigmoidal`` coupling.

    **Static threshold** (``dynamic=0``) — operates on a single source
    coupling variable:

    .. math::
        y = H \\cdot \\left(Q + \\tanh(G \\cdot (P \\cdot x - \\theta))\\right)

    **Dynamic threshold** (``dynamic=True``) — uses two source state
    variables.  The first carries the afferent signal and the second
    provides a per-node threshold:

    .. math::
        y = H \\cdot \\left(Q + \\tanh(G \\cdot (P \\cdot x_0 - x_1))\\right)

    where ``x_0 = x_j[0]`` and ``x_1 = x_j[1]``.

    When ``globalT=1``, the second state variable is averaged across all
    source nodes to form a single global threshold.

    Parameters
    ----------
    H : float or ndarray, default=0.5
        Amplitude scale.  Matches classic TVB default.
    Q : float or ndarray, default=1.0
        Baseline offset.  Matches classic TVB default.
    G : float or ndarray, default=60.0
        Input sensitivity (gain).  Matches classic TVB default.
    P : float or ndarray, default=1.0
        Input projection.
    theta : float or ndarray, default=0.5
        Activation threshold (static mode).  Matches classic TVB default.
    dynamic : int, default=1
        If 1, use dynamic threshold (two source cvars).  If 0, use
        static threshold (single source cvar).  Matches classic TVB
        default of ``dynamic=True``.
    globalT : int, default=0
        If 1, average the dynamic threshold across all source nodes
        (global threshold).  If 0, use per-node threshold (local).

    Attributes
    ----------
    H : NArray
        Amplitude with domain [0.0, 10.0] and default 0.5.
    Q : NArray
        Baseline offset, domain [-10.0, 10.0], default 1.0.
    G : NArray
        Input sensitivity, domain [0.01, 100.0], default 60.0.
    P : NArray
        Input projection, domain [-10.0, 10.0], default 1.0.
    theta : NArray
        Threshold, domain [-100.0, 100.0], default 0.5.
    dynamic : Int
        Whether threshold is dynamic (default 1).
    globalT : Int
        Whether threshold is global (default 0).

    Examples
    --------
    >>> # Static threshold (single source cvar)
    >>> ps = PreSigmoidal(dynamic=0)
    >>> ps.pre(np.array([0.5])).round(2)
    array([0.5])

    >>> # Dynamic threshold (two source cvars)
    >>> ps_dyn = PreSigmoidal(dynamic=True)
    >>> x = np.zeros((2, 5, 1)); x[0] = 1.0
    >>> ps_dyn.pre(x).shape
    (1, 5, 1)

    Notes
    -----
    - In dynamic mode, ``pre()`` collapses 2 source cvars into 1 output
      (same pattern as ``SigmoidalJansenRit``).
    - The ``c_1`` diagonal self-connection term from classic TVB is not
      yet implemented in the projection pipeline; see the TODO in
      ``post()``.

    See Also
    --------
    SigmoidalJansenRit : Another 2-cvar pre-summation coupling
    Sigmoidal : Post-synaptic sigmoidal coupling
    HyperbolicTangent : Simpler tanh-based coupling
    tvb.simulator.coupling.PreSigmoidal : Classic TVB reference
    """

    H = t.NArray(
        label=r":math:`H`",
        default=np.array([0.5]),
        domain=t.Range(lo=0.0, hi=10.0, step=0.01),
        doc="Amplitude scale.  Classic TVB default is 0.5.",
    )

    Q = t.NArray(
        label=r":math:`Q`",
        default=np.array([1.0]),
        domain=t.Range(lo=-10.0, hi=10.0, step=0.1),
        doc="Baseline offset.  Classic TVB default is 1.0.",
    )

    G = t.NArray(
        label=r":math:`G`",
        default=np.array([60.0]),
        domain=t.Range(lo=0.01, hi=100.0, step=0.1),
        doc="Input sensitivity (gain).  Classic TVB default is 60.0.",
    )

    P = t.NArray(
        label=r":math:`P`",
        default=np.array([1.0]),
        domain=t.Range(lo=-10.0, hi=10.0, step=0.1),
        doc="Input projection.",
    )

    theta = t.NArray(
        label=r":math:`\theta`",
        default=np.array([0.5]),
        domain=t.Range(lo=-100.0, hi=100.0, step=1.0),
        doc="Activation threshold (static mode).  Classic TVB default is 0.5.",
    )

    dynamic = t.Attr(
        field_type=bool,
        label="dynamic",
        default=True,
        doc="If True (default), use dynamic threshold (two source cvars). "
            "If False, use static threshold (single source cvar). "
            "Matches classic TVB default of dynamic=True.",
    )

    globalT = t.Attr(
        field_type=bool,
        label=r":math:`global_{\theta}`",
        default=False,
        doc="If True, average dynamic threshold across source nodes "
            "(global threshold). If False, use per-node threshold (local).",
    )

    @property
    def n_cvar_in(self):
        """Number of source state variables required by this coupling."""
        return 2 if self.dynamic else 1

    def _is_dynamic(self, x_j):
        """Determine whether the input should be treated as dynamic 2-cvar.

        In dynamic mode, when ``x_j`` has a leading dimension of 2
        (i.e. shape ``(2, nnz, modes)``), we compute
        ``P * x_j[0] - x_j[1]`` before applying tanh.

        When ``dynamic=1`` but the data shape disagrees, a warning
        is emitted and we fall back to the static formula so the
        simulation can continue, but the results will NOT match classic
        TVB.
        """
        if not self.dynamic:
            return False
        shape_matches = getattr(x_j, 'ndim', 0) >= 1 and x_j.shape[0] == 2
        if not shape_matches:
            warnings.warn(
                f"{type(self).__name__}: dynamic is {self.dynamic} but input has "
                f"shape {getattr(x_j, 'shape', '?')} (leading dim != 2). "
                f"Falling back to static single-cvar formula. "
                f"Results will NOT match classic TVB.",
                RuntimeWarning, stacklevel=2,
            )
        return shape_matches

    def pre(self, x_j, x_i=None, mode=0):
        """Apply pre-synaptic sigmoidal coupling (PreSigmoidal).

        Parameters
        ----------
        x_j : ndarray
            Input afferent activity.  In dynamic mode with two source
            state variables this has shape ``(2, nnz, modes)``; in static
            mode it is a 2-D or 3-D array.
        x_i : ndarray, optional
            Target (local) activity. Not used by this coupling.

        Returns
        -------
        ndarray
            Transformed activity.  In dynamic 2-cvar mode the returned
            array has shape ``(1, nnz, modes)``.  In static mode the
            shape matches ``x_j``.
        """
        if self._is_dynamic(x_j):
            # Dynamic threshold: use second source cvar as threshold
            # x_j shape: (2, nnz, modes)
            # x_j[0] = afferent signal, x_j[1] = dynamic threshold
            if self.globalT:
                # Global threshold: average x_j[1] across all source nodes
                threshold = x_j[1].mean(axis=0, keepdims=True)
            else:
                threshold = x_j[1]
            arg = self.P[:, None] * x_j[0] - threshold
            result = self.H[:, None] * (
                self.Q[:, None] + np.tanh(self.G[:, None] * arg)
            )
            # Return shape: (1, nnz, modes)
            return result[np.newaxis, ...]

        # Static threshold mode
        return self.H[:, None] * (
            self.Q[:, None]
            + np.tanh(
                self.G[:, None] * (self.P[:, None] * x_j - self.theta[:, None])
            )
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

        TODO
        ----
        Classic TVB PreSigmoidal in dynamic mode also returns a ``c_1``
        term (diagonal self-connection ``A_j[i,i]``) that provides a
        local feedback signal.  The hybrid projection pipeline currently
        has no mechanism for injecting a per-node diagonal term back
        into the coupling array.  This will require extending
        ``BaseProjection.apply()`` with a post-hook for diagonal
        contributions.
        """
        return x

