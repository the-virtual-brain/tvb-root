# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)

"""
This module defines the common imports and abstract base class for model definitions.

"""

import numpy
import numexpr
from scipy.integrate import trapz as scipy_integrate_trapz
from scipy.stats import norm as scipy_stats_norm
from tvb.simulator.common import get_logger
import tvb.datatypes.arrays as arrays
import tvb.basic.traits.core as core
import tvb.basic.traits.types_basic as basic
import tvb.simulator.noise as noise_module


LOG = get_logger(__name__)


# NOTE: For UI convenience set the step in all parameters ranges such that there
#      are approximately 10 steps from lo to hi...



class Model(core.Type):
    """
    Defines the abstract class for neuronal models.

    .. automethod:: Model.__init__
    .. automethod:: Model.dfun
    .. automethod:: Model.update_derived_parameters

    """
    _base_classes = ['Model', ]
    # NOTE: the parameters that are contained in the following list will be
    # editable from the ui in an visual manner
    ui_configurable_parameters = []

    state_variables = []
    variables_of_interest = []

    def __init__(self, **kwargs):
        """
        Initialize the model with parameters as keywords arguments, a sensible
        default parameter set should be provided via the trait mechanism.

        """
        super(Model, self).__init__(**kwargs)
        LOG.debug(str(kwargs))
        self._nvar = None
        self.number_of_modes = 1

    def _build_observer(self):
        template = ("def observe(state):\n"
                            "    {svars} = state\n"
                            "    return numpy.array([{voi_names}])")
        svars = ','.join(self.state_variables)
        if len(self.state_variables) == 1:
            svars += ','
        code = template.format(
            svars = svars,
            voi_names = ','.join(self.variables_of_interest)
        )
        namespace = {'numpy': numpy}
        LOG.debug('building observer with code:\n%s', code)
        exec code in namespace
        self.observe = namespace['observe']
        self.observe.code = code

    def configure(self):
        "Configure base model."
        super(Model, self).configure()
        self.update_derived_parameters()
        self._build_observer()

    @property
    def nvar(self):
        """ The number of state variables in this model. """
        return self._nvar

#    @property
#    def distal_coupling(self):
#        """ Heterogeneous coupling given by the connectivity matrix"""
#        return self._distal_coupling
#
#    @property
#    def local_coupling(self):
#        """ Homogeneous connectivity given by a local connectivity kernel"""
#        return self._local_coupling
#
#    @property
#    def internal_coupling(self):
#        """ Internal connectivity between neural masses of a model"""
#        return self._internal_coupling
#
#    @property
#    def state_coupling(self):
#        """ State operator: A matrix where each elemeent represents
#        a parameter of the model """
#        return self._state_coupling

    def update_derived_parameters(self):
        """
        When needed, this should be a method for calculating parameters that are
        calculated based on paramaters directly set by the caller. For example,
        see, ReducedSetFitzHughNagumo. When not needed, this pass simplifies
        code that updates an arbitrary models parameters -- ie, this can be
        safely called on any model, whether it's used or not.
        """
        pass

    def initial(self, dt, history_shape, rng=numpy.random):
        """Generates uniformly distributed initial conditions,
        bounded by the state variable limits defined by the model.
        """
        nt, nvar, nnode, nmode = history_shape
        ic = numpy.empty(history_shape)
        svr = self.state_variable_range
        sv = self.state_variables
        block = nt, nnode, nmode
        for i, (lo, hi) in enumerate([svr[sv[i]] for i in range(nvar)]):
            ic[:, i] = rng.uniform(low=lo, high=hi, size=block)
        return ic

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        """
        Defines the dynamic equations. That is, the derivative of the
        state-variables given their current state ``state_variables``, the past
        state from other regions of the brain currently arriving ``coupling``,
        and the current state of the "local" neighbourhood ``local_coupling``.

        """
        pass

    # TODO refactor as a NodeSimulator class
    def stationary_trajectory(self,
                              coupling=numpy.array([[0.0]]),
                              initial_conditions=None,
                              n_step=1000, n_skip=10, dt=2 ** -4,
                              map=map):
        """
        Computes the state space trajectory of a single mass model system
        where coupling is static, with a deterministic Euler method.

        Models expect coupling of shape (n_cvar, n_node), so if this method
        is called with coupling (:, n_cvar, n_ode), it will compute a
        stationary trajectory for each coupling[i, ...]

        """

        if coupling.ndim == 3:
            def mapped(coupling_i):
                kwargs = dict(initial_conditions=initial_conditions,
                              n_step=n_step, n_skip=n_skip, dt=dt)
                ts, ys = self.stationary_trajectory(coupling_i, **kwargs)
                return ts, ys

            out = [ys for ts, ys in map(mapped, coupling)]
            return ts, numpy.array(out)

        state = initial_conditions
        if type(state) == type(None):
            n_mode = self.number_of_modes
            state = numpy.empty((self.nvar, n_mode))
            for i, (lo, hi) in enumerate(self.state_variable_range.values()):
                state[i, :] = numpy.random.uniform(size=n_mode) * (hi - lo) / 2. + lo
        state = state[:, numpy.newaxis]

        out = [state.copy()]
        for i in xrange(n_step):
            state += dt * self.dfun(state, coupling)
            if i % n_skip == 0:
                out.append(state.copy())

        return numpy.r_[0:dt * n_step:1j * len(out)], numpy.array(out)

# TODO: both coupling/connectivity and local_coupling should be generalised to
#      couplings and local_couplings that can be set to independantly on each
#      state variable of a model at run time. Current functionality would then
#      come via defaults that turn off the coupling on some state variables.
