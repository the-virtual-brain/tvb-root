# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)

"""
This module defines the common imports and abstract base class for model definitions.

"""
import abc
import numpy
import typing
from tvb.basic.neotraits.api import HasTraits




class Model(HasTraits):
    """
    Defines the abstract base class for neuronal models.

    """

    state_variables = ()  # type: typing.Tuple[str]
    non_integrated_variables = None  # type: typing.Tuple[str]
    variables_of_interest = ()
    _nvar = None   # todo make this a prop len(state_variables)
    _nintvar = _nvar
    number_of_modes = 1
    cvar = None
    stvar = None
    state_variable_boundaries = None
    state_variable_mask = None

    def _build_observer(self):
        template = ("def observe(state):\n"
                    "    {svars} = state\n"
                    "    return numpy.array([{voi_names}])")
        svars = ','.join(self.state_variables)
        if len(self.state_variables) == 1:
            svars += ','
        code = template.format(
            svars=svars,
            voi_names=','.join(self.variables_of_interest)
        )
        namespace = {'numpy': numpy}
        self.log.debug('building observer with code:\n%s', code)
        exec(code, namespace)
        self.observe = namespace['observe']
        self.observe.code = code

    def _setup_sv_boundaries(self):
        for sv, sv_bounds in self.state_variable_boundaries.items():
            try:
                # ...the boundaries correspond to model's state variables,
                self.state_variables.index(sv)
            except IndexError:
                raise ValueError("Non-existent state variable in bounds %s!" % str(sv_bounds))
            infs = [-numpy.inf, numpy.inf]
            minmax = [numpy.finfo("double").min, numpy.finfo("double").max]
            for i_bound, (sv_bound, inf, default) in enumerate(zip(sv_bounds, infs, minmax)):
                if sv_bound is None or sv_bound == inf:
                    sv_bounds[i_bound] = default
            self.state_variable_boundaries[sv] = sv_bounds.astype("float64")

    def configure(self):
        "Configure base model."
        for req_attr in 'nvar number_of_modes cvar'.split():
            assert hasattr(self, req_attr)
        if self.stvar is None:
            self.stvar = self.cvar.copy()
        super(Model, self).configure()
        # Make sure that if there are any state variable boundaries, ...
        if isinstance(self.state_variable_boundaries, dict):
            self._setup_sv_boundaries()
        elif self.state_variable_boundaries is not None:
            self.state_variable_boundaries = None
            Warning("Non dict model state variable boundaries ignored!: %s" % str(self.state_variable_boundaries))
        self.state_variable_mask = numpy.array([True] * self.nvar)
        if self.non_integrated_variables is not None:
            for var in self.non_integrated_variables:
                self.state_variable_mask[self.state_variables.index(var)] = False
        self._nintvar = numpy.sum(self.state_variable_mask)
        self.update_derived_parameters()
        self._build_observer()

    @property
    def nvar(self):
        """ The number of state variables in this model. """
        return self._nvar

    @property
    def nintvar(self):
        """ The number of integrated state variables in this model. """
        return self._nintvar

    @property
    def nnonintvar(self):
        """ The number of non integrated state variables in this model. """
        return self._nvar - self._nintvar

    @property
    def has_nonint_vars(self):
        """ Flag to determine if there are any non integrated state variables. """
        return self.nnonintvar > 0

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

    def initial_for_simulator(self, integrator, shape):
        "Generate initial conditions with integrator and shape."
        rng = integrator.noise.random_stream if hasattr(integrator, 'noise') else numpy.random
        dt = integrator.dt
        return self.initial(dt, shape, rng)

    @abc.abstractmethod
    def dfun(self, state_variables, coupling, local_coupling=0.0):
        """
        Defines the dynamic equations. That is, the derivative of the
        state-variables given their current state ``state_variables``, the past
        state from other regions of the brain currently arriving ``coupling``,
        and the current state of the "local" neighbourhood ``local_coupling``.

        """


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
        if self.number_of_modes == 3:
            coupling = numpy.tile(coupling, (1,1,3))
        for i in range(n_step):
            state += dt * self.dfun(state, coupling)
            if i % n_skip == 0:
                out.append(state.copy())

        return numpy.r_[0:dt * n_step:1j * len(out)], numpy.array(out)

    @property
    def spatial_param_reshape(self):
        "Returns reshape argument for a spatialized parameter."
        return -1, 1

    def _spatialize_model_parameters(self, sim):
        # Make sure spatialised model parameters have the right shape (number_of_nodes, 1)
        # todo: this exclusion list is fragile, consider excluding declarative attrs that are not arrays
        excluded_params = ("state_variable_range", "state_variable_boundaries", "variables_of_interest",
                           "noise", "psi_table", "nerf_table", "gid", "state_variable_dfuns",
                           "parameter_names", "coupling_terms")
        spatial_reshape = self.spatial_param_reshape
        for param in type(self).declarative_attrs:
            if param in excluded_params:
                continue
            region_parameters = getattr(self, param)
            self._map_roi_param_to_surface(sim, param, region_parameters, spatial_reshape)
            self._reshape_model_param_for_modes(sim, param, spatial_reshape)

    def _reshape_model_param_for_modes(self, sim, param, spatial_reshape):
        region_parameters = getattr(self, param)
        if region_parameters.size == sim.number_of_nodes:
            new_parameters = region_parameters.reshape(spatial_reshape)
            setattr(self, param, new_parameters)

    def _map_roi_param_to_surface(self, sim, param, region_parameters, spatial_reshape):
        if sim.surface is not None:
            if region_parameters.size == sim.connectivity.number_of_regions:
                new_parameters = region_parameters[sim.surface.region_mapping].reshape(spatial_reshape)
                setattr(self, param, new_parameters)

    def update_state_variables_before_integration(self, state_variables, coupling, local_coupling=0.0, stimulus=0.0):
        return state_variables

    def update_state_variables_after_integration(self, state_variables):
        return state_variables

    @property
    def spatial_parameter_names(self):
        return [_ for _ in self.parameter_names if getattr(self, _).size != 1]
    
    @property
    def global_parameter_names(self):
        return [_ for _ in self.parameter_names if getattr(self, _).size == 1]

    @property
    def spatial_parameter_matrix(self):
        names = self.spatial_parameter_names
        matrix = numpy.array([getattr(self,_).reshape((-1,)) for _ in names])
        return matrix
    


class ModelNumbaDfun(Model):
    "Base model for Numba-implemented dfuns."

    @property
    def spatial_param_reshape(self):
        return -1,
