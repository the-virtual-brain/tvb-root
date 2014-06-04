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
#
#

"""
A collection of neuronal dynamics models.

Specific models inherit from the abstract class Model, which in turn inherits
from the class Trait from the tvb.basic.traits module.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>
.. moduleauthor:: Gaurav Malhotra <Gaurav@tvb.invalid>
.. moduleauthor:: Marmaduke Woodman <mw@eml.cc>

"""

import inspect
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
    _base_classes = ['Model', "JRFast"]
    # NOTE: the parameters that are contained in the following list will be
    # editable from the ui in an visual manner
    ui_configurable_parameters = []

    noise = noise_module.Noise(
        fixed_type=True,  # Can only be Noise in UI, and not a subclass of Noise
        label="Initial Conditions Noise",
        default=noise_module.Noise,
        doc="""A noise source used to provide random initial conditions when
        no, or insufficient, explicit initial conditions are provided.
        NOTE: Dispersion is computed based on ``state_variable_range``.""",
        order=42 ** 42)  # to be displayed last

    def __init__(self, **kwargs):
        """
        Initialize the model with parameters as keywords arguments, a sensible
        default parameter set should be provided via the trait mechanism.

        """
        super(Model, self).__init__(**kwargs)
        LOG.debug(str(kwargs))

        # self._state_variables = None
        self._nvar = None
        self.number_of_modes = 1  # NOTE: Models without modes can ignore this.

    def configure(self):
        """  """
        super(Model, self).configure()
        self.update_derived_parameters()

    def __repr__(self):
        """ A formal, executable, representation of a Model object. """
        class_name = self.__class__.__name__
        traited_kwargs = self.trait.keys()
        formal = class_name + "(" + "=%s, ".join(traited_kwargs) + "=%s)"
        return formal % eval("(self." + ", self.".join(traited_kwargs) + ")")

    def __str__(self):
        """ An informal, human readable, representation of a Model object. """
        # NOTE: We don't explicitly list kwds cause some models have too many.
        informal = self.__class__.__name__ + "(**kwargs)"
        return informal

    @property
    def state_variables(self):
        """ A list of the state variables in this model. """
        return self.trait['variables_of_interest'].trait.options

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

    def configure_initial(self, dt, shape):
        """Docs..."""
        # Configure the models noise stream
        if self.noise.ntau > 0.0:
            self.noise.configure_coloured(dt, shape)
        else:
            # The same applies for truncated rv
            self.noise.configure_white(dt, shape)

    def initial(self, dt, history_shape):
        """
        Defines a set of sensible initial conditions, it is only expected that
        these initial conditions be guaranteed to be sensible for the default
        parameter set.

        It is often seen that initial conditions or initial history are filled
        following ic = nsig * noise + loc, where noise is a stream of random
        numbers drawn from a uniform or normal distribution.

        Here, the noise is integrated, that is the initial conditions are
        described by a diffusion process.

        """
        # NOTE: The following max_history_length factor is set such that it
        # considers the worst case possible for the history array, that is, low
        # conduction speed and the longest white matter fibre. With this, plus
        # drawing the random stream of numbers from a truncated normal
        # distribution, we expect to bound the diffussion process used to set
        # the initial history to the state variable ranges. (even for the case
        # of rate based models)

        # Example:
        #        + longest fiber: 200 mm (Ref:)
        #        + min conduction speed: 3 mm/ms (Ref: http://www.scholarpedia.org/article/Axonal_conduction_delays)
        #          corpus callosum in the macaque monkey.
        #        + max time delay: tau = longest fibre / speed \approx 67 ms
        #        + let's hope for the best...

        max_history_length = 67.  # ms
        # TODO: There is an issue of allignment when the current implementation
        #      is used to pad out explicit inital conditions that aren't as long
        #      as the required history...
        # TODO: We still ideally want to support spatial colour for surfaces --
        #      though this will probably have to be done at the Simulator level
        #      with a linear, strictly-stable, spatially invariant filter
        #      defined on the mesh surface... and in a way that won't change the
        #      temporal correllation structure... and I'm also assuming that the
        #      temporally coloured noise hasn't un-whitened the spatial noise
        #      distribution to start with... [ie, spatial colour is a longer
        #      term problem to solve...])
        #TODO: Ideally we'd like to have a independent random stream for each node.
        #      Currently, if the number of nodes is different we'll get the
        #      slightly different values further in the history array.

        initial_conditions = numpy.zeros(history_shape)
        tpts = history_shape[0]
        nvar = history_shape[1]
        history_shape = history_shape[2:]

        self.configure_initial(dt, history_shape)
        noise = numpy.zeros((tpts, nvar) + history_shape)
        nsig = numpy.zeros(nvar)
        loc = numpy.zeros(nvar)

        for tpt in range(tpts):
            for var in range(nvar):
                loc[var] = self.state_variable_range[self.state_variables[var]].mean()
                nsig[var] = (self.state_variable_range[self.state_variables[var]][1] -
                             self.state_variable_range[self.state_variables[var]][0]) / 2.0
                nsig[var] = nsig[var] / max_history_length

                # define lower und upper bounds to truncate the random variates to the sv range.
                lo = self.state_variable_range[self.state_variables[var]][0]
                hi = self.state_variable_range[self.state_variables[var]][1]
                lo_bound, up_bound = (lo - loc[var]) / nsig[var], (hi - loc[var]) / nsig[var]

                noise[tpt, var, :] = self.noise.generate(history_shape, truncate=True,
                                                         lo=lo_bound, hi=up_bound)

        for var in range(nvar):
                # TODO: Hackery, validate me...-noise.mean(axis=0) ... self.noise.nsig
            initial_conditions[:, var, :] = numpy.sqrt(2.0 * nsig[var]) * numpy.cumsum(noise[:, var, :],axis=0) + loc[var]

        return initial_conditions

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        """
        Defines the dynamic equations. That is, the derivative of the
        state-variables given their current state ``state_variables``, the past
        state from other regions of the brain currently arriving ``coupling``,
        and the current state of the "local" neighbourhood ``local_coupling``.

        """
        pass

    def stationary_trajectory(self,
                              coupling=numpy.array([[0.0]]),
                              initial_conditions=None,
                              n_step=1000, n_skip=10, dt=2 ** -4,
                              map=map):
        """
        Computes the state space trajectory of a single mass model system
        where coupling is static, with a deteministic Euler method.

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


class model_device_info(object):
    """
    A model_device_info is a class that provides enough additional information,
    on request, in order to run the owner on a native code device, e.g. a CUDA
    GPU.

    All such device_data classes will need to furnish their corresponding
    global values as well as corresponding array arguments for update(.),
    already initialized based on the instance object that the device_data
    instance is attached to.

    Such things are built on conventions: a class should declare both
    a device_data attribute as well as class specific information, that is
    not programmatically available via traits. In general, each of the
    abstract classes (Model, Noise, Integrator & Coupling) have n_xxpr
    and xxpr, but they don't have symmetric properties, so it will be done
    case by case.

    """

    def __init__(self, pars=[], kernel=""):
        """
        Make a model_device_info instance with a list of trait attributes
        corresponding to the mass model parameters, and a kernel which is
        a string of code required to implement the model's dfun on device.

        The order of the parameters MUST identify the order of the parameters
        in the array returned by the mmpr property and thus the order in
        which the parameters can be read via P(i) macro in the kernel code.

        Don't forget, the kernel code defines per node, per thread update.

        Unfortunately, I did not forsee an obvious way to nicely handle modes
        in the reduced models. Kernel code must handle all variables as scalars.

        """

        self._pars = pars
        self._kernel = kernel

    @property
    def n_mmpr(self):
        if self.n_mode == 1:
            return len(self._pars)
        else:
            # have to figure out default par size
            new = self.inst.__class__()
            new.configure()
            count = 0
            for k in new.device_info._pars:
                att = getattr(new, k if type(k) == str else k.trait.name)
                count += att.size
            return count

    @property
    def mmpr(self):  # build mmpr array from known inst traits

        nm = self.inst.number_of_modes
        if nm == 1:
            pars = []
            for par in self._pars:
                name = par if type(par) == str else par.trait.name
                pars.append(getattr(self.inst, name))
            return numpy.vstack(pars).T  # so self.mmpr[0] yield all pars for node 0
        else:
            pars = []
            new = self.inst.__class__()
            new.configure()
            for k in self._pars:
                name = k if type(k) == str else k.trait.name
                att, attnew = getattr(self.inst, name), getattr(new, name)
                if att.size == attnew.size:
                    pars.append(att.flat[:])
                else:
                    msg = """%r.%r.mmpr requires that modal models be initialized with
                    spatially homogeneous pars, i.e. array([0.0]) not
                    array([0.0, 0.1, ...]). Please set the per node parameters
                    by hand, after configuring the device handler. Sorry."""
                    raise AttributeError(msg % (self.inst, self))
            return numpy.hstack(pars)[numpy.newaxis, :]

    @property
    def n_mode(self):
        return getattr(self.inst, 'number_of_modes', 1)

    @property
    def n_svar(self):
        return self.inst._nvar * self.n_mode

    @property
    def n_cvar(self):
        return self.inst.cvar.size * self.n_mode

    @property
    def cvar(self):
        assert self.inst.cvar.ndim == 1
        if self.n_mode == 1:
            return self.inst.cvar.astype(numpy.int32)
        else:
            return numpy.tile(self.inst.cvar, (self.n_mode, 1)).flat[:].astype(numpy.int32)

    @property
    def kernel(self):
        return self._kernel

    def __get__(self, inst, ownr):
        if inst:
            self.inst = inst
            return self
        else:
            return None

    def __set__(self, inst, val):
        raise AttributeError('%r is not to be set' % (self,))


class WilsonCowan(Model):
    r"""
    **References**:

    .. [WC_1972] Wilson, H.R. and Cowan, J.D. *Excitatory and inhibitory
        interactions in localized populations of model neurons*, Biophysical
        journal, 12: 1-24, 1972.
    .. [WC_1973] Wilson, H.R. and Cowan, J.D  *A Mathematical Theory of the
        Functional Dynamics of Cortical and Thalamic Nervous Tissue*

    .. [D_2011] Daffertshofer, A. and van Wijk, B. *On the influence of
        amplitude on the connectivity between phases*
        Frontiers in Neuroinformatics, July, 2011

    Used Eqns 11 and 12 from [WC_1972]_ in ``dfun``.  P and Q represent external
    inputs, which when exploring the phase portrait of the local model are set
    to constant values. However in the case of a full network, P and Q are the
    entry point to our long range and local couplings, that is, the  activity
    from all other nodes is the external input to the local population.

    The default parameters are taken from figure 4 of [WC_1972]_, pag. 10

    In [WC_1973]_ they present a model of neural tissue on the pial surface is.
    See Fig. 1 in page 58. The following local couplings (lateral interactions)
    occur given a region i and a region j:

      E_i-> E_j
      E_i-> I_j
      I_i-> I_j
      I_i-> E_j


    +---------------------------+
    |          Table 1          |
    +--------------+------------+
    |                           |
    |  SanzLeonetAl,   2014     |
    +--------------+------------+
    |Parameter     |  Value     |
    +==============+============+
    | k_e, k_i     |    1.00    |
    +--------------+------------+
    | r_e, r_i     |    0.00    |
    +--------------+------------+
    | tau_e, tau_i |    10.0    |
    +--------------+------------+
    | c_1          |    10.0    |
    +--------------+------------+
    | c_2          |    6.0     |
    +--------------+------------+
    | c_3          |    1.0     |
    +--------------+------------+
    | c_4          |    1.0     |
    +--------------+------------+
    | a_e, a_i     |    1.0     |
    +--------------+------------+
    | b_e, b_i     |    0.0     |
    +--------------+------------+
    | theta_e      |    2.0     |
    +--------------+------------+
    | theta_i      |    3.5     |
    +--------------+------------+
    | alpha_e      |    1.2     |
    +--------------+------------+
    | alpha_i      |    2.0     |
    +--------------+------------+
    | P            |    0.5     |
    +--------------+------------+
    | Q            |    0       |
    +--------------+------------+
    | c_e, c_i     |    1.0     |
    +--------------+------------+
    | alpha_e      |    1.2     |
    +--------------+------------+
    | alpha_i      |    2.0     |
    +--------------+------------+
    |                           |
    |  frequency peak at 20  Hz |
    |                           |
    +---------------------------+


    The parameters in Table 1 reproduce Figure A1 in  [D_2011]_
    but set the limit cycle frequency to a sensible value (eg, 20Hz).

    Model bifurcation parameters:
        * :math:`c_1`
        * :math:`P`



    The models (:math:`E`, :math:`I`) phase-plane, including a representation of
    the vector field as well as its nullclines, using default parameters, can be
    seen below:

        .. _phase-plane-WC:
        .. figure :: img/WilsonCowan_01_mode_0_pplane.svg
            :alt: Wilson-Cowan phase plane (E, I)

            The (:math:`E`, :math:`I`) phase-plane for the Wilson-Cowan model.

    .. automethod:: WilsonCowan.__init__

    The general formulation for the \textit{\textbf{Wilson-Cowan}} model as a
    dynamical unit at a node $k$ in a BNM with $l$ nodes reads:

    .. math::
            \dot{E}_k &= \dfrac{1}{\tau_e} (-E_k  + (k_e - r_e E_k) \mathcal{S}_e (\alpha_e \left( c_{ee} E_k - c_{ei} I_k  + P_k - \theta_e + \mathbf{\Gamma}(E_k, E_j, u_{kj}) + W_{\zeta}\cdot E_j + W_{\zeta}\cdot I_j\right) ))\\
            \dot{I}_k &= \dfrac{1}{\tau_i} (-I_k  + (k_i - r_i I_k) \mathcal{S}_i (\alpha_i \left( c_{ie} E_k - c_{ee} I_k  + Q_k - \theta_i + \mathbf{\Gamma}(E_k, E_j, u_{kj}) + W_{\zeta}\cdot E_j + W_{\zeta}\cdot I_j\right) )),

    """
    _ui_name = "Wilson-Cowan"
    ui_configurable_parameters = ['c_ee', 'c_ei', 'c_ie', 'c_ii', 'tau_e', 'tau_i',
                                  'a_e', 'b_e', 'c_e', 'a_i', 'b_i', 'c_i', 'r_e',
                                  'r_i', 'k_e', 'k_i', 'P', 'Q', 'theta_e', 'theta_i',
                                  'alpha_e', 'alpha_i']

    # Define traited attributes for this model, these represent possible kwargs.
    c_ee = arrays.FloatArray(
        label=":math:`c_{ee}`",
        default=numpy.array([12.0]),
        range=basic.Range(lo=11.0, hi=16.0, step=0.01),
        doc="""Excitatory to excitatory  coupling coefficient""",
        order=1)

    c_ie = arrays.FloatArray(
        label=":math:`c_{ei}`",
        default=numpy.array([4.0]),
        range=basic.Range(lo=2.0, hi=15.0, step=0.01),
        doc="""Inhibitory to excitatory coupling coefficient""",
        order=2)

    c_ei = arrays.FloatArray(
        label=":math:`c_{ie}`",
        default=numpy.array([13.0]),
        range=basic.Range(lo=2.0, hi=22.0, step=0.01),
        doc="""Excitatory to inhibitory coupling coefficient.""",
        order=3)

    c_ii = arrays.FloatArray(
        label=":math:`c_{ii}`",
        default=numpy.array([11.0]),
        range=basic.Range(lo=2.0, hi=15.0, step=0.01),
        doc="""Inhibitory to inhibitory coupling coefficient.""",
        order=4)

    tau_e = arrays.FloatArray(
        label=r":math:`\tau_e`",
        default=numpy.array([10.0]),
        range=basic.Range(lo=0.0, hi=150.0, step=0.01),
        doc="""Excitatory population, membrane time-constant [ms]""",
        order=5)

    tau_i = arrays.FloatArray(
        label=r":math:`\tau_i`",
        default=numpy.array([10.0]),
        range=basic.Range(lo=0.0, hi=150.0, step=0.01),
        doc="""Inhibitory population, membrane time-constant [ms]""",
        order=6)

    a_e = arrays.FloatArray(
        label=":math:`a_e`",
        default=numpy.array([1.2]),
        range=basic.Range(lo=0.0, hi=1.4, step=0.01),
        doc="""The slope parameter for the excitatory response function""",
        order=7)

    b_e = arrays.FloatArray(
        label=":math:`b_e`",
        default=numpy.array([2.8]),
        range=basic.Range(lo=1.4, hi=6.0, step=0.01),
        doc="""Position of the maximum slope of the excitatory sigmoid function""",
        order=8)

    c_e = arrays.FloatArray(
        label=":math:`c_e`",
        default=numpy.array([1.0]),
        range=basic.Range(lo=1.0, hi=20.0, step=1.0),
        doc="""The amplitude parameter for the excitatory response function""",
        order=9)

    theta_e = arrays.FloatArray(
        label=r":math:`\theta_e`",
        default=numpy.array([0.0]),
        range=basic.Range(lo=0.0, hi=60., step=0.01),
        doc="""Excitatory threshold""",
        order=10)

    a_i = arrays.FloatArray(
        label=":math:`a_i`",
        default=numpy.array([1.0]),
        range=basic.Range(lo=0.0, hi=2.0, step=0.01),
        doc="""The slope parameter for the inhibitory response function""",
        order=11)

    b_i = arrays.FloatArray(
        label=r":math:`b_i`",
        default=numpy.array([4.0]),
        range=basic.Range(lo=2.0, hi=6.0, step=0.01),
        doc="""Position of the maximum slope of a sigmoid function [in
        threshold units]""",
        order=12)

    theta_i = arrays.FloatArray(
        label=r":math:`\theta_i`",
        default=numpy.array([0.0]),
        range=basic.Range(lo=0.0, hi=60.0, step=0.01),
        doc="""Inhibitory threshold""",
        order=13)

    c_i = arrays.FloatArray(
        label=":math:`c_i`",
        default=numpy.array([1.0]),
        range=basic.Range(lo=1.0, hi=20.0, step=1.0),
        doc="""The amplitude parameter for the inhibitory response function""",
        order=14)

    r_e = arrays.FloatArray(
        label=":math:`r_e`",
        default=numpy.array([1.0]),
        range=basic.Range(lo=0.5, hi=2.0, step=0.01),
        doc="""Excitatory refractory period""",
        order=15)

    r_i = arrays.FloatArray(
        label=":math:`r_i`",
        default=numpy.array([1.0]),
        range=basic.Range(lo=0.5, hi=2.0, step=0.01),
        doc="""Inhibitory refractory period""",
        order=16)

    k_e = arrays.FloatArray(
        label=":math:`k_e`",
        default=numpy.array([1.0]),
        range=basic.Range(lo=0.5, hi=2.0, step=0.01),
        doc="""Maximum value of the excitatory response function""",
        order=17)

    k_i = arrays.FloatArray(
        label=":math:`k_i`",
        default=numpy.array([1.0]),
        range=basic.Range(lo=0.0, hi=2.0, step=0.01),
        doc="""Maximum value of the inhibitory response function""",
        order=18)

    P = arrays.FloatArray(
        label=":math:`P`",
        default=numpy.array([0.0]),
        range=basic.Range(lo=0.0, hi=20.0, step=0.01),
        doc="""External stimulus to the excitatory population.
        Constant intensity.Entry point for coupling.""",
        order=19)

    Q = arrays.FloatArray(
        label=":math:`Q`",
        default=numpy.array([0.0]),
        range=basic.Range(lo=0.0, hi=20.0, step=0.01),
        doc="""External stimulus to the inhibitory population.
        Constant intensity.Entry point for coupling.""",
        order=20)

    alpha_e = arrays.FloatArray(
        label=r":math:`\alpha_e`",
        default=numpy.array([1.0]),
        range=basic.Range(lo=0.0, hi=20.0, step=0.01),
        doc="""External stimulus to the excitatory population.
        Constant intensity.Entry point for coupling.""",
        order=21)

    alpha_i = arrays.FloatArray(
        label=r":math:`\alpha_i`",
        default=numpy.array([1.0]),
        range=basic.Range(lo=0.0, hi=20.0, step=0.01),
        doc="""External stimulus to the inhibitory population.
        Constant intensity.Entry point for coupling.""",
        order=22)

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = basic.Dict(
        label="State Variable ranges [lo, hi]",
        default={"E": numpy.array([0.0, 1.0]),
                 "I": numpy.array([0.0, 1.0])},
        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current
        parameters, it is used as a mechanism for bounding random inital
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""",
        order=23)

    #    variables_of_interest = arrays.IntegerArray(
    #        label = "Variables watched by Monitors",
    #        range = basic.Range(lo = 0.0, hi = 2.0, step = 1.0),
    #        default = numpy.array([0], dtype=numpy.int32),
    #        doc = """This represents the default state-variables of this Model to be
    #        monitored. It can be overridden for each Monitor if desired. The
    #        corresponding state-variable indices for this model are :math:`E = 0`
    #        and :math:`I = 1`.""",
    #        order = 16)

    variables_of_interest = basic.Enumerate(
        label="Variables watched by Monitors",
        options=["E", "I"],
        default=["E"],
        select_multiple=True,
        doc="""This represents the default state-variables of this Model to be
               monitored. It can be overridden for each Monitor if desired. The
               corresponding state-variable indices for this model are :math:`E = 0`
               and :math:`I = 1`.""",
        order=24)

    #    coupling_variables = arrays.IntegerArray(
    #        label = "Variables to couple activity through",
    #        default = numpy.array([0], dtype=numpy.int32))

    #    nsig = arrays.FloatArray(
    #        label = "Noise dispersion",
    #        default = numpy.array([0.0]),
    #        range = basic.Range(lo = 0.0, hi = 1.0))

    def __init__(self, **kwargs):
        """
        Initialize the WilsonCowan model's traited attributes, any provided as
        keywords will overide their traited default.

        """
        LOG.info('%s: initing...' % str(self))
        super(WilsonCowan, self).__init__(**kwargs)
        # self._state_variables = ["E", "I"]
        self._nvar = 2
        self.cvar = numpy.array([0, 1], dtype=numpy.int32)
        LOG.debug('%s: inited.' % repr(self))

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""

        .. math::
            \tau \dot{x}(t) &= -z(t) + \phi(z(t)) \\
            \phi(x) &= \frac{c}{1-exp(-a (x-b))}

        """

        E = state_variables[0, :]
        I = state_variables[1, :]

        # long-range coupling
        c_0 = coupling[0, :]

        # short-range (local) coupling
        lc_0 = local_coupling * E
        lc_1 = local_coupling * I

        x_e = self.alpha_e * (self.c_ee * E - self.c_ei * I + self.P  - self.theta_e +  c_0 + lc_0 + lc_1)
        x_i = self.alpha_i * (self.c_ie * E - self.c_ii * I + self.Q  - self.theta_i + lc_0 + lc_1)

        s_e = self.c_e / (1.0 + numpy.exp(-self.a_e * (x_e - self.b_e)))
        s_i = self.c_i / (1.0 + numpy.exp(-self.a_i * (x_i - self.b_i)))

        dE = (-E + (self.k_e - self.r_e * E) * s_e) / self.tau_e
        dI = (-I + (self.k_i - self.r_i * I) * s_i) / self.tau_i

        derivative = numpy.array([dE, dI])

        return derivative

    # info for device_data
    device_info = model_device_info(

        pars=['c_1', 'c_2', 'c_3', 'c_4', 'tau_e', 'tau_i', 'a_e', 'theta_e',
              'a_i', 'theta_i', 'r_e', 'r_i', 'k_e', 'k_i'],

        kernel="""
        // read parameters
        float c_1     = P(0)
            , c_2     = P(1)
            , c_3     = P(2)
            , c_4     = P(3)
            , tau_e   = P(4)
            , tau_i   = P(5)
            , a_e     = P(6)
            , theta_e = P(7)
            , a_i     = P(8)
            , theta_i = P(9)
            , r_e     = P(10)
            , r_i     = P(11)
            , k_e     = P(12)
            , k_i     = P(13)

        // state variables
            , e = X(0)
            , i = X(1)

        // aux variables
            , c_0 = I(0)

            , x_e = c_1 * e - c_2 * i + c_0
            , x_i = c_3 * e - c_4 * i

            , s_e = 1.0 / (1.0 + exp(-a_e * (x_e - theta_e)))
            , s_i = 1.0 / (1.0 + exp(-a_i * (x_i - theta_i)));

        // set derivatives
        DX(0) = (-e + (k_e - r_e * e) * s_e) / tau_e;
        DX(1) = (-i + (k_i - r_i * i) * s_i) / tau_e;
        """
    )


class ReducedSetFitzHughNagumo(Model):
    r"""
    A reduced representation of a set of Fitz-Hugh Nagumo oscillators,
    [SJ_2008]_.

    The models (:math:`\xi`, :math:`\eta`) phase-plane, including a
    representation of the vector field as well as its nullclines, using default
    parameters, can be seen below:

        .. _phase-plane-rFHN_0:
        .. figure :: img/ReducedSetFitzHughNagumo_01_mode_0_pplane.svg
            :alt: Reduced set of FitzHughNagumo phase plane (xi, eta), 1st mode.

            The (:math:`\xi`, :math:`\eta`) phase-plane for the first mode of
            a reduced set of Fitz-Hugh Nagumo oscillators.

        .. _phase-plane-rFHN_1:
        .. figure :: img/ReducedSetFitzHughNagumo_01_mode_1_pplane.svg
            :alt: Reduced set of FitzHughNagumo phase plane (xi, eta), 2nd mode.

            The (:math:`\xi`, :math:`\eta`) phase-plane for the second mode of
            a reduced set of Fitz-Hugh Nagumo oscillators.

        .. _phase-plane-rFHN_2:
        .. figure :: img/ReducedSetFitzHughNagumo_01_mode_2_pplane.svg
            :alt: Reduced set of FitzHughNagumo phase plane (xi, eta), 3rd mode.

            The (:math:`\xi`, :math:`\eta`) phase-plane for the third mode of
            a reduced set of Fitz-Hugh Nagumo oscillators.


    .. automethod:: ReducedSetFitzHughNagumo.__init__

    The system's equations for the i-th mode at node q are:

    .. math::
                \dot{\xi}_{i}    &=  c\left(\xi_i-e_i\frac{\xi_{i}^3}{3} -\eta_{i}\right)
                                  + K_{11}\left[\sum_{k=1}^{o} A_{ik}\xi_k-\xi_i\right]
                                  - K_{12}\left[\sum_{k =1}^{o} B_{i k}\alpha_k-\xi_i\right] + cIE_i                       \\
                                 &\, + \left[\sum_{k=1}^{o} \mathbf{\Gamma}(\xi_{kq}, \xi_{kr}, u_{qr})\right]
                                  +  \left[\sum_{k=1}^{o} W_{\zeta}\cdot\xi_{kr} \right],                            \\
                \dot{\eta}_i     &= \frac{1}{c}\left(\xi_i-b\eta_i+m_i\right),                                              \\
                &                                                                                                \\
                \dot{\alpha}_i   &= c\left(\alpha_i-f_i\frac{\alpha_i^3}{3}-\beta_i\right)
                                  + K_{21}\left[\sum_{k=1}^{o} C_{ik}\xi_i-\alpha_i\right] + cII_i                          \\
                                 & \, + \left[\sum_{k=1}^{o} \mathbf{\Gamma}(\xi_{kq}, \xi_{kr}, u_{qr})\right]
                                  + \left[\sum_{k=1}^{o} W_{\zeta}\cdot\xi_{kr}\right],                          \\
                                 &                                                                               \\
                \dot{\beta}_i    &= \frac{1}{c}\left(\alpha_i-b\beta_i+n_i\right),

    .. automethod:: ReducedSetFitzHughNagumo.update_derived_parameters

    #NOTE: In the Article this modelis called StefanescuJirsa2D

    """
    _ui_name = "Stefanescu-Jirsa 2D"
    ui_configurable_parameters = ['tau', 'a', 'b', 'K11', 'K12', 'K21', 'sigma',
                                  'mu']

    # Define traited attributes for this model, these represent possible kwargs.
    tau = arrays.FloatArray(
        label=r":math:`\tau`",
        default=numpy.array([3.0]),
        range=basic.Range(lo=1.5, hi=4.5, step=0.01),
        doc="""doc...(prob something about timescale seperation)""",
        order=1)

    a = arrays.FloatArray(
        label=":math:`a`",
        default=numpy.array([0.45]),
        range=basic.Range(lo=0.0, hi=1.0, step=0.01),
        doc="""doc...""",
        order=2)

    b = arrays.FloatArray(
        label=":math:`b`",
        default=numpy.array([0.9]),
        range=basic.Range(lo=0.0, hi=1.0, step=0.01),
        doc="""doc...""",
        order=3)

    K11 = arrays.FloatArray(
        label=":math:`K_{11}`",
        default=numpy.array([0.5]),
        range=basic.Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Internal coupling, excitatory to excitatory""",
        order=4)

    K12 = arrays.FloatArray(
        label=":math:`K_{12}`",
        default=numpy.array([0.15]),
        range=basic.Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Internal coupling, excitatory to inhibitory""",
        order=5)

    K21 = arrays.FloatArray(
        label=":math:`K_{21}`",
        default=numpy.array([0.15]),
        range=basic.Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Internal coupling, inhibitory to excitatory""",
        order=6)

    sigma = arrays.FloatArray(
        label=r":math:`\sigma`",
        default=numpy.array([0.35]),
        range=basic.Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Standard deviation of Gaussian distribution""",
        order=7)

    mu = arrays.FloatArray(
        label=r":math:`\mu`",
        default=numpy.array([0.0]),
        range=basic.Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Mean of Gaussian distribution""",
        order=8)

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = basic.Dict(
        label="State Variable ranges [lo, hi]",
        default={"xi": numpy.array([-4.0, 4.0]),
                 "eta": numpy.array([-3.0, 3.0]),
                 "alpha": numpy.array([-4.0, 4.0]),
                 "beta": numpy.array([-3.0, 3.0])},
        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current
        parameters, it is used as a mechanism for bounding random inital
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""",
        order=9)

    #    variables_of_interest = arrays.IntegerArray(
    #        label = "Variables watched by Monitors",
    #        range = basic.Range(lo = 0.0, hi = 4.0, step = 1.0),
    #        default = numpy.array([0, 2], dtype=numpy.int32),
    #        doc = r"""This represents the default state-variables of this Model to be
    #        monitored. It can be overridden for each Monitor if desired. The
    #        corresponding state-variable indices for this model are :math:`\xi = 0`,
    #        :math:`\eta = 1`, :math:`\alpha = 2`, and :math:`\beta= 3`.""",
    #        order = 10)

    variables_of_interest = basic.Enumerate(
        label="Variables watched by Monitors",
        options=["xi", "eta", "alpha", "beta"],
        default=["xi", "alpha"],
        select_multiple=True,
        doc=r"""This represents the default state-variables of this Model to be
                                    monitored. It can be overridden for each Monitor if desired. The
                                    corresponding state-variable indices for this model are :math:`\xi = 0`,
                                    :math:`\eta = 1`, :math:`\alpha = 2`, and :math:`\beta= 3`.""",
        order=10)

    #    number_of_modes = Integer(
    #        order = -1, #-1 => don't show me as a configurable option in the UI...
    #        label = "Number of modes",
    #        default = 3)
    #
    #    nu = Integer(
    #        order = -1, #-1 => don't show me as a configurable option in the UI...
    #        label = "nu",
    #        default = 1500,
    #        range = basic.Range(lo = 0, hi = 10000, step = 100),
    #        doc = """Discretisation of Inhibitory distribution""")
    #
    #    nv = Integer(
    #        order = -1, #-1 => don't show me as a configurable option in the UI...
    #        label = "nv",
    #        default = 1500,
    #        range = basic.Range(lo = 0, hi = 10000, step = 100),
    #        doc = """Discretisation of Excitatory distribution""")

    #    coupling_variables = trait.Array(
    #        label = "Variables to couple activity through",
    #        default = numpy.array([0, 2], dtype=numpy.int32))

    #    nsig = trait.Array(label = "Noise dispersion",
    #                       default = numpy.array([0.0]))

    def __init__(self, **kwargs):
        """
        Initialise parameters for a reduced representation of a set of
        Fitz-Hugh Nagumo oscillators.

        """
        super(ReducedSetFitzHughNagumo, self).__init__(**kwargs)
        # self._state_variables = ["xi", "eta", "alpha", "beta"]
        self._nvar = 4
        self.cvar = numpy.array([0, 2], dtype=numpy.int32)

        # TODO: Hack fix, these cause issues with mapping spatialised parameters
        #      at the region level to the surface for surface sims.
        # NOTE: Existing modes definition (from the paper) is not properly
        #      normalised, so number_of_modes can't really be changed
        #      meaningfully anyway adnd nu and nv just need to be "large enough"
        #      so chaning them is only really an optimisation thing...
        self.number_of_modes = 3
        self.nu = 15000
        self.nv = 15000

        # Derived parameters
        self.Aik = None
        self.Bik = None
        self.Cik = None
        self.e_i = None
        self.f_i = None
        self.IE_i = None
        self.II_i = None
        self.m_i = None
        self.n_i = None

    def configure(self):
        """  """
        super(ReducedSetFitzHughNagumo, self).configure()

        if numpy.mod(self.nv, self.number_of_modes):
            error_msg = "nv must be divisible by the number_of_modes: %s"
            LOG.error(error_msg % repr(self))

        if numpy.mod(self.nu, self.number_of_modes):
            error_msg = "nu must be divisible by the number_of_modes: %s"
            LOG.error(error_msg % repr(self))

        self.update_derived_parameters()

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""


        The system's equations for the i-th mode at node q are:

        .. math::
                \dot{\xi}_{i}    &=  c\left(\xi_i-e_i\frac{\xi_{i}^3}{3} -\eta_{i}\right)
                                  + K_{11}\left[\sum_{k=1}^{o} A_{ik}\xi_k-\xi_i\right]
                                  - K_{12}\left[\sum_{k =1}^{o} B_{i k}\alpha_k-\xi_i\right] + cIE_i                       \\
                                 &\, + \left[\sum_{k=1}^{o} \mathbf{\Gamma}(\xi_{kq}, \xi_{kr}, u_{qr})\right]
                                  +  \left[\sum_{k=1}^{o} W_{\zeta}\cdot\xi_{kr} \right],                            \\
                \dot{\eta}_i     &= \frac{1}{c}\left(\xi_i-b\eta_i+m_i\right),                                              \\
                &                                                                                                \\
                \dot{\alpha}_i   &= c\left(\alpha_i-f_i\frac{\alpha_i^3}{3}-\beta_i\right)
                                  + K_{21}\left[\sum_{k=1}^{o} C_{ik}\xi_i-\alpha_i\right] + cII_i                          \\
                                 & \, + \left[\sum_{k=1}^{o} \mathbf{\Gamma}(\xi_{kq}, \xi_{kr}, u_{qr})\right]
                                  + \left[\sum_{k=1}^{o} W_{\zeta}\cdot\xi_{kr}\right],                          \\
                                 &                                                                               \\
                \dot{\beta}_i    &= \frac{1}{c}\left(\alpha_i-b\beta_i+n_i\right),

        """

        xi = state_variables[0, :]
        eta = state_variables[1, :]
        alpha = state_variables[2, :]
        beta = state_variables[3, :]

        # sum the activity from the modes
        c_0 = coupling[0, :].sum(axis=1)[:, numpy.newaxis]

        # TODO: generalize coupling variables to a matrix form
        # c_1 = coupling[1, :] # this cv represents alpha

        dxi = (self.tau * (xi - self.e_i * xi ** 3 / 3.0 - eta) +
               self.K11 * (numpy.dot(xi, self.Aik) - xi) -
               self.K12 * (numpy.dot(alpha, self.Bik) - xi) +
               self.tau * (self.IE_i + c_0 + local_coupling * xi))

        deta = (xi - self.b * eta + self.m_i) / self.tau

        dalpha = (self.tau * (alpha - self.f_i * alpha ** 3 / 3.0 - beta) +
                  self.K21 * (numpy.dot(xi, self.Cik) - alpha) +
                  self.tau * (self.II_i + c_0 + local_coupling * xi))

        dbeta = (alpha - self.b * beta + self.n_i) / self.tau

        derivative = numpy.array([dxi, deta, dalpha, dbeta])
        # import pdb; pdb.set_trace()
        return derivative

    def update_derived_parameters(self):
        """
        Calculate coefficients for the Reduced FitzHugh-Nagumo oscillator based
        neural field model. Specifically, this method implements equations for
        calculating coefficients found in the supplemental material of
        [SJ_2008]_.

        Include equations here...

        """

        newaxis = numpy.newaxis
        trapz = scipy_integrate_trapz

        stepu = 1.0 / (self.nu + 2 - 1)
        stepv = 1.0 / (self.nv + 2 - 1)

        norm = scipy_stats_norm(loc=self.mu, scale=self.sigma)

        Zu = norm.ppf(numpy.arange(stepu, 1.0, stepu))
        Zv = norm.ppf(numpy.arange(stepv, 1.0, stepv))

        # Define the modes
        V = numpy.zeros((self.number_of_modes, self.nv))
        U = numpy.zeros((self.number_of_modes, self.nu))

        nv_per_mode = self.nv / self.number_of_modes
        nu_per_mode = self.nu / self.number_of_modes

        for i in range(self.number_of_modes):
            V[i, i * nv_per_mode:(i + 1) * nv_per_mode] = numpy.ones(nv_per_mode)
            U[i, i * nu_per_mode:(i + 1) * nu_per_mode] = numpy.ones(nu_per_mode)

        # Normalise the modes
        V = V / numpy.tile(numpy.sqrt(trapz(V * V, Zv, axis=1)), (self.nv, 1)).T
        U = U / numpy.tile(numpy.sqrt(trapz(U * U, Zu, axis=1)), (self.nv, 1)).T

        # Get Normal PDF's evaluated with sampling Zv and Zu
        g1 = norm.pdf(Zv)
        g2 = norm.pdf(Zu)
        G1 = numpy.tile(g1, (self.number_of_modes, 1))
        G2 = numpy.tile(g2, (self.number_of_modes, 1))

        cV = numpy.conj(V)
        cU = numpy.conj(U)

        intcVdZ = trapz(cV, Zv, axis=1)[:, newaxis]
        intG1VdZ = trapz(G1 * V, Zv, axis=1)[newaxis, :]
        intcUdZ = trapz(cU, Zu, axis=1)[:, newaxis]
        # import pdb; pdb.set_trace()
        # Calculate coefficients
        self.Aik = numpy.dot(intcVdZ, intG1VdZ).T
        self.Bik = numpy.dot(intcVdZ, trapz(G2 * U, Zu, axis=1)[newaxis, :])
        self.Cik = numpy.dot(intcUdZ, intG1VdZ).T

        self.e_i = trapz(cV * V ** 3, Zv, axis=1)[newaxis, :]
        self.f_i = trapz(cU * U ** 3, Zu, axis=1)[newaxis, :]

        self.IE_i = trapz(Zv * cV, Zv, axis=1)[newaxis, :]
        self.II_i = trapz(Zu * cU, Zu, axis=1)[newaxis, :]

        self.m_i = (self.a * intcVdZ).T
        self.n_i = (self.a * intcUdZ).T
        # import pdb; pdb.set_trace()

    # DRAGONS BE HERE
    device_info = model_device_info(
        pars=[
            # given parameters
            'tau', 'a', 'b', 'K11', 'K12', 'K21', 'sigma', 'mu',

            # derived parameters
            'Aik', 'Bik', 'Cik', 'e_i', 'f_i', 'IE_i', 'II_i', 'm_i', 'n_i'
        ],

        kernel="""
        // read given parameters

        float tau   = P(0)
            , a     = P(1)
            , b     = P(2)
            , K11   = P(3)
            , K12   = P(4)
            , K21   = P(5)
            , sigma = P(6)
            , mu    = P(7)

        // modal derived par macros
#define A(i,k) P((8 + 0*9 + 3*(i) + (k)))
#define B(i,k) P((8 + 1*9 + 3*(i) + (k)))
#define C(i,k) P((8 + 2*9 + 3*(i) + (k)))

#define E(i)   P((8 + 3*9 + 3*0 + (i)))
#define F(i)   P((8 + 3*9 + 3*1 + (i)))
#define IE(i)  P((8 + 3*9 + 3*2 + (i)))
#define II(i)  P((8 + 3*9 + 3*3 + (i)))
#define M(i)   P((8 + 3*9 + 3*4 + (i)))
#define N(i)   P((8 + 3*9 + 3*5 + (i)))

        // state variables macros
#define XI(i) X((0*n_mode + (i)))
#define ETA(i) X((1*n_mode + (i)))
#define ALPHA(i) X((2*n_mode + (i)))
#define BETA(i) X((3*n_mode + (i)))

        // aux variables

            , c_0 = I(0)
            , c_1 = I(1) /* --->>>>>>> */ ; /* <<<-------- extremely important semicolon */

        // nothing else but the dot product of the modal interaction coefficients with the instantanoneous state
#define XI_dot_A(k) (XI(1)*A(1, (k)) + XI(2)*A(2, (k)) + XI(3)*A(3, (k)))
#define XI_dot_C(k) (XI(1)*C(1, (k)) + XI(2)*C(2, (k)) + XI(3)*C(3, (k)))
#define ALPHA_dot_B(k) (ALPHA(1)*B(1, (k)) + ALPHA(2)*B(2, (k)) + ALPHA(3)*B(3, (k)))

        // derivatives
        for (int i=0; i<n_mode; i++)
        {
            DX(n_mode*0 + i) = (tau * (XI(i) - E(i)*XI(i)*XI(i)*XI(i)/3.0 - ETA(i)) + \
                                K11 * (XI_dot_A(i) - XI(i)) - \
                                K12 * (ALPHA_dot_B(i) - XI(i)) + \
                                tau * (IE(i) + c_0));

            DX(n_mode*1 + i) = (XI(i) - b*ETA(i) + M(i)) / tau;

            DX(n_mode*2 + i) = (tau * (ALPHA(i) - F(i)*ALPHA(i)*ALPHA(i)*ALPHA(i)/3.0 - BETA(i)) + \
                                K21 * (XI_dot_C(i) - ALPHA(i)) + \
                                tau * (II(i) + c_1));

            DX(n_mode*3 + i) = (ALPHA(i) - b*BETA(i) + N(i)) / tau;
        }

        // clean up
#undef A
#undef B
#undef C
#undef E
#undef F
#undef IE
#undef II
#undef M
#undef N
#undef XI
#undef ETA
#undef ALPHA
#undef BETA
#undef XI_dot_A
#undef XI_dot_C
#undef ALPHA_dot_B
        """
    )


class ReducedSetHindmarshRose(Model):
    r"""
    .. [SJ_2008] Stefanescu and Jirsa, PLoS Computational Biology, *A Low
        Dimensional Description of Globally Coupled Heterogeneous Neural
        Networks of Excitatory and Inhibitory*  4, 11, 26--36, 2008.

    The models (:math:`\xi`, :math:`\eta`) phase-plane, including a
    representation of the vector field as well as its nullclines, using default
    parameters, can be seen below:

        .. _phase-plane-rHR_0:
        .. figure :: img/ReducedSetHindmarshRose_01_mode_0_pplane.svg
            :alt: Reduced set of FitzHughNagumo phase plane (xi, eta), 1st mode.

            The (:math:`\xi`, :math:`\eta`) phase-plane for the first mode of
            a reduced set of Hindmarsh-Rose oscillators.

        .. _phase-plane-rHR_1:
        .. figure :: img/ReducedSetHindmarshRose_01_mode_1_pplane.svg
            :alt: Reduced set of FitzHughNagumo phase plane (xi, eta), 2nd mode.

            The (:math:`\xi`, :math:`\eta`) phase-plane for the second mode of
            a reduced set of Hindmarsh-Rose oscillators.

        .. _phase-plane-rHR_2:
        .. figure :: img/ReducedSetHindmarshRose_01_mode_2_pplane.svg
            :alt: Reduced set of FitzHughNagumo phase plane (xi, eta), 3rd mode.

            The (:math:`\xi`, :math:`\eta`) phase-plane for the third mode of
            a reduced set of Hindmarsh-Rose oscillators.

    .. automethod:: ReducedSetHindmarshRose.__init__

    The dynamic equations were orginally taken from [SJ_2008]_.

    The equations of the population model for i-th mode at node q are:

    .. math::
                \dot{\xi}_i     &=  \eta_i-a_i\xi_i^3 + b_i\xi_i^2- \tau_i
                                 + K_{11} \left[\sum_{k=1}^{o} A_{ik} \xi_k - \xi_i \right]
                                 - K_{12} \left[\sum_{k=1}^{o} B_{ik} \alpha_k - \xi_i\right] + IE_i                \\
                                &\, + \left[\sum_{k=1}^{o} \mathbf{\Gamma}(\xi_{kq}, \xi_{kr}, u_{qr})\right]
                                 + \left[\sum_{k=1}^{o} W_{\zeta}\cdot\xi_{kr} \right],                     \\
                                &                                                                         \\
                \dot{\eta}_i    &=  c_i-d_i\xi_i^2 -\tau_i,                                                         \\
                %
                \dot{\tau}_i    &=  rs\xi_i - r\tau_i -m_i,                                                         \\
                %
                \dot{\alpha}_i  &=  \beta_i - e_i \alpha_i^3 + f_i \alpha_i^2 - \gamma_i
                                 + K_{21} \left[\sum_{k=1}^{o} C_{ik} \xi_k - \alpha_i \right] + II_i               \\
                                &\, +\left[\sum_{k=1}^{o}\mathbf{\Gamma}(\xi_{kq}, \xi_{kr}, u_{qr})\right]
                                 + \left[\sum_{k=1}^{o}W_{\zeta}\cdot\xi_{kr}\right],                    \\
                                &                                                                         \\
                \dot{\beta}_i   &= h_i - p_i \alpha_i^2 - \beta_i,                                                   \\
                \dot{\gamma}_i  &= rs \alpha_i - r \gamma_i - n_i,

    .. automethod:: ReducedSetHindmarshRose.update_derived_parameters

    #NOTE: In the Article this modelis called StefanescuJirsa3D

    """
    _ui_name = "Stefanescu-Jirsa 3D"
    ui_configurable_parameters = ['r', 'a', 'b', 'c', 'd', 's', 'xo', 'K11',
                                  'K12', 'K21', 'sigma', 'mu']

    # Define traited attributes for this model, these represent possible kwargs.
    r = arrays.FloatArray(
        label=":math:`r`",
        default=numpy.array([0.006]),
        range=basic.Range(lo=0.0, hi=0.1, step=0.0005),
        doc="""Adaptation parameter""",
        order=1)

    a = arrays.FloatArray(
        label=":math:`a`",
        default=numpy.array([1.0]),
        range=basic.Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Dimensionless parameter as in the Hindmarsh-Rose model""",
        order=2)

    b = arrays.FloatArray(
        label=":math:`b`",
        default=numpy.array([3.0]),
        range=basic.Range(lo=0.0, hi=3.0, step=0.01),
        doc="""Dimensionless parameter as in the Hindmarsh-Rose model""",
        order=3)

    c = arrays.FloatArray(
        label=":math:`c`",
        default=numpy.array([1.0]),
        range=basic.Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Dimensionless parameter as in the Hindmarsh-Rose model""",
        order=4)

    d = arrays.FloatArray(
        label=":math:`d`",
        default=numpy.array([5.0]),
        range=basic.Range(lo=2.5, hi=7.5, step=0.01),
        doc="""Dimensionless parameter as in the Hindmarsh-Rose model""",
        order=5)

    s = arrays.FloatArray(
        label=":math:`s`",
        default=numpy.array([4.0]),
        range=basic.Range(lo=2.0, hi=6.0, step=0.01),
        doc="""Adaptation paramters, governs feedback""",
        order=6)

    xo = arrays.FloatArray(
        label=":math:`x_{o}`",
        default=numpy.array([-1.6]),
        range=basic.Range(lo=-2.4, hi=-0.8, step=0.01),
        doc="""Leftmost equilibrium point of x""",
        order=7)

    K11 = arrays.FloatArray(
        label=":math:`K_{11}`",
        default=numpy.array([0.5]),
        range=basic.Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Internal coupling, excitatory to excitatory""",
        order=8)

    K12 = arrays.FloatArray(
        label=":math:`K_{12}`",
        default=numpy.array([0.1]),
        range=basic.Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Internal coupling, excitatory to inhibitory""",
        order=9)

    K21 = arrays.FloatArray(
        label=":math:`K_{21}`",
        default=numpy.array([0.15]),
        range=basic.Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Internal coupling, inhibitory to excitatory""",
        order=10)

    sigma = arrays.FloatArray(
        label=r":math:`\sigma`",
        default=numpy.array([0.3]),
        range=basic.Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Standard deviation of Gaussian distribution""",
        order=11)

    mu = arrays.FloatArray(
        label=r":math:`\mu`",
        default=numpy.array([3.3]),
        range=basic.Range(lo=1.1, hi=3.3, step=0.01),
        doc="""Mean of Gaussian distribution""",
        order=12)

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = basic.Dict(
        label="State Variable ranges [lo, hi]",
        default={"xi": numpy.array([-4.0, 4.0]),
                 "eta": numpy.array([-25.0, 20.0]),
                 "tau": numpy.array([2.0, 10.0]),
                 "alpha": numpy.array([-4.0, 4.0]),
                 "beta": numpy.array([-20.0, 20.0]),
                 "gamma": numpy.array([2.0, 10.0])},
        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current
        parameters, it is used as a mechanism for bounding random inital
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""",
        order=13)

    variables_of_interest = basic.Enumerate(
        label="Variables watched by Monitors",
        options=["xi", "eta", "tau", "alpha", "beta", "gamma"],
        default=["xi", "eta", "tau"],
        select_multiple=True,
        doc=r"""This represents the default state-variables of this Model to be
                monitored. It can be overridden for each Monitor if desired. The
                corresponding state-variable indices for this model are :math:`\xi = 0`,
                :math:`\eta = 1`, :math:`\tau = 2`, :math:`\alpha = 3`,
                :math:`\beta = 4`, and :math:`\gamma = 5`""",
        order=14)

    #    variables_of_interest = arrays.IntegerArray(
    #        label = "Variables watched by Monitors",
    #        range = basic.Range(lo = 0.0, hi = 6.0, step = 1.0),
    #        default = numpy.array([0, 3], dtype=numpy.int32),
    #        doc = r"""This represents the default state-variables of this Model to be
    #        monitored. It can be overridden for each Monitor if desired. The
    #        corresponding state-variable indices for this model are :math:`\xi = 0`,
    #        :math:`\eta = 1`, :math:`\tau = 2`, :math:`\alpha = 3`,
    #        :math:`\beta = 4`, and :math:`\gamma = 5`""",
    #        order = 14)

    #    number_of_modes = Integer(
    #        order = -1, #-1 => don't show me as a configurable option in the UI...
    #        label = "Number of modes",
    #        default = 3,
    #        doc = """Number of modes""")
    #
    #    nu = Integer(
    #        order = -1, #-1 => don't show me as a configurable option in the UI...
    #        label = "nu",
    #        default = 1500,
    #        range = basic.Range(lo = 500, hi = 10000, step = 500),
    #        doc = """Discretisation of Inhibitory distribution""")
    #
    #    nv = Integer(
    #        order = -1, #-1 => don't show me as a configurable option in the UI...
    #        label = "nv",
    #        default = 1500,
    #        range = basic.Range(lo = 500, hi = 10000, step = 500),
    #        doc = """Discretisation of Excitatory distribution""")

    #    coupling_variables = arrays.IntegerArray(
    #        label = "Variables to couple activity through",
    #        default = numpy.array([0, 3], dtype=numpy.int32))

    #    nsig = arrays.FloatArray(label = "Noise dispersion",
    #                       default = numpy.array([0.0]))

    def __init__(self, **kwargs):
        """
        Initialise parameters for a reduced representation of a set of
        Hindmarsh Rose oscillators, [SJ_2008]_.

        """
        super(ReducedSetHindmarshRose, self).__init__(**kwargs)
        # self._state_variables = ["xi", "eta", "tau", "alpha", "beta", "gamma"]
        self._nvar = 6
        self.cvar = numpy.array([0, 3], dtype=numpy.int32)

        # TODO: Hack fix, these cause issues with mapping spatialised parameters
        #      at the region level to the surface for surface sims.
        # NOTE: Existing modes definition (from the paper) is not properly
        #      normalised, so number_of_modes can't really be changed
        #      meaningfully anyway adnd nu and nv just need to be "large enough"
        #      so chaning them is only really an optimisation thing...
        self.number_of_modes = 3
        self.nu = 1500
        self.nv = 1500

        # derived parameters
        self.A_ik = None
        self.B_ik = None
        self.C_ik = None
        self.a_i = None
        self.b_i = None
        self.c_i = None
        self.d_i = None
        self.e_i = None
        self.f_i = None
        self.h_i = None
        self.p_i = None
        self.IE_i = None
        self.II_i = None
        self.m_i = None
        self.n_i = None

    def configure(self):
        """  """
        super(ReducedSetHindmarshRose, self).configure()

        if numpy.mod(self.nv, self.number_of_modes):
            error_msg = "nv must be divisible by the number_of_modes: %s"
            LOG.error(error_msg % repr(self))

        if numpy.mod(self.nu, self.number_of_modes):
            error_msg = "nu must be divisible by the number_of_modes: %s"
            LOG.error(error_msg % repr(self))

        self.update_derived_parameters()

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        The equations of the population model for i-th mode at node q are:

        .. math::
                \dot{\xi}_i     &=  \eta_i-a_i\xi_i^3 + b_i\xi_i^2- \tau_i
                                 + K_{11} \left[\sum_{k=1}^{o} A_{ik} \xi_k - \xi_i \right]
                                 - K_{12} \left[\sum_{k=1}^{o} B_{ik} \alpha_k - \xi_i\right] + IE_i                \\
                                &\, + \left[\sum_{k=1}^{o} \mathbf{\Gamma}(\xi_{kq}, \xi_{kr}, u_{qr})\right]
                                 + \left[\sum_{k=1}^{o} W_{\zeta}\cdot\xi_{kr} \right],                     \\
                                &                                                                         \\
                \dot{\eta}_i    &=  c_i-d_i\xi_i^2 -\tau_i,                                                         \\
                %
                \dot{\tau}_i    &=  rs\xi_i - r\tau_i -m_i,                                                         \\
                %
                \dot{\alpha}_i  &=  \beta_i - e_i \alpha_i^3 + f_i \alpha_i^2 - \gamma_i
                                 + K_{21} \left[\sum_{k=1}^{o} C_{ik} \xi_k - \alpha_i \right] + II_i               \\
                                &\, +\left[\sum_{k=1}^{o}\mathbf{\Gamma}(\xi_{kq}, \xi_{kr}, u_{qr})\right]
                                 + \left[\sum_{k=1}^{o}W_{\zeta}\cdot\xi_{kr}\right],                    \\
                                &                                                                         \\
                \dot{\beta}_i   &= h_i - p_i \alpha_i^2 - \beta_i,                                                   \\
                \dot{\gamma}_i  &= rs \alpha_i - r \gamma_i - n_i,

        """

        xi = state_variables[0, :]
        eta = state_variables[1, :]
        tau = state_variables[2, :]
        alpha = state_variables[3, :]
        beta = state_variables[4, :]
        gamma = state_variables[5, :]

        c_0 = coupling[0, :].sum(axis=1)[:, numpy.newaxis]
        # c_1 = coupling[1, :]

        dxi = (eta - self.a_i * xi ** 3 + self.b_i * xi ** 2 - tau +
               self.K11 * (numpy.dot(xi, self.A_ik) - xi) -
               self.K12 * (numpy.dot(alpha, self.B_ik) - xi) +
               self.IE_i + c_0 + local_coupling * xi)

        deta = self.c_i - self.d_i * xi ** 2 - eta

        dtau = self.r * self.s * xi - self.r * tau - self.m_i

        dalpha = (beta - self.e_i * alpha ** 3 + self.f_i * alpha ** 2 - gamma +
                  self.K21 * (numpy.dot(xi, self.C_ik) - alpha) +
                  self.II_i + c_0 + local_coupling * xi)

        dbeta = self.h_i - self.p_i * alpha ** 2 - beta

        dgamma = self.r * self.s * alpha - self.r * gamma - self.n_i

        derivative = numpy.array([dxi, deta, dtau, dalpha, dbeta, dgamma])

        return derivative


    def update_derived_parameters(self):
        """
        Calculate coefficients for the neural field model based on a Reduced set
        of Hindmarsh-Rose oscillators. Specifically, this method implements
        equations for calculating coefficients found in the supplemental
        material of [SJ_2008]_.

        Include equations here...

        """

        newaxis = numpy.newaxis
        trapz = scipy_integrate_trapz

        stepu = 1.0 / (self.nu + 2 - 1)
        stepv = 1.0 / (self.nv + 2 - 1)

        norm = scipy_stats_norm(loc=self.mu, scale=self.sigma)

        Iu = norm.ppf(numpy.arange(stepu, 1.0, stepu))
        Iv = norm.ppf(numpy.arange(stepv, 1.0, stepv))

        # Define the modes
        V = numpy.zeros((self.number_of_modes, self.nv))
        U = numpy.zeros((self.number_of_modes, self.nu))

        nv_per_mode = self.nv / self.number_of_modes
        nu_per_mode = self.nu / self.number_of_modes

        for i in range(self.number_of_modes):
            V[i, i * nv_per_mode:(i + 1) * nv_per_mode] = numpy.ones(nv_per_mode)
            U[i, i * nu_per_mode:(i + 1) * nu_per_mode] = numpy.ones(nu_per_mode)

        # Normalise the modes
        V = V / numpy.tile(numpy.sqrt(trapz(V * V, Iv, axis=1)), (self.nv, 1)).T
        U = U / numpy.tile(numpy.sqrt(trapz(U * U, Iu, axis=1)), (self.nu, 1)).T

        # Get Normal PDF's evaluated with sampling Zv and Zu
        g1 = norm.pdf(Iv)
        g2 = norm.pdf(Iu)
        G1 = numpy.tile(g1, (self.number_of_modes, 1))
        G2 = numpy.tile(g2, (self.number_of_modes, 1))

        cV = numpy.conj(V)
        cU = numpy.conj(U)

        #import pdb; pdb.set_trace()
        intcVdI = trapz(cV, Iv, axis=1)[:, newaxis]
        intG1VdI = trapz(G1 * V, Iv, axis=1)[newaxis, :]
        intcUdI = trapz(cU, Iu, axis=1)[:, newaxis]

        #Calculate coefficients
        self.A_ik = numpy.dot(intcVdI, intG1VdI).T
        self.B_ik = numpy.dot(intcVdI, trapz(G2 * U, Iu, axis=1)[newaxis, :])
        self.C_ik = numpy.dot(intcUdI, intG1VdI).T

        self.a_i = self.a * trapz(cV * V ** 3, Iv, axis=1)[newaxis, :]
        self.e_i = self.a * trapz(cU * U ** 3, Iu, axis=1)[newaxis, :]
        self.b_i = self.b * trapz(cV * V ** 2, Iv, axis=1)[newaxis, :]
        self.f_i = self.b * trapz(cU * U ** 2, Iu, axis=1)[newaxis, :]
        self.c_i = (self.c * intcVdI).T
        self.h_i = (self.c * intcUdI).T

        self.IE_i = trapz(Iv * cV, Iv, axis=1)[newaxis, :]
        self.II_i = trapz(Iu * cU, Iu, axis=1)[newaxis, :]

        self.d_i = (self.d * intcVdI).T
        self.p_i = (self.d * intcUdI).T

        self.m_i = (self.r * self.s * self.xo * intcVdI).T
        self.n_i = (self.r * self.s * self.xo * intcUdI).T

    # DRAGONS BE HERE
    device_info = model_device_info(
        pars=[
            # given parameters
            'r', 'a', 'b', 'c', 'd', 's', 'xo', 'K11', 'K12', 'K21', 'sigma', 'mu',
            # derived parameters
            'A_ik', 'B_ik', 'C_ik', 'a_i', 'b_i', 'c_i', 'd_i', 'e_i',
            'f_i', 'h_i', 'p_i', 'IE_i', 'II_i', 'm_i', 'n_i'],


        kernel="""
        // read given parameters

        float r     = P(0)
            , a     = P(1)
            , b     = P(2)
            , c     = P(3)
            , d     = P(4)
            , s     = P(5)
            , xo    = P(6)
            , K11   = P(7)
            , K12   = P(8)
            , K21   = P(9)
            , sigma = P(10)
            , mu    = P(11)

        // modal derived par macros (mX is X_ik, vX is x_i (sorry))
#define mA(i, k) P((12 + 0*9 + 3*(i) + (k)))
#define mB(i, k) P((12 + 1*9 + 3*(i) + (k)))
#define mC(i, k) P((12 + 2*9 + 3*(i) + (k)))

#define vA(i)    P((12 + 3*9 + 3*0 + (i)))
#define vB(i)    P((12 + 3*9 + 3*1 + (i)))
#define vC(i)    P((12 + 3*9 + 3*2 + (i)))
#define vD(i)    P((12 + 3*9 + 3*3 + (i)))
#define vE(i)    P((12 + 3*9 + 3*4 + (i)))
#define vF(i)    P((12 + 3*9 + 3*5 + (i)))
#define vH(i)    P((12 + 3*9 + 3*6 + (i)))
#define vP(i)    P((12 + 3*9 + 3*7 + (i)))
#define vIE(i)   P((12 + 3*9 + 3*8 + (i)))
#define vII(i)   P((12 + 3*9 + 3*9 + (i)))
#define vM(i)    P((12 + 3*9 + 3*10 + (i)))
#define vN(i)    P((12 + 3*9 + 3*11 + (i)))

        // state variable macros
#define XI(i)    X((0*n_mode + (i)))
#define ETA(i)   X((1*n_mode + (i)))
#define TAU(i)   X((2*n_mode + (i)))
#define ALPHA(i) X((3*n_mode + (i)))
#define BETA(i)  X((4*n_mode + (i)))
#define GAMMA(i) X((5*n_mode + (i)))

        // aux variables
            , c_0 = I(0)
            , c_1 = I(1)       /* the semicolon --> */  ;  /* don't forget it */

        // modal interactions
#define XI_dot_A(k) (XI(1)*mA(1, (k)) + XI(2)*mA(2, (k)) + XI(3)*mA(3, (k)))
#define XI_dot_C(k) (XI(1)*mC(1, (k)) + XI(2)*mC(2, (k)) + XI(3)*mC(3, (k)))
#define ALPHA_dot_B(k) (ALPHA(1)*mB(1, (k)) + ALPHA(2)*mB(2, (k)) + ALPHA(3)*mB(3, (k)))

        // derivatives
        for (int i=0; i<n_mode; i++)
        {
/* xi */    DX(n_mode*0 + i) = (ETA(i) - vA(i)*XI(i)*XI(i)*XI(i) + vB(i)*XI(i)*XI(i) - TAU(i) + \
                               K11 * (XI_dot_A(i) - XI(i)) - \
                               K12 * (ALPHA_dot_B(i) - XI(i)) + \
                               vIE(i) + c_0);

/* eta */   DX(n_mode*1 + i) = vC(i) - vD(i)*XI(i)*XI(i) - ETA(i);

/* tau */   DX(n_mode*2 + i) = r*s*XI(i) - r*TAU(i) - vM(i);

/* alpha */ DX(n_mode*3 + i) = (BETA(i) - vE(i)*ALPHA(i)*ALPHA(i)*ALPHA(i) + vF(i)*ALPHA(i)*ALPHA(i) - GAMMA(i) +\
                                K21 * (XI_dot_C(i) - ALPHA(i)) + \
                                vII(i) + c_1);

/* beta */  DX(n_mode*4 + i) = vH(i) - vP(i)*ALPHA(i)*ALPHA(i) - BETA(i);

/* gamma */ DX(n_mode*5 + i) = r*s*ALPHA(i) - r*GAMMA(i) - vN(i);
        }

#undef mA
#undef mB
#undef mC

#undef vA
#undef vB
#undef vC
#undef vD
#undef vE
#undef vF
#undef vH
#undef vP
#undef vIE
#undef vII
#undef vM
#undef vN

#undef XI
#undef ETA
#undef TAU
#undef ALPHA
#undef BETA
#undef GAMMA

#undef XI_dot_A
#undef XI_dot_C
#undef ALPHA_dot_B

"""
    )



class JansenRit(Model):
    r"""
    The Jansen and Rit is a biologically inspired mathematical framework
    originally conceived to simulate the spontaneous electrical activity of
    neuronal assemblies, with a particular focus on alpha activity, for instance,
    as measured by EEG. Later on, it was discovered that in addition to alpha
    activity, this model was also able to simulate evoked potentials.

    .. [JR_1995]  Jansen, B., H. and Rit V., G., *Electroencephalogram and
        visual evoked potential generation in a mathematical model of
        coupled cortical columns*, Biological Cybernetics (73) 357:366, 1995.

    .. [J_1993] Jansen, B., Zouridakis, G. and Brandt, M., *A
        neurophysiologically-based mathematical model of flash visual evoked
        potentials*

    .. figure :: img/JansenRit_45_mode_0_pplane.svg
        :alt: Jansen and Rit phase plane (y4, y5)

        The (:math:`y_4`, :math:`y_5`) phase-plane for the Jansen and Rit model.

    .. automethod:: JansenRit.__init__

    The dynamic equations were taken from [JR_1995]_

    .. math::
        \dot{y_0} &= y_3 \\
        \dot{y_3} &= A a\,S[y_1 - y_2] - 2a\,y_3 - 2a^2\, y_0 \\
        \dot{y_1} &= y_4\\
        \dot{y_4} &= A a \,[p(t) + \alpha_2 J + S[\alpha_1 J\,y_0]+ c_0]
                    -2a\,y - a^2\,y_1 \\
        \dot{y_2} &= y_5 \\
        \dot{y_5} &= B b (\alpha_4 J\, S[\alpha_3 J \,y_0]) - 2 b\, y_5
                    - b^2\,y_2 \\
        S[v] &= \frac{2\, \nu_{max}}{1 + \exp^{r(v_0 - v)}}

    """

    _ui_name = "Jansen-Rit"
    ui_configurable_parameters = ['A', 'B', 'a', 'b', 'v0', 'nu_max', 'r', 'J',
                                  'a_1', 'a_2', 'a_3', 'a_4', 'p_min', 'p_max',
                                  'mu']

    #Define traited attributes for this model, these represent possible kwargs.
    A = arrays.FloatArray(
        label=":math:`A`",
        default=numpy.array([3.25]),
        range=basic.Range(lo=2.6, hi=9.75, step=0.05),
        doc="""Maximum amplitude of EPSP [mV]. Also called average synaptic gain.""",
        order=1)

    B = arrays.FloatArray(
        label=":math:`B`",
        default=numpy.array([22.0]),
        range=basic.Range(lo=17.6, hi=110.0, step=0.2),
        doc="""Maximum amplitude of IPSP [mV]. Also called average synaptic gain.""",
        order=2)

    a = arrays.FloatArray(
        label=":math:`a`",
        default=numpy.array([0.1]),
        range=basic.Range(lo=0.05, hi=0.15, step=0.01),
        doc="""Reciprocal of the time constant of passive membrane and all
        other spatially distributed delays in the dendritic network [ms^-1].
        Also called average synaptic time constant.""",
        order=3)

    b = arrays.FloatArray(
        label=":math:`b`",
        default=numpy.array([0.05]),
        range=basic.Range(lo=0.025, hi=0.075, step=0.005),
        doc="""Reciprocal of the time constant of passive membrane and all
        other spatially distributed delays in the dendritic network [ms^-1].
        Also called average synaptic time constant.""",
        order=4)

    v0 = arrays.FloatArray(
        label=":math:`v_0`",
        default=numpy.array([5.52]),
        range=basic.Range(lo=3.12, hi=6.0, step=0.02),
        doc="""Firing threshold (PSP) for which a 50% firing rate is achieved.
        In other words, it is the value of the average membrane potential
        corresponding to the inflection point of the sigmoid [mV].""",
        order=5)

    nu_max = arrays.FloatArray(
        label=r":math:`\nu_{max}`",
        default=numpy.array([0.0025]),
        range=basic.Range(lo=0.00125, hi=0.00375, step=0.00001),
        doc="""Determines the maximum firing rate of the neural population
        [s^-1].""",
        order=6)

    r = arrays.FloatArray(
        label=":math:`r`",
        default=numpy.array([0.56]),
        range=basic.Range(lo=0.28, hi=0.84, step=0.01),
        doc="""Steepness of the sigmoidal transformation [mV^-1].""",
        order=7)

    J = arrays.FloatArray(
        label=":math:`J`",
        default=numpy.array([135.0]),
        range=basic.Range(lo=65.0, hi=1350.0, step=1.),
        doc="""Average number of synapses between populations.""",
        order=8)

    a_1 = arrays.FloatArray(
        label=r":math:`\alpha_1`",
        default=numpy.array([1.0]),
        range=basic.Range(lo=0.5, hi=1.5, step=0.1),
        doc="""Average probability of synaptic contacts in the feedback
        excitatory loop.""",
        order=9)

    a_2 = arrays.FloatArray(
        label=r":math:`\alpha_2`",
        default=numpy.array([0.8]),
        range=basic.Range(lo=0.4, hi=1.2, step=0.1),
        doc="""Average probability of synaptic contacts in the feedback
        excitatory loop.""",
        order=10)

    a_3 = arrays.FloatArray(
        label=r":math:`\alpha_3`",
        default=numpy.array([0.25]),
        range=basic.Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the feedback
        excitatory loop.""",
        order=11)

    a_4 = arrays.FloatArray(
        label=r":math:`\alpha_4`",
        default=numpy.array([0.25]),
        range=basic.Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the slow feedback
        inhibitory loop.""",
        order=12)

    p_min = arrays.FloatArray(
        label=":math:`p_{min}`",
        default=numpy.array([0.12]),
        range=basic.Range(lo=0.0, hi=0.12, step=0.01),
        doc="""Minimum input firing rate.""",
        order=13)

    p_max = arrays.FloatArray(
        label=":math:`p_{max}`",
        default=numpy.array([0.32]),
        range=basic.Range(lo=0.0, hi=0.32, step=0.01),
        doc="""Maximum input firing rate.""",
        order=14)

    mu = arrays.FloatArray(
        label=r":math:`\mu_{max}`",
        default=numpy.array([0.22]),
        range=basic.Range(lo=0.0, hi=0.22, step=0.01),
        doc="""Mean input firing rate""",
        order=15)

    #Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = basic.Dict(
        label="State Variable ranges [lo, hi]",
        default={"y0": numpy.array([-1.0, 1.0]),
                 "y1": numpy.array([-500.0, 500.0]),
                 "y2": numpy.array([-50.0, 50.0]),
                 "y3": numpy.array([-6.0, 6.0]),
                 "y4": numpy.array([-20.0, 20.0]),
                 "y5": numpy.array([-500.0, 500.0])},
        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current
        parameters, it is used as a mechanism for bounding random inital
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""",
        order=16)

    variables_of_interest = basic.Enumerate(
        label="Variables watched by Monitors",
        options=["y0", "y1", "y2", "y3", "y4", "y5"],
        default=["y0", "y1", "y2", "y3"],
        select_multiple=True,
        doc="""This represents the default state-variables of this Model to be
                                    monitored. It can be overridden for each Monitor if desired. The
                                    corresponding state-variable indices for this model are :math:`y0 = 0`,
                                    :math:`y1 = 1`, :math:`y2 = 2`, :math:`y3 = 3`, :math:`y4 = 4`, and
                                    :math:`y5 = 5`""",
        order=17)

    #    variables_of_interest = arrays.IntegerArray(
    #        label = "Variables watched by Monitors",
    #        range = basic.Range(lo = 0.0, hi = 6.0, step = 1.0),
    #        default = numpy.array([0, 3], dtype=numpy.int32),
    #        doc = """This represents the default state-variables of this Model to be
    #        monitored. It can be overridden for each Monitor if desired. The
    #        corresponding state-variable indices for this model are :math:`y0 = 0`,
    #        :math:`y1 = 1`, :math:`y2 = 2`, :math:`y3 = 3`, :math:`y4 = 4`, and
    #        :math:`y5 = 5`""",
    #        order = 17)


    def __init__(self, **kwargs):
        """
        Initialise parameters for the Jansen Rit column, [JR_1995]_.

        """
        LOG.info("%s: initing..." % str(self))
        super(JansenRit, self).__init__(**kwargs)

        #self._state_variables = ["y0", "y1", "y2", "y3", "y4", "y5"]
        self._nvar = 6

        self.cvar = numpy.array([1, 2], dtype=numpy.int32)

        #TODO: adding an update_derived_parameters method to remove some of the
        #      redundant parameter multiplication in dfun should gain about 7%
        #      maybe not worth it... The three exp() kill us at ~90 times *
        #self.nu_max2 = None #2.0 * self.nu_max
        #self.Aa = None # self.A * self.a
        #self.Bb = None # self.B * self.b
        #self.aa = None # self.a**2
        #self.a2 = None # 2.0 * self.a
        #self.b2 = None # 2.0 * self.b
        #self.a_1J = None # self.a_1 * self.J
        #self.a_2J = None # self.a_2 * self.J
        #self.a_3J = None # self.a_3 * self.J
        #self.a_4J = None # self.a_4 * self.J

        LOG.debug('%s: inited.' % repr(self))


    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        The dynamic equations were taken from [JR_1995]_

        .. math::
            \dot{y_0} &= y_3 \\
            \dot{y_3} &= A a\,S[y_1 - y_2] - 2a\,y_3 - 2a^2\, y_0 \\
            \dot{y_1} &= y_4\\
            \dot{y_4} &= A a \,[p(t) + \alpha_2 J + S[\alpha_1 J\,y_0]+ c_0]
                        -2a\,y - a^2\,y_1 \\
            \dot{y_2} &= y_5 \\
            \dot{y_5} &= B b (\alpha_4 J\, S[\alpha_3 J \,y_0]) - 2 b\, y_5
                        - b^2\,y_2 \\
            S[v] &= \frac{2\, \nu_{max}}{1 + \exp^{r(v_0 - v)}}


        :math:`p(t)` can be any arbitrary function, including white noise or
        random numbers taken from a uniform distribution, representing a pulse
        density with an amplitude varying between 120 and 320

        For Evoked Potentials, a transient component of the input,
        representing the impulse density attribuable to a brief visual input is
        applied. Time should be in seconds.

        .. math::
            p(t) = q\,(\frac{t}{w})^n \, \exp{-\frac{t}{w}} \\
            q = 0.5 \\
            n = 7 \\
            w = 0.005 [s]


        """
        #NOTE: We could speed up this model by making the number below smaller,
        #      because the exp() dominate runtime, though we'd need to validate
        #      the trade-off in numerical accuracy...
        magic_exp_number = 709

        y0 = state_variables[0, :]
        y1 = state_variables[1, :]
        y2 = state_variables[2, :]
        y3 = state_variables[3, :]
        y4 = state_variables[4, :]
        y5 = state_variables[5, :]

        lrc = coupling[0, :] -  coupling[1, :]
        short_range_coupling =  local_coupling*(y1 -  y2)

        # NOTE: for local couplings
        # 0: pyramidal cells
        # 1: excitatory interneurons
        # 2: inhibitory interneurons
        # 0 -> 1,
        # 0 -> 2,
        # 1 -> 0,
        # 2 -> 0,

        #p_min = self.p_min
        #p_max = self.p_max
        #p = p_min + (p_max - p_min) * numpy.random.uniform()

        #NOTE: We were getting numerical overflow in the three exp()s below...
        temp = self.r * (self.v0 - (y1 - y2))
        sigm_y1_y2 = numpy.where(temp > magic_exp_number, 0.0, 2.0 * self.nu_max / (1.0 + numpy.exp(temp)))

        temp = self.r * (self.v0 - (self.a_1 * self.J * y0))
        sigm_y0_1 = numpy.where(temp > magic_exp_number, 0.0, 2.0 * self.nu_max / (1.0 + numpy.exp(temp)))

        temp = self.r * (self.v0 - (self.a_3 * self.J * y0))
        sigm_y0_3 = numpy.where(temp > magic_exp_number, 0.0, 2.0 * self.nu_max / (1.0 + numpy.exp(temp)))

        dy0 = y3
        dy3 = self.A * self.a * sigm_y1_y2 - 2.0 * self.a * y3 - self.a ** 2 * y0
        dy1 = y4
        dy4 = self.A * self.a * (self.mu + self.a_2 * self.J * sigm_y0_1 + lrc + short_range_coupling) - 2.0 * self.a * y4 - self.a ** 2 * y1
        dy2 = y5
        dy5 = self.B * self.b * (self.a_4 * self.J * sigm_y0_3) - 2.0 * self.b * y5 - self.b ** 2 * y2

        derivative = numpy.array([dy0, dy1, dy2, dy3, dy4, dy5])

        return derivative

    device_info = model_device_info(

        pars=['A', 'B', 'a', 'b', 'v0', 'nu_max', 'r', 'J', 'a_1', 'a_2', 'a_3', 'a_4',
              'p_min', 'p_max', 'mu'],

        kernel="""
        // read parameters
        float A      = P(0)
            , B      = P(1)
            , a      = P(2)
            , b      = P(3)
            , v0     = P(4)
            , nu_max = P(5)
            , r      = P(6)
            , J      = P(7)
            , a_1    = P(8)
            , a_2    = P(9)
            , a_3    = P(10)
            , a_4    = P(11)
            , p_min  = P(12)
            , p_max  = P(13)
            , mu     = P(14)

        // state variables
            , y0 = X(0)
            , y1 = X(1)
            , y2 = X(2)
            , y3 = X(3)
            , y4 = X(4)
            , y5 = X(5)
            , y6 = X(6)

        // aux variables
            , c_0 = I(0)
            , sigm_y1_y2 = 2.0 * nu_max / (1.0 + exp( r * (v0 - (y1 - y2)) ))
            , sigm_y0_1  = 2.0 * nu_max / (1.0 + exp( r * (v0 - (a_1 * J * y0))))
            , sigm_y0_3  = 2.0 * nu_max / (1.0 + exp( r * (v0 - (a_3 * J * y0))));

        // derivatives
        DX(0) = y3;
        DX(1) = y4;
        DX(2) = y5;
        DX(3) = A * a * sigm_y1_y2 - 2.0 * a * y3 - a*a*y0;
        DX(4) = A * a * (mu + a_2 * J * sigm_y0_1 + c_0) - 2.0 * a * y4 - a*a*y1;
        DX(5) = B * b * (a_4 + J * sigm_y0_3) - 2.0 * b * y5 - b*b*y2;
        """
    )



class JRFast(JansenRit):
    """
    This is an optimized version of the above JansenRit model, using numexpr and
    constant memory (as far as is obvious).

    Note that it caches parameters and derivative arrays on the first call to
    the dfun method, so if you change the number of nodes or the parameters, you
    need to invalidate the cache by setting the invalid_dfun_cache attribute to
    True.

    """

    invalid_dfun_cache = True

    #@profile
    def dfun(self, y, coupling, local_coupling=0.0, ev=numexpr.evaluate):

        if self.invalid_dfun_cache:
            self.dy = y.copy() * 0.0
            self.y1m2 = y[1].copy()
            self.dfunlocals = {}
            for k in ['nu_max', 'r', 'v0', 'a_1', 'J', 'a_3', 'A', 'a', 'mu',
                    'a_2', 'B', 'b', 'a_4']:
                self.dfunlocals[k] = getattr(self, k)
            self.dfunglobals = {}
            self.invalid_dfun_cache = False

        l = self.dfunlocals
        g = self.dfunglobals

        l['y0'], l['y1'], l['y2'], l['y3'], l['y4'], l['y5'] = y

        l['c0'], l['c1'] = coupling

        # self.cvar = numpy.array([1, 2], dtype=numpy.int32)
        self.y1m2[:] = y[1]
        self.y1m2 -= y[2]
        l['lc'] = local_coupling * self.y1m2

        self.dy[:3] = y[3:]

        ev('A * a * (2.0 * nu_max / (1.0 + exp(r * (v0 - (y1 - y2))))) - 2.0 * a * y3 - a ** 2 * y0', l, g, out=self.dy[3], casting='no')
        ev('A * a * (mu + a_2 * J * (2.0 * nu_max / (1.0 + exp(r * (v0 - (a_1 * J * y0))))) + (c0 - c1) + lc) - 2.0 * a * y4 - a ** 2 * y1', l, g, out=self.dy[4], casting='no')
        ev('B * b * (a_4 * J * (2.0 * nu_max / (1.0 + exp(r * (v0 - (a_3 * J * y0)))))) - 2.0 * b * y5 - b ** 2 * y2', l, g, out=self.dy[5], casting='no')

        return self.dy



class ZetterbergJansen(Model):
    """
    The Jansen and Rit is a biologically inspired mathematical framework
    originally conceived to simulate the spontaneous electrical activity of
    neuronal assemblies, with a particular focus on alpha activity, for instance,
    as measured by EEG. Later on, it was discovered that in addition to alpha
    activity, this model was also able to simulate evoked potentials.

    .. [JB_1995]  Jansen, B., H. and Rit V., G., *Electroencephalogram and
        visual evoked potential generation in a mathematical model of
        coupled cortical columns*, Biological Cybernetics (73) 357:366, 1995.

    .. [JB_1993] Jansen, B., Zouridakis, G. and Brandt, M., *A
        neurophysiologically-based mathematical model of flash visual evoked
        potentials*

    .. [M_2007] Moran

    .. [S_2010] Spiegler

    .. [A_2012] Auburn

    .. figure :: img/ZetterbergJansen_01_mode_0_pplane.svg
        :alt: Jansen and Rit phase plane

    .. automethod:: ZetterbergJansen.__init__
    .. automethod:: ZetterbergJansen.dfun

    """

    _ui_name = "Zetterberg-Jansen"
    ui_configurable_parameters = ['He', 'Hi', 'ke', 'ki', 'e0', 'rho_2', 'rho_1', 'gamma_1',
                                  'gamma_2', 'gamma_3', 'gamma_4', 'gamma_5', 'P', 'U', 'Q']

    #Define traited attributes for this model, these represent possible kwargs.
    He = arrays.FloatArray(
        label=":math:`H_e`",
        default=numpy.array([3.25]),
        range=basic.Range(lo=2.6, hi=9.75, step=0.05),
        doc="""Maximum amplitude of EPSP [mV]. Also called average synaptic gain.""",
        order=1)

    Hi = arrays.FloatArray(
        label=":math:`H_i`",
        default=numpy.array([22.0]),
        range=basic.Range(lo=17.6, hi=110.0, step=0.2),
        doc="""Maximum amplitude of IPSP [mV]. Also called average synaptic gain.""",
        order=2)

    ke = arrays.FloatArray(
        label=r":math:`\kappa_e`",
        default=numpy.array([0.1]),
        range=basic.Range(lo=0.05, hi=0.15, step=0.01),
        doc="""Reciprocal of the time constant of passive membrane and all
        other spatially distributed delays in the dendritic network [ms^-1].
        Also called average synaptic time constant.""",
        order=3)

    ki = arrays.FloatArray(
        label=r":math:`\kappa_i`",
        default=numpy.array([0.05]),
        range=basic.Range(lo=0.025, hi=0.075, step=0.005),
        doc="""Reciprocal of the time constant of passive membrane and all
        other spatially distributed delays in the dendritic network [ms^-1].
        Also called average synaptic time constant.""",
        order=4)


    e0 = arrays.FloatArray(
        label=r":math:`e_0`",
        default=numpy.array([0.0025]),
        range=basic.Range(lo=0.00125, hi=0.00375, step=0.00001),
        doc="""Half of the maximum population mean firing rate [ms^-1].""",
        order=6)


    rho_2 = arrays.FloatArray(
        label=r":math:`\rho_2`",
        default=numpy.array([6.0]),
        range=basic.Range(lo=3.12, hi=10.0, step=0.02),
        doc="""Firing threshold (PSP) for which a 50% firing rate is achieved.
        In other words, it is the value of the average membrane potential
        corresponding to the inflection point of the sigmoid [mV]. Population mean firing threshold.""",
        order=5)

    rho_1 = arrays.FloatArray(
        label=r":math:`\rho_1`",
        default=numpy.array([0.56]),
        range=basic.Range(lo=0.28, hi=0.84, step=0.01),
        doc="""Steepness of the sigmoidal transformation [mV^-1].""",
        order=7)

    gamma_1 = arrays.FloatArray(
        label=r":math:`\gamma_1`",
        default=numpy.array([135.0]),
        range=basic.Range(lo=65.0, hi=1350.0, step=5.),
        doc="""Average number of synapses between populations (pyramidal to stellate).""",
        order=8)

    gamma_2 = arrays.FloatArray(
        label=r":math:`\gamma_2`",
        default=numpy.array([108.]),
        range=basic.Range(lo=0.0, hi=200, step=10.0),
        doc="""Average number of synapses between populations (stellate to pyramidal).""",
        order=9)

    gamma_3 = arrays.FloatArray(
        label=r":math:`\gamma_3`",
        default=numpy.array([33.75]),
        range=basic.Range(lo=0.0, hi=200, step=10.0),
        doc="""Connectivity constant (pyramidal to interneurons)""",
        order=10)

    gamma_4 = arrays.FloatArray(
        label=r":math:`\gamma_4`",
        default=numpy.array([33.75]),
        range=basic.Range(lo=0.0, hi=200, step=10.0),
        doc="""Connectivity constant (interneurons to pyramidal)""",
        order=11)


    gamma_5 = arrays.FloatArray(
        label=r":math:`\gamma_5`",
        default=numpy.array([15]),
        range=basic.Range(lo=0.0, hi=100, step=10.0),
        doc="""Connectivity constant (interneurons to interneurons)""",
        order=12)

    gamma_1T = arrays.FloatArray(
        label=r":math:`\gamma_{1T}`",
        default=numpy.array([1.0]),
        range=basic.Range(lo=0.0, hi=1000.0, step=5.),
        doc="""Coupling factor from the extrinisic input to the spiny stellate population.""",
        order=17)

    gamma_3T = arrays.FloatArray(
        label=r":math:`\gamma_{3T}`",
        default=numpy.array([1.0]),
        range=basic.Range(lo=0.0, hi=1000.0, step=5.),
        doc="""Coupling factor from the extrinisic input to the pyramidal population.""",
        order=18)

    gamma_2T = arrays.FloatArray(
        label=r":math:`\gamma_{2T}`",
        default=numpy.array([1.0]),
        range=basic.Range(lo=0.0, hi=1000.0, step=5.),
        doc="""Coupling factor from the extrinisic input to the inhibitory population.""",
        order=19)

    P = arrays.FloatArray(
        label=":math:`P`",
        default=numpy.array([0.12]),
        range=basic.Range(lo=0.0, hi=0.350, step=0.01),
        doc="""Maximum firing rate to the pyramidal population [ms^-1].
        (External stimulus. Constant intensity.Entry point for coupling.)""",
        order=13)

    U = arrays.FloatArray(
        label=":math:`U`",
        default=numpy.array([0.12]),
        range=basic.Range(lo=0.0, hi=0.350, step=0.01),
        doc="""Maximum firing rate to the stellate population [ms^-1].
        (External stimulus. Constant intensity.Entry point for coupling.)""",
        order=14)

    Q = arrays.FloatArray(
        label=":math:`Q`",
        default=numpy.array([0.12]),
        range=basic.Range(lo=0.0, hi=0.350, step=0.01),
        doc="""Maximum firing rate to the interneurons population [ms^-1].
        (External stimulus. Constant intensity.Entry point for coupling.)""",
        order=15)

    #Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = basic.Dict(
        label="State Variable ranges [lo, hi]",
        default={"v1": numpy.array([-100.0, 100.0]),
                 "y1": numpy.array([-500.0, 500.0]),
                 "v2": numpy.array([-100.0, 50.0]),
                 "y2": numpy.array([-100.0, 6.0]),
                 "v3": numpy.array([-100.0, 6.0]),
                 "y3": numpy.array([-100.0, 6.0]),
                 "v4": numpy.array([-100.0, 20.0]),
                 "y4": numpy.array([-100.0, 20.0]),
                 "v5": numpy.array([-100.0, 20.0]),
                 "y5": numpy.array([-500.0, 500.0]),
                 "v6": numpy.array([-100.0, 20.0]),
                 "v7": numpy.array([-100.0, 20.0]),},
        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current
        parameters, it is used as a mechanism for bounding random inital
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""",
        order=16)

    variables_of_interest = basic.Enumerate(
        label="Variables watched by Monitors",
        options=["v1", "y1", "v2", "y2", "v3", "y3", "v4", "y4", "v5", "y5", "v6", "v7"],
        default=["v6", "v7", "v2", "v3", "v4", "v5"],
        select_multiple=True,
        doc="""This represents the default state-variables of this Model to be
                                    monitored. It can be overridden for each Monitor if desired. The
                                    corresponding state-variable indices for this model are :math:`v_6 = 0`,
                                    :math:`v_7 = 1`, :math:`v_2 = 2`, :math:`v_3 = 3`, :math:`v_4 = 4`, and
                                    :math:`v_5 = 5`""",
        order=42)


    def __init__(self, **kwargs):
        """
        Initialise parameters for the Zetterberg-Jansen model

        """
        LOG.info("%s: initing..." % str(self))
        super(ZetterbergJansen, self).__init__(**kwargs)

        self._nvar = 12

        self.cvar = numpy.array([10], dtype=numpy.int32)


        self.Heke = None # self.He * self.ke
        self.Hiki = None # self.Hi * self.ki
        self.ke_2 = None # 2 * self.ke
        self.ki_2 = None # 2 * self.ki
        self.keke = None # self.ke **2
        self.kiki = None # self.ki **2

        LOG.debug('%s: inited.' % repr(self))


    def configure(self):
        """  """
        super(ZetterbergJansen, self).configure()
        self.update_derived_parameters()


    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""

        .. math:: do something


        """

        magic_exp_number = 709

        v1 = state_variables[0, :]
        y1 = state_variables[1, :]
        v2 = state_variables[2, :]
        y2 = state_variables[3, :]
        v3 = state_variables[4, :]
        y3 = state_variables[5, :]
        v4 = state_variables[6, :]
        y4 = state_variables[7, :]
        v5 = state_variables[8, :]
        y5 = state_variables[9, :]
        v6 = state_variables[10, :]
        v7 = state_variables[11, :]

        # NOTE: long_range_coupling term: coupling variable is v6 . EQUATIONS
        #       ASSUME linear coupling is used. 'coupled_input' represents a rate. It
        #       is very likely that coeffs gamma_xT should be independent for each of the
        #       terms considered as extrinsic input (P, Q, U) (long range coupling) (local coupling)
        #       and noise.

        coupled_input =  self.sigma_fun(coupling[0, :] + local_coupling * v6)

        # exc input to the excitatory interneurons
        dv1 = y1
        dy1 = self.Heke * (self.gamma_1 * self.sigma_fun(v2 - v3) + self.gamma_1T * (self.U + coupled_input )) - self.ke_2 * y1 - self.keke * v1
        # exc input to the pyramidal cells
        dv2 = y2
        dy2 = self.Heke * (self.gamma_2 * self.sigma_fun(v1)      + self.gamma_2T * (self.P + coupled_input )) - self.ke_2 * y2 - self.keke * v2
        # inh input to the pyramidal cells
        dv3 = y3
        dy3 = self.Hiki * (self.gamma_4 * self.sigma_fun(v4 - v5)) - self.ki_2 * y3 - self.kiki * v3
        dv4 = y4
        # exc input to the inhibitory interneurons
        dy4 = self.Heke * (self.gamma_3 * self.sigma_fun(v2 - v3) + self.gamma_3T * (self.Q + coupled_input)) - self.ke_2 * y4 - self.keke * v4
        dv5 = y5
        # inh input to the inhibitory interneurons
        dy5 = self.Hiki * (self.gamma_5 * self.sigma_fun(v4 - v5)) - self.ki_2 * y5 - self.keke * v5
        # aux variables (the sum gathering the postsynaptic inh & exc potentials)
        # pyramidal cells
        dv6 = y2 - y3
        # inhibitory cells
        dv7 = y4 - y5

        derivative = numpy.array([dv1, dy1, dv2, dy2, dv3, dy3, dv4, dy4, dv5, dy5, dv6, dv7])

        return derivative

    def sigma_fun(self, sv):
        """
        Neuronal activation function. This sigmoidal function
        increases from 0 to Q_max as "sv" increases.
        sv represents a membrane potential state variable (V).

        """
        #HACKERY: Hackery for exponential s that blow up.
        # Set to inf, so the result will be effectively zero.
        magic_exp_number = 709
        temp = self.rho_1 * (self.rho_2 - sv)
        temp = numpy.where(temp > magic_exp_number, numpy.inf, temp)
        sigma_v = (2* self.e0) / (1 + numpy.exp(temp))

        return sigma_v


    def update_derived_parameters(self):
        self.Heke = self.He * self.ke
        self.Hiki = self.Hi * self.ki
        self.ke_2 = 2 * self.ke
        self.ki_2 = 2 * self.ki
        self.keke = self.ke**2
        self.kiki = self.ki**2



class Generic2dOscillator(Model):
    r"""
    The Generic2dOscillator model is a generic dynamic system with two state
    variables. The dynamic equations of this model are composed of two ordinary
    differential equations comprising two nullclines. The first nullcline is a
    cubic function as it is found in most neuron and population models; the
    second nullcline is arbitrarily configurable as a polynomial function up to
    second order. The manipulation of the latter nullcline's parameters allows
    to generate a wide range of different behaviours.

    Equations:

    .. math::
                \dot{V} &= d \, \tau (-f V^3 + e V^2 + g V + \alpha W + \gamma I), \\
                \dot{W} &= \dfrac{d}{\tau}\,\,(c V^2 + b V - \beta W + a),

    See:


        .. [FH_1961] FitzHugh, R., *Impulses and physiological states in theoretical
            models of nerve membrane*, Biophysical Journal 1: 445, 1961.

        .. [Nagumo_1962] Nagumo et.al, *An Active Pulse Transmission Line Simulating
            Nerve Axon*, Proceedings of the IRE 50: 2061, 1962.

        .. [SJ_2011] Stefanescu, R., Jirsa, V.K. *Reduced representations of
            heterogeneous mixed neural networks with synaptic coupling*.
            Physical Review E, 83, 2011.

        .. [SJ_2010]	Jirsa VK, Stefanescu R.  *Neural population modes capture
            biologically realistic large-scale network dynamics*. Bulletin of
            Mathematical Biology, 2010.

        .. [SJ_2008_a] Stefanescu, R., Jirsa, V.K. *A low dimensional description
            of globally coupled heterogeneous neural networks of excitatory and
            inhibitory neurons*. PLoS Computational Biology, 4(11), 2008).


    The model's (:math:`V`, :math:`W`) time series and phase-plane its nullclines
    can be seen in the figure below.

    The model with its default parameters exhibits FitzHugh-Nagumo like dynamics.

    +---------------------------+
    |  Table 1                  |
    +--------------+------------+
    |  EXCITABLE CONFIGURATION  |
    +--------------+------------+
    |Parameter     |  Value     |
    +==============+============+
    | a            |     -2.0   |
    +--------------+------------+
    | b            |    -10.0   |
    +--------------+------------+
    | c            |      0.0   |
    +--------------+------------+
    | d            |      0.02  |
    +--------------+------------+
    | I            |      0.0   |
    +--------------+------------+
    |  limit cycle if a is 2.0  |
    +---------------------------+


    +---------------------------+
    |   Table 2                 |
    +--------------+------------+
    |   BISTABLE CONFIGURATION  |
    +--------------+------------+
    |Parameter     |  Value     |
    +==============+============+
    | a            |      1.0   |
    +--------------+------------+
    | b            |      0.0   |
    +--------------+------------+
    | c            |     -5.0   |
    +--------------+------------+
    | d            |      0.02  |
    +--------------+------------+
    | I            |      0.0   |
    +--------------+------------+
    | monostable regime:        |
    | fixed point if Iext=-2.0  |
    | limit cycle if Iext=-1.0  |
    +---------------------------+


    +---------------------------+
    |  Table 3                  |
    +--------------+------------+
    |  EXCITABLE CONFIGURATION  |
    +--------------+------------+
    |  (similar to Morris-Lecar)|
    +--------------+------------+
    |Parameter     |  Value     |
    +==============+============+
    | a            |      0.5   |
    +--------------+------------+
    | b            |      0.6   |
    +--------------+------------+
    | c            |     -4.0   |
    +--------------+------------+
    | d            |      0.02  |
    +--------------+------------+
    | I            |      0.0   |
    +--------------+------------+
    | excitable regime if b=0.6 |
    | oscillatory if b=0.4      |
    +---------------------------+


    +---------------------------+
    |  Table 4                  |
    +--------------+------------+
    |  GhoshetAl,  2008         |
    |  KnocketAl,  2009         |
    +--------------+------------+
    |Parameter     |  Value     |
    +==============+============+
    | a            |    1.05    |
    +--------------+------------+
    | b            |   -1.00    |
    +--------------+------------+
    | c            |    0.0     |
    +--------------+------------+
    | d            |    0.1     |
    +--------------+------------+
    | I            |    0.0     |
    +--------------+------------+
    | alpha        |    1.0     |
    +--------------+------------+
    | beta         |    0.2     |
    +--------------+------------+
    | gamma        |    -1.0    |
    +--------------+------------+
    | e            |    0.0     |
    +--------------+------------+
    | g            |    1.0     |
    +--------------+------------+
    | f            |    1/3     |
    +--------------+------------+
    | tau          |    1.25    |
    +--------------+------------+
    |                           |
    |  frequency peak at 10Hz   |
    |                           |
    +---------------------------+


    +---------------------------+
    |  Table 5                  |
    +--------------+------------+
    |  SanzLeonetAl  2013       |
    +--------------+------------+
    |Parameter     |  Value     |
    +==============+============+
    | a            |    - 0.5   |
    +--------------+------------+
    | b            |    -10.0   |
    +--------------+------------+
    | c            |      0.0   |
    +--------------+------------+
    | d            |      0.02  |
    +--------------+------------+
    | I            |      0.0   |
    +--------------+------------+
    |                           |
    |  intrinsic frequency is   |
    |  approx 10 Hz             |
    |                           |
    +---------------------------+

    NOTE: This regime, if I = 2.1, is called subthreshold regime.
    Unstable oscillations appear through a subcritical Hopf bifurcation.


    .. figure :: img/Generic2dOscillator_01_mode_0_pplane.svg
    .. _phase-plane-Generic2D:
        :alt: Phase plane of the generic 2D population model with (V, W)

        The (:math:`V`, :math:`W`) phase-plane for the generic 2D population
        model for default parameters. The dynamical system has an equilibrium
        point.

    .. #Currently there seems to be a clash between traits and autodoc, autodoc
    .. #can't find the methods of the class, the class specific names below get
    .. #us around this...
    .. automethod:: Generic2dOscillator.__init__
    .. automethod:: Generic2dOscillator.dfun

    """

    _ui_name = "Generic 2d Oscillator"
    ui_configurable_parameters = ['tau', 'a', 'b', 'c', 'I', 'd', 'e', 'f', 'g', 'alpha', 'beta', 'gamma']

    #Define traited attributes for this model, these represent possible kwargs.
    tau = arrays.FloatArray(
        label=r":math:`\tau`",
        default=numpy.array([1.0]),
        range=basic.Range(lo=1.0, hi=5.0, step=0.01),
        doc="""A time-scale hierarchy can be introduced for the state
        variables :math:`V` and :math:`W`. Default parameter is 1, which means
        no time-scale hierarchy.""",
        order=1)

    I = arrays.FloatArray(
        label=":math:`I_{ext}`",
        default=numpy.array([0.0]),
        range=basic.Range(lo=-5.0, hi=5.0, step=0.01),
        doc="""Baseline shift of the cubic nullcline""",
        order=2)

    a = arrays.FloatArray(
        label=":math:`a`",
        default=numpy.array([-2.0]),
        range=basic.Range(lo=-5.0, hi=5.0, step=0.01),
        doc="""Vertical shift of the configurable nullcline""",
        order=3)

    b = arrays.FloatArray(
        label=":math:`b`",
        default=numpy.array([-10.0]),
        range=basic.Range(lo=-20.0, hi=15.0, step=0.01),
        doc="""Linear slope of the configurable nullcline""",
        order=4)

    c = arrays.FloatArray(
        label=":math:`c`",
        default=numpy.array([0.0]),
        range=basic.Range(lo=-10.0, hi=10.0, step=0.01),
        doc="""Parabolic term of the configurable nullcline""",
        order=5)

    d = arrays.FloatArray(
        label=":math:`d`",
        default=numpy.array([0.02]),
        range=basic.Range(lo=0.0001, hi=1.0, step=0.0001),
        doc="""Temporal scale factor. Warning: do not use it unless
        you know what you are doing and know about time tides.""",
        order=13)

    e = arrays.FloatArray(
        label=":math:`e`",
        default=numpy.array([3.0]),
        range=basic.Range(lo=-5.0, hi=5.0, step=0.0001),
        doc="""Coefficient of the quadratic term of the cubic nullcline.""",
        order=6)

    f = arrays.FloatArray(
        label=":math:`f`",
        default=numpy.array([1.0]),
        range=basic.Range(lo=-5.0, hi=5.0, step=0.0001),
        doc="""Coefficient of the cubic term of the cubic nullcline.""",
        order=7)

    g = arrays.FloatArray(
        label=":math:`g`",
        default=numpy.array([0.0]),
        range=basic.Range(lo=-5.0, hi=5.0, step=0.5),
        doc="""Coefficient of the linear term of the cubic nullcline.""",
        order=8)

    alpha = arrays.FloatArray(
        label=r":math:`\alpha`",
        default=numpy.array([1.0]),
        range=basic.Range(lo=-5.0, hi=5.0, step=0.0001),
        doc="""Constant parameter to scale the rate of feedback from the
            slow variable to the fast variable.""",
        order=9)

    beta = arrays.FloatArray(
        label=r":math:`\beta`",
        default=numpy.array([1.0]),
        range=basic.Range(lo=-5.0, hi=5.0, step=0.0001),
        doc="""Constant parameter to scale the rate of feedback from the
            slow variable to itself""",
        order=10)

    # This parameter is basically a hack to avoid having a negative lower boundary in the global coupling strength.
    gamma = arrays.FloatArray(
        label=r":math:`\gamma`",
        default=numpy.array([1.0]),
        range=basic.Range(lo=-1.0, hi=1.0, step=0.1),
        doc="""Constant parameter to reproduce FHN dynamics where
               excitatory input currents are negative.
               It scales both I and the long range coupling term.""",
        order=13)

    #Informational attribute, used for phase-plane and initial()
    state_variable_range = basic.Dict(
        label="State Variable ranges [lo, hi]",
        default={"V": numpy.array([-2.0, 4.0]),
                 "W": numpy.array([-6.0, 6.0])},
        doc="""The values for each state-variable should be set to encompass
            the expected dynamic range of that state-variable for the current
            parameters, it is used as a mechanism for bounding random initial
            conditions when the simulation isn't started from an explicit
            history, it is also provides the default range of phase-plane plots.""",
        order=11)

    #    variables_of_interest = arrays.IntegerArray(
    #        label = "Variables watched by Monitors.",
    #        range = basic.Range(lo = 0.0, hi = 2.0, step = 1.0),
    #        default = numpy.array([0], dtype=numpy.int32),
    #        doc = """This represents the default state-variables of this Model to be
    #        monitored. It can be overridden for each Monitor if desired. The
    #        corresponding state-variable indices for this model are :math:`V = 0`
    #        and :math:`W = 1`""",
    #        order = 7)

    variables_of_interest = basic.Enumerate(
        label="Variables watched by Monitors",
        options=["V", "W"],
        default=["V", ],
        select_multiple=True,
        doc="""This represents the default state-variables of this Model to be
                                        monitored. It can be overridden for each Monitor if desired. The
                                        corresponding state-variable indices for this model are :math:`V = 0`
                                        and :math:`W = 1`.""",
        order=12)


    def __init__(self, **kwargs):
        """
        May need to put kwargs back if we can't get them from trait...

        """

        LOG.info("%s: initing..." % str(self))

        super(Generic2dOscillator, self).__init__(**kwargs)

        #self._state_variables = ["V", "W"]
        self._nvar = 2
        self.cvar = numpy.array([0], dtype=numpy.int32)

        LOG.debug("%s: inited." % repr(self))


    def dfun(self, state_variables, coupling, local_coupling=0.0, ev=numexpr.evaluate):
        r"""
        The two state variables :math:`V` and :math:`W` are typically considered
        to represent a function of the neuron's membrane potential, such as the
        firing rate or dendritic currents, and a recovery variable, respectively.
        If there is a time scale hierarchy, then typically :math:`V` is faster
        than :math:`W` corresponding to a value of :math:`\tau` greater than 1.

        The equations of the generic 2D population model read

        .. math::
                \dot{V} &= d \, \tau (-f V^3 + e V^2 + g V + \alpha W + \gamma I), \\
                \dot{W} &= \dfrac{d}{\tau}\,\,(c V^2 + b V - \beta W + a),

        where external currents :math:`I` provide the entry point for local,
        long-range connectivity and stimulation.

        """

        V = state_variables[0, :]
        W = state_variables[1, :]

        #[State_variables, nodes]
        c_0 = coupling[0, :]

        tau = self.tau
        I = self.I
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        e = self.e
        f = self.f
        g = self.g
        beta = self.beta
        alpha = self.alpha
        gamma = self.gamma

        lc_0 = local_coupling * V


        #if not hasattr(self, 'derivative'):
        #    self.derivative = numpy.empty((2,)+V.shape)

        ## numexpr
        dV = ev('d * tau * (alpha * W - f * V**3 + e * V**2 + g * V + gamma * I + gamma *c_0 + lc_0)')
        dW = ev('d * (a + b * V + c * V**2 - beta * W) / tau')

        ## regular ndarray operation
        ##dV = tau * (W - 0.5* V**3.0 + 3.0 * V**2 + I + c_0 + lc_0)
        ##dW = d * (a + b * V + c * V**2 - W) / tau

        self.derivative = numpy.array([dV, dW])

        return self.derivative

    device_info = model_device_info(
        pars=['tau', 'a', 'b', 'c', 'd', 'I'],
        kernel="""

        // read parameters
        float tau  = P(0)
            , a    = P(1)
            , b    = P(2)
            , c    = P(3)
            , d    = P(4)
            , I    = P(5)

        // state variables
            , v    = X(0)
            , w    = X(1)

        // aux variables
            , c_0  = I(0)   ;

        // derivatives
        DX(0) = d * (tau * (w - v*v*v + 3.0*v*v + I + c_0));
        DX(1) = d * ((a + b*v + c*v*v - w) / tau);
        """
    )



class LarterBreakspear(Model):
    r"""
    A modified Morris-Lecar model that includes a third equation which simulates
    the effect of a population of inhibitory interneurons synapsing on
    the pyramidal cells.

    .. [Larteretal_1999] Larter et.al. *A coupled ordinary differential equation
        lattice model for the simulation of epileptic seizures.* Chaos. 9(3):
        795, 1999.

    .. [Breaksetal_2003_a] Breakspear, M.; Terry, J. R. & Friston, K. J.  *Modulation of excitatory
        synaptic coupling facilitates synchronization and complex dynamics in an
        onlinear model of neuronal dynamics*. Neurocomputing 52–54 (2003).151–158

    .. [Breaksetal_2003_b] M. J. Breakspear et.al. *Modulation of excitatory
        synaptic coupling facilitates synchronization and complex dynamics in a
        biophysical model of neuronal dynamics.* Network: Computation in Neural
        Systems 14: 703-732, 2003.

    .. [Honeyetal_2007] Honey, C.; Kötter, R.; Breakspear, M. & Sporns, O. * Network structure of
        cerebral cortex shapes functional connectivity on multiple time scales*. (2007)
        PNAS, 104, 10240

    .. [Honeyetal_2009] Honey, C. J.; Sporns, O.; Cammoun, L.; Gigandet, X.; Thiran, J. P.; Meuli,
        R. & Hagmann, P. *Predicting human resting-state functional connectivity
        from structural connectivity.* (2009), PNAS, 106, 2035-2040

    .. [Alstottetal_2009] Alstott, J.; Breakspear, M.; Hagmann, P.; Cammoun, L. & Sporns, O.
        *Modeling the impact of lesions in the human brain*. (2009)),  PLoS Comput Biol, 5, e1000408

    Equations and default parameters are taken from [Breaksetal_2003_b]_.
    All equations and parameters are non-dimensional and normalized.
    For values of d_v  < 0.55, the dynamics of a single column settles onto a
    solitary fixed point attractor.


    Parameters used for simulations in [Breaksetal_2003_a]_ Table 1. Page 153.
    Two nodes were coupled. C=0.1

    +---------------------------+
    |          Table 1          |
    +--------------+------------+
    |Parameter     |  Value     |
    +==============+============+
    | I            |      0.3   |
    +--------------+------------+
    | a_ee         |      0.4   |
    +--------------+------------+
    | a_ei         |      0.1   |
    +--------------+------------+
    | a_ie         |      1.0   |
    +--------------+------------+
    | a_ne         |      1.0   |
    +--------------+------------+
    | a_ni         |      0.4   |
    +--------------+------------+
    | r_NMDA       |      0.2   |
    +--------------+------------+
    | delta        |      0.001 |
    +--------------+------------+
    |   Breakspear et al. 2003  |
    +---------------------------+


    +---------------------------+
    |          Table 2          |
    +--------------+------------+
    |Parameter     |  Value     |
    +==============+============+
    | gK           |      2.0   |
    +--------------+------------+
    | gL           |      0.5   |
    +--------------+------------+
    | gNa          |      6.7   |
    +--------------+------------+
    | gCa          |      1.0   |
    +--------------+------------+
    | a_ne         |      1.0   |
    +--------------+------------+
    | a_ni         |      0.4   |
    +--------------+------------+
    | a_ee         |      0.36  |
    +--------------+------------+
    | a_ei         |      2.0   |
    +--------------+------------+
    | a_ie         |      2.0   |
    +--------------+------------+
    | VK           |     -0.7   |
    +--------------+------------+
    | VL           |     -0.5   |
    +--------------+------------+
    | VNa          |      0.53  |
    +--------------+------------+
    | VCa          |      1.0   |
    +--------------+------------+
    | phi          |      0.7   |
    +--------------+------------+
    | b            |      0.1   |
    +--------------+------------+
    | I            |      0.3   |
    +--------------+------------+
    | r_NMDA       |      0.25  |
    +--------------+------------+
    | C            |      0.1   |
    +--------------+------------+
    | TCa          |     -0.01  |
    +--------------+------------+
    | d_Ca         |      0.15  |
    +--------------+------------+
    | TK           |      0.0   |
    +--------------+------------+
    | d_K          |      0.3   |
    +--------------+------------+
    | VT           |      0.0   |
    +--------------+------------+
    | ZT           |      0.0   |
    +--------------+------------+
    | TNa          |      0.3   |
    +--------------+------------+
    | d_Na         |      0.15  |
    +--------------+------------+
    | d_V          |      0.65  |
    +--------------+------------+
    | d_Z          |      d_V   |
    +--------------+------------+
    | QV_max       |      1.0   |
    +--------------+------------+
    | QZ_max       |      1.0   |
    +--------------+------------+
    |   Alstott et al. 2009     |
    +---------------------------+


    NOTES about parameters

    :math:`\delta_V` : for :math:`\delta_V` < 0.55, in an uncoupled network,
    the system exhibits fixed point dynamics; for 0.55 < :math:`\delta_V` < 0.59,
    limit cycle attractors; and for :math:`\delta_V` > 0.59 chaotic attractors
    (eg, d_V=0.6,aee=0.5,aie=0.5, gNa=0, Iext=0.165)

    :math:`\delta_Z`
    this parameter might be spatialized: ones(N,1).*0.65 + modn*(rand(N,1)-0.5);

    :math:`C`
    The long-range coupling :math:`\delta_C` is ‘weak’ in the sense that
    the model is well behaved for parameter values for which C < a_ee and C << a_ie.



    .. figure :: img/LarterBreakspear_01_mode_0_pplane.svg
            :alt: Larter-Breaskpear phase plane (V, W)

            The (:math:`V`, :math:`W`) phase-plane for the Larter-Breakspear model.

    .. automethod:: LarterBreakspear.__init__

    Dynamic equations:

    .. math::
            \dot{V}_k & = - (g_{Ca} + (1 - C) \, r_{NMDA} \, a_{ee} \, Q_V + C \, r_{NMDA} \, a_{ee} \, \langle Q_V\rangle^{k}) \, m_{Ca} \, (V - VCa) \\
                           & \,\,- g_K \, W \, (V - VK) -  g_L \, (V - VL) \\
                           & \,\,- (g_{Na} \, m_{Na} + (1 - C) \, a_{ee} \, Q_V + C \, a_{ee} \, \langle Q_V\rangle^{k}) \,(V - VNa) \\
                           & \,\,- a_{ei} \, Z \, Q_Z + a_{ne} \, I, \\
                           & \\
            \dot{W}_k & = \phi \, \dfrac{m_K - W}{\tau_{K}},\\
                           & \nonumber\\
            \dot{Z}_k &= b (a_{ni}\, I + a_{ei}\,V\,Q_V),\\
            Q_{V}   &= Q_{V_{max}} \, (1 + \tanh\left(\dfrac{V_{k} - VT}{\delta_{V}}\right)),\\
            Q_{Z}   &= Q_{Z_{max}} \, (1 + \tanh\left(\dfrac{Z_{k} - ZT}{\delta_{Z}}\right)),

        See Equations (7), (3), (6) and (2) respectively in [Breaksetal_2003_a]_.
        Pag: 705-706

    """

    _ui_name = "Larter-Breakspear"
    ui_configurable_parameters = ['gCa', 'gK', 'gL', 'phi', 'gNa', 'TK', 'TCa',
                                  'TNa', 'VCa', 'VK', 'VL', 'VNa', 'd_K', 'tau_K',
                                  'd_Na', 'd_Ca', 'aei', 'aie', 'b', 'C', 'ane',
                                  'ani', 'aee', 'Iext', 'rNMDA', 'VT', 'd_V', 'ZT',
                                  'd_Z', 'QV_max', 'QZ_max']

    #Define traited attributes for this model, these represent possible kwargs.
    gCa = arrays.FloatArray(
        label = ":math:`g_{Ca}`",
        default = numpy.array([1.1]),
        range = basic.Range(lo = 0.9, hi = 1.5, step = 0.1),
        doc = """Conductance of population of Ca++ channels.""")

    gK = arrays.FloatArray(
        label = ":math:`g_{K}`",
        default = numpy.array([2.0]),
        range = basic.Range(lo = 1.95, hi= 2.05, step = 0.025),
        doc = """Conductance of population of K channels.""")

    gL = arrays.FloatArray(
        label = ":math:`g_{L}`",
        default = numpy.array([0.5]),
        range = basic.Range(lo = 0.45 , hi = 0.55, step = 0.05),
        doc = """Conductance of population of leak channels.""")

    phi = arrays.FloatArray(
        label = r":math:`\phi`",
        default = numpy.array([0.7]),
        range = basic.Range(lo = 0.3, hi = 0.9, step = 0.1),
        doc = """Temperature scaling factor.""")

    gNa = arrays.FloatArray(
        label = ":math:`g_{Na}`",
        default = numpy.array([6.7]),
        range = basic.Range(lo = 0.0, hi = 10.0, step = 0.1),
        doc = """Conductance of population of Na channels.""")

    TK = arrays.FloatArray(
        label = ":math:`T_{K}`",
        default = numpy.array([0.0]),
        range = basic.Range(lo = 0.0, hi = 0.0001, step = 0.00001),
        doc = """Threshold value for K channels.""")

    TCa = arrays.FloatArray(
        label = ":math:`T_{Ca}`",
        default = numpy.array([-0.01]),
        range = basic.Range(lo = -0.02, hi=-0.01, step = 0.0025),
        doc = "Threshold value for Ca channels.")

    TNa = arrays.FloatArray(
        label = ":math:`T_{Na}`",
        default = numpy.array([0.3]),
        range = basic.Range(lo = 0.25, hi= 0.3, step = 0.025),
        doc = "Threshold value for Na channels.")

    VCa = arrays.FloatArray(
        label = ":math:`V_{Ca}`",
        default = numpy.array([1.0]),
        range = basic.Range(lo = 0.9, hi = 1.1, step = 0.05),
        doc = """Ca Nernst potential.""")

    VK = arrays.FloatArray(
        label = ":math:`V_{K}`",
        default = numpy.array([-0.7]),
        range = basic.Range(lo = -0.8, hi = 1., step = 0.1),
        doc = """K Nernst potential.""")

    VL = arrays.FloatArray(
        label = ":math:`V_{L}`",
        default = numpy.array([-0.5]),
        range = basic.Range(lo = -0.7, hi = -0.4, step = 0.1),
        doc = """Nernst potential leak channels.""")

    VNa = arrays.FloatArray(
        label = ":math:`V_{Na}`",
        default = numpy.array([0.53]),
        range = basic.Range(lo = 0.51, hi = 0.55, step = 0.01),
        doc = """Na Nernst potential.""")

    d_K = arrays.FloatArray(
        label = r":math:`\delta_{K}`",
        default = numpy.array([0.3]),
        range = basic.Range(lo = 0.1, hi = 0.4, step = 0.1),
        doc = """Variance of K channel threshold.""")

    tau_K = arrays.FloatArray(
        label = r":math:`\tau_{K}`",
        default = numpy.array([1.0]),
        range = basic.Range(lo = 1.0, hi = 10.0, step = 1.0),
        doc = """Time constant for K relaxation time (ms)""")

    d_Na = arrays.FloatArray(
        label = r":math:`\delta_{Na}`",
        default = numpy.array([0.15]),
        range = basic.Range(lo = 0.1, hi = 0.2, step = 0.05),
        doc = "Variance of Na channel threshold.")

    d_Ca = arrays.FloatArray(
        label = r":math:`\delta_{Ca}`",
        default = numpy.array([0.15]),
        range = basic.Range(lo = 0.1, hi = 0.2, step = 0.05),
        doc = "Variance of Ca channel threshold.")

    aei = arrays.FloatArray(
        label = ":math:`a_{ei}`",
        default = numpy.array([2.0]),
        range = basic.Range(lo = 0.1, hi = 2.0, step = 0.1),
        doc = """Excitatory-to-inhibitory synaptic strength.""")

    aie = arrays.FloatArray(
        label = ":math:`a_{ie}`",
        default = numpy.array([2.0]),
        range = basic.Range(lo = 0.5, hi = 2.0, step = 0.1),
        doc = """Inhibitory-to-excitatory synaptic strength.""")

    b = arrays.FloatArray(
        label = ":math:`b`",
        default = numpy.array([0.1]),
        range = basic.Range(lo = 0.0001, hi = 1.0, step = 0.0001),
        doc = """Time constant scaling factor. The original value is 0.1""")

    C = arrays.FloatArray(
        label = ":math:`C`",
        default = numpy.array([0.1]),
        range = basic.Range(lo = 0.0, hi = 0.2, step = 0.01),
        doc = """Strength of excitatory coupling. Balance between internal and
        local (and global) coupling strength. C > 0 introduces interdependences between
        consecutive columns/nodes. C=1 corresponds to maximum coupling.
        This strenght should be set to sensible values when a whole network is connected. """)

    ane = arrays.FloatArray(
        label = ":math:`a_{ne}`",
        default = numpy.array([1.0]),
        range = basic.Range(lo = 0.4, hi = 1.0, step = 0.05),
        doc = """Non-specific-to-excitatory synaptic strength.""")

    ani = arrays.FloatArray(
        label = ":math:`a_{ni}`",
        default = numpy.array([0.4]),
        range = basic.Range(lo = 0.3, hi = 0.5, step = 0.05),
        doc = """Non-specific-to-inhibitory synaptic strength.""")

    aee = arrays.FloatArray(
        label = ":math:`a_{ee}`",
        default = numpy.array([0.4]),
        range = basic.Range(lo = 0.0, hi = 0.6, step = 0.05),
        doc = """Excitatory-to-excitatory synaptic strength.""")

    Iext = arrays.FloatArray(
       label = ":math:`I_{ext}`",
       default = numpy.array([0.3]),
       range = basic.Range(lo = 0.165, hi = 0.3, step = 0.005),
       doc = """Subcortical input strength. It represents a non-specific
       excitation or thalamic inputs.""")

    rNMDA = arrays.FloatArray(
        label = ":math:`r_{NMDA}`",
        default = numpy.array([0.25]),
        range = basic.Range(lo = 0.2, hi = 0.3, step = 0.05),
        doc = """Ratio of NMDA to AMPA receptors.""")

    VT = arrays.FloatArray(
        label = ":math:`V_{T}`",
        default = numpy.array([0.0]),
        range = basic.Range(lo = 0.0, hi = 0.7, step = 0.01),
        doc = """Threshold potential (mean) for excitatory neurons.
        In [Breaksetal_2003_b]_ this values is 0.""")

    d_V = arrays.FloatArray(
        label = r":math:`\delta_{V}`",
        default = numpy.array([0.65]),
        range = basic.Range(lo = 0.49, hi = 0.7, step = 0.01),
        doc = """Variance of the excitatory threshold. It is one of the main
        parameters explored in [Breaksetal_2003_b]_.""")

    ZT = arrays.FloatArray(
        label = ":math:`Z_{T}`",
        default = numpy.array([0.0]),
        range = basic.Range(lo = 0.0, hi = 0.1, step = 0.005),
        doc = """Threshold potential (mean) for inihibtory neurons.""")

    d_Z = arrays.FloatArray(
        label = r":math:`\delta_{Z}`",
        default = numpy.array([0.7]),
        range = basic.Range(lo = 0.001, hi = 0.75, step = 0.05),
        doc = """Variance of the inhibitory threshold.""")

    # NOTE: the values were not in the article.
    QV_max = arrays.FloatArray(
        label = ":math:`Q_{max}`",
        default = numpy.array([1.0]),
        range = basic.Range(lo = 0.1, hi = 1., step = 0.001),
        doc = """Maximal firing rate for excitatory populations (kHz)""")

    QZ_max = arrays.FloatArray(
        label = ":math:`Q_{max}`",
        default = numpy.array([1.0]),
        range = basic.Range(lo = 0.1, hi = 1., step = 0.001),
        doc = """Maximal firing rate for excitatory populations (kHz)""")


    variables_of_interest = basic.Enumerate(
        label="Variables watched by Monitors",
        options=["V", "W", "Z"],
        default=["V"],
        select_multiple=True,
        doc="""This represents the default state-variables of this Model to be
        monitored. It can be overridden for each Monitor if desired.""",
        order=10)

    #Informational attribute, used for phase-plane and initial()
    state_variable_range = basic.Dict(
        label = "State Variable ranges [lo, hi]",
        default = {"V": numpy.array([-1.5, 1.5]),
                   "W": numpy.array([-1.5, 1.5]),
                   "Z": numpy.array([-1.5, 1.5])},
        doc = """The values for each state-variable should be set to encompass
            the expected dynamic range of that state-variable for the current
            parameters, it is used as a mechanism for bounding random inital
            conditions when the simulation isn't started from an explicit
            history, it is also provides the default range of phase-plane plots.""")


    def __init__(self, **kwargs):
        """
        .. May need to put kwargs back if we can't get them from trait...

        """

        LOG.info('%s: initing...' % str(self))

        super(LarterBreakspear, self).__init__(**kwargs)

        self._state_variables = ["V", "W", "Z"]

        self._nvar = 3
        self.cvar = numpy.array([0], dtype=numpy.int32)

        LOG.debug('%s: inited.' % repr(self))


    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        Dynamic equations:

        .. math::
            \dot{V}_k & = - (g_{Ca} + (1 - C) \, r_{NMDA} \, a_{ee} \, Q_V + C \, r_{NMDA} \, a_{ee} \, \langle Q_V\rangle^{k}) \, m_{Ca} \, (V - VCa) \\
                           & \,\,- g_K \, W \, (V - VK) -  g_L \, (V - VL) \\
                           & \,\,- (g_{Na} \, m_{Na} + (1 - C) \, a_{ee} \, Q_V + C \, a_{ee} \, \langle Q_V\rangle^{k}) \,(V - VNa) \\
                           & \,\,- a_{ei} \, Z \, Q_Z + a_{ne} \, I, \\
                           & \\
            \dot{W}_k & = \phi \, \dfrac{m_K - W}{\tau_{K}},\\
                           & \nonumber\\
            \dot{Z}_k &= b (a_{ni}\, I + a_{ei}\,V\,Q_V),\\
            Q_{V}   &= Q_{V_{max}} \, (1 + \tanh\left(\dfrac{V_{k} - VT}{\delta_{V}}\right)),\\
            Q_{Z}   &= Q_{Z_{max}} \, (1 + \tanh\left(\dfrac{Z_{k} - ZT}{\delta_{Z}}\right)),

        """
        V = state_variables[0, :]
        W = state_variables[1, :]
        Z = state_variables[2, :]

        c_0   = coupling[0, :]


        # relationship between membrane voltage and channel conductance
        m_Ca = 0.5 * (1 + numpy.tanh((V - self.TCa) / self.d_Ca))
        m_Na = 0.5 * (1 + numpy.tanh((V - self.TNa) / self.d_Na))
        m_K  = 0.5 * (1 + numpy.tanh((V - self.TK )  / self.d_K))

        # voltage to firing rate
        QV    = 0.5 * self.QV_max * (1 + numpy.tanh((V - self.VT) / self.d_V))
        QZ    = 0.5 * self.QZ_max * (1 + numpy.tanh((Z - self.ZT) / self.d_Z))
        lc_0  = local_coupling * QV


        dV = (- (self.gCa + (1.0 - self.C) * (self.rNMDA * self.aee) * (QV + lc_0)+ self.C * self.rNMDA * self.aee * c_0) * m_Ca * (V - self.VCa) - self.gK * W * (V - self.VK) -  self.gL * (V - self.VL) - (self.gNa * m_Na + (1.0 - self.C) * self.aee * (QV  + lc_0) + self.C * self.aee * c_0) * (V - self.VNa) - self.aei * Z * QZ + self.ane * self.Iext)

        dW = (self.phi * (m_K - W) / self.tau_K)

        dZ = (self.b * (self.ani * self.Iext + self.aei * V * QV))

        derivative = numpy.array([dV, dW, dZ])

        return derivative



class ReducedWongWang(Model):
    r"""
    .. [WW_2006] Kong-Fatt Wong and Xiao-Jing Wang,  *A Recurrent Network
                Mechanism of Time Integration in Perceptual Decisions*.
                Journal of Neuroscience 26(4), 1314-1328, 2006.

    .. [DPA_2013] Deco Gustavo, Ponce Alvarez Adrian, Dante Mantini, Gian Luca
                  Romani, Patric Hagmann and Maurizio Corbetta. *Resting-State
                  Functional Connectivity Emerges from Structurally and
                  Dynamically Shaped Slow Linear Fluctuations*. The Journal of
                  Neuroscience 32(27), 11239-11252, 2013.



    .. automethod:: ReducedWongWang.__init__

    Equations taken from [DPA_2013]_ , page 11242

    .. math::
                 x_k       &=   w\,J_N \, S_k + I_o + J_N \mathbf\Gamma(S_k, S_j, u_{kj}),\\
                 H(x_k)    &=  \dfrac{ax_k - b}{1 - \exp(-d(ax_k -b))},\\
                 \dot{S}_k &= -\dfrac{S_k}{\tau_s} + (1 - S_k) \, H(x_k) \, \gamma

    """
    _ui_name = "Reduced Wong-Wang"

    #Define traited attributes for this model, these represent possible kwargs.
    a = arrays.FloatArray(
        label=":math:`a`",
        default=numpy.array([0.270, ]),
        range=basic.Range(lo=0.0, hi=0.270),
        doc=""" [nC]^{-1}. Parameter chosen to fit numerical solutions.""",
        order=1)

    b = arrays.FloatArray(
        label=":math:`b`",
        default=numpy.array([0.108, ]),
        range=basic.Range(lo=0.0, hi=1.0),
        doc="""[kHz]. Parameter chosen to fit numerical solutions.""",
        order=2)

    d = arrays.FloatArray(
        label=":math:`d`",
        default=numpy.array([154., ]),
        range=basic.Range(lo=0.0, hi=200.0),
        doc="""[ms]. Parameter chosen to fit numerical solutions.""",
        order=3)

    gamma = arrays.FloatArray(
        label=r":math:`\gamma`",
        default=numpy.array([0.641, ]),
        range=basic.Range(lo=0.0, hi=1.0),
        doc="""Kinetic parameter""",
        order=4)

    tau_s = arrays.FloatArray(
        label=r":math:`\tau_S`",
        default=numpy.array([100., ]),
        range=basic.Range(lo=50.0, hi=150.0),
        doc="""Kinetic parameter. NMDA decay time constant.""",
        order=5)

    w = arrays.FloatArray(
        label=r":math:`w`",
        default=numpy.array([0.6, ]),
        range=basic.Range(lo=0.0, hi=1.0, step=0.1),
        doc="""Excitatory recurrence""",
        order=6)

    J_N = arrays.FloatArray(
        label=r":math:`J_{N}`",
        default=numpy.array([0.2609, ]),
        range=basic.Range(lo=0.2609, hi=0.5, step=0.01),
        doc="""Excitatory recurrence""",
        order=7)

    I_o = arrays.FloatArray(
        label=":math:`I_{o}`",
        default=numpy.array([0.33, ]),
        range=basic.Range(lo=0.0, hi=1.0),
        doc="""[nA] Effective external input""",
        order=8)

    sigma_noise = arrays.FloatArray(
        label=r":math:`\sigma_{noise}`",
        default=numpy.array([0.000000001, ]),
        range=basic.Range(lo=0.0, hi=0.005),
        doc="""[nA] Noise amplitude. Take this value into account for stochatic
        integration schemes.""",
        order=-1)

    state_variable_range = basic.Dict(
        label="State variable ranges [lo, hi]",
        default={"S": numpy.array([0.0, 1.0])},
        doc="Population firing rate",
        order=9
    )

    variables_of_interest = basic.Enumerate(
        label="Variables watched by Monitors",
        options=["S"],
        default=["S"],
        select_multiple=True,
        doc="""default state variables to be monitored""",
        order=10)

    #    variables_of_interest = arrays.IntegerArray(
    #        label="Variables watched by Monitors",
    #        range=basic.Range(lo=0.0, hi=1.0, step=1.0),
    #        default=numpy.array([0], dtype=numpy.int32),
    #        doc="default state variables to be monitored",
    #        order=10)


    def __init__(self, **kwargs):
        """
        .. May need to put kwargs back if we can't get them from trait...

        """

        #LOG.info('%s: initing...' % str(self))

        super(ReducedWongWang, self).__init__(**kwargs)

        #self._state_variables = ["S1"]
        self._nvar = 1
        self.cvar = numpy.array([0], dtype=numpy.int32)

        LOG.debug('%s: inited.' % repr(self))

    def configure(self):
        """  """
        super(ReducedWongWang, self).configure()
        self.update_derived_parameters()


    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        Equations taken from [DPA_2013]_ , page 11242

        .. math::
                 x_k       &=   w\,J_N \, S_k + I_o + J_N \mathbf\Gamma(S_k, S_j, u_{kj}),\\
                 H(x_k)    &=  \dfrac{ax_k - b}{1 - \exp(-d(ax_k -b))},\\
                 \dot{S}_k &= -\dfrac{S_k}{\tau_s} + (1 - S_k) \, H(x_k) \, \gamma

        """

        S   = state_variables[0, :]
        S[S<0] = 0.
        S[S>1] = 1.
        c_0 = coupling[0, :]


        # if applicable
        lc_0 = local_coupling * S

        x  = self.w * self.J_N * S + self.I_o + self.J_N * c_0 + self.J_N * lc_0
        H = ((self.a * x ) - self.b) / (1 - numpy.exp(-self.d * ((self.a *x)- self.b)))
        dS = - (S / self.tau_s) + (1 - S) * H * self.gamma

        derivative = numpy.array([dS])
        return derivative



class Kuramoto(Model):
    r"""
    The Kuramoto model is a model of synchronization phenomena derived by
    Yoshiki Kuramoto in 1975 which has since been applied to diverse domains
    including the study of neuronal oscillations and synchronization.

    See:

        .. [YK_1975] Y. Kuramoto, in: H. Arakai (Ed.), International Symposium
            on Mathematical Problems in Theoretical Physics, *Lecture Notes in
            Physics*, page 420, vol. 39, 1975.

        .. [SS_2000] S. H. Strogatz. *From Kuramoto to Crawford: exploring the
            onset of synchronization in populations of coupled oscillators*.
            Physica D, 143, 2000.

        .. [JC_2011] J. Cabral, E. Hugues, O. Sporns, G. Deco. *Role of local
            network oscillations in resting-state functional connectivity*.
            NeuroImage, 57, 1, 2011.

    The :math:`\theta` variable is the phase angle of the oscillation.

    Dynamic equations:
        .. math::

                \dot{\theta}_{k} = \omega_{k} + \mathbf{\Gamma}(\theta_k, \theta_j, u_{kj}) + \sin(W_{\zeta}\theta)

    """

    _ui_name = "Kuramoto Oscillator"
    ui_configurable_parameters = ['omega']

    #Define traited attributes for this model, these represent possible kwargs.
    omega = arrays.FloatArray(
        label=r":math:`\omega`",
        default=numpy.array([1.0]),
        range=basic.Range(lo=0.01, hi=200.0, step=0.1),
        doc=""":math:`\omega` sets the base line frequency for the
            Kuramoto oscillator in [rad/ms]""",
        order=1)

    #Informational attribute, used for phase-plane and initial()
    state_variable_range = basic.Dict(
        label="State Variable ranges [lo, hi]",
        default={"theta": numpy.array([0.0, numpy.pi * 2.0]),
        },
        doc="""The values for each state-variable should be set to encompass
            the expected dynamic range of that state-variable for the current
            parameters, it is used as a mechanism for bounding random initial
            conditions when the simulation isn't started from an explicit
            history, it is also provides the default range of phase-plane plots.""",
        order=6)

    variables_of_interest = basic.Enumerate(
        label="Variables watched by Monitors",
        options=["theta"],
        default=["theta"],
        select_multiple=True,
        doc="""This represents the default state-variables of this Model to be
                            monitored. It can be overridden for each Monitor if desired. The Kuramoto
                            model, however, only has one state variable with and index of 0, so it
                            is not necessary to change the default here.""",
        order=7)



    def __init__(self, **kwargs):
        """
        May need to put kwargs back if we can't get them from trait...

        """

        LOG.info("%s: initing..." % str(self))

        super(Kuramoto, self).__init__(**kwargs)

        #self._state_variables = ["theta"]
        self._nvar = 1
        self.cvar = numpy.array([0], dtype=numpy.int32)

        LOG.debug("%s: inited." % repr(self))


    def dfun(self, state_variables, coupling, local_coupling=0.0,
             ev=numexpr.evaluate, sin=numpy.sin, pi2=numpy.pi * 2):
        r"""
        The :math:`\theta` variable is the phase angle of the oscillation.

        .. math::
            \dot{\theta}_{k} = \omega_{k} + \mathbf{\Gamma}(\theta_k, \theta_j, u_{kj}) + \sin(W_{\zeta}\theta)

        where :math:`I` is the input via local and long range connectivity,
        passing first through the Kuramoto coupling function,
        :py:class:tvb.simulator.coupling.Kuramoto.

        """

        theta = state_variables[0, :]
        #import pdb; pdb.set_trace()

        #A) Distribution of phases according to the local connectivity kernel
        local_range_coupling = numpy.sin(local_coupling * theta)

        # NOTE: To evaluate.
        #B) Strength of the interactions
        #local_range_coupling = local_coupling * numpy.sin(theta)

        I = coupling[0, :] + local_range_coupling

        if not hasattr(self, 'derivative'):
            self.derivative = numpy.empty((1,) + theta.shape)

        # phase update
        self.derivative[0] = self.omega + I

        # all this pi makeh me have great hungary, can has sum NaN?
        return self.derivative

    device_info = model_device_info(
        pars=['omega'],
        kernel="""
        float omega = P(0)
            , theta = X(0)
            , c_0 = I(0) ;

                    // update state array
        if (theta>(2*PI)) X(0)-= 2*PI;
        DX(0) = omega + c_0;

        """
    )


class Hopfield(Model):
    r"""

    The Hopfield neural network is a discrete time dynamical system composed
    of multiple binary nodes, with a connectivity matrix built from a
    predetermined set of patterns. The update, inspired from the spin-glass
    model (used to describe magnetic properties of dilute alloys), is based on
    a random scanning of every node. The existence of a fixed point dynamics
    is guaranteed by a Lyapunov function. The Hopfield network is expected to
    have those multiple patterns as attractors (multistable dynamical system).
    When the initial conditions are close to one of the 'learned' patterns,
    the dynamical system is expected to relax on the corresponding attractor.
    A possible output of the system is the final attractive state (interpreted
    as an associative memory).

    Various extensions of the initial model have been proposed, among which a
    noiseless and continuous version [Hopfield 1984] having a slightly
    different Lyapunov function, but essentially the same dynamical
    properties, with more straightforward physiological interpretation. A
    continuous Hopfield neural network (with a sigmoid transfer function) can
    indeed be interpreted as a network of neural masses with every node
    corresponding to the mean field activity of a local brain region, with
    many bridges with the Wilson Cowan model [WC_1972].

    **References**:

        .. [Hopfield1982] Hopfield, J. J., *Neural networks and physical systems with emergent collective
                        computational abilities*, Proc. Nat. Acad. Sci. (USA) 79, 2554-2558, 1982.

        .. [Hopfield1984] Hopfield, J. J., *Neurons with graded response have collective computational
                        properties like those of two-sate neurons*, Proc. Nat. Acad. Sci. (USA) 81, 3088-3092, 1984.

    See also, http://www.scholarpedia.org/article/Hopfield_network

    .. #This model can use a global threshold permitting multistable dynamic for
    .. #a positive structural connectivity matrix.

    .. automethod:: Hopfield.__init__
    .. automethod:: Hopfield.configure

    Dynamic equations:

    dfun equation
        .. math::
                \dot{x_{i}} &= 1 / \tau_{x} (-x_{i} + c_0)
    dfun dynamic equation
        .. math::
            \dot{x_{i}} &= 1 / \tau_{x} (-x_{i} + c_0(i)) \\
            \dot{\\theta_{i}} &= 1 / \tau_{\theta_{i}} (-\theta + c_1(i))

    """

    _ui_name = "Hopfield"
    ui_configurable_parameters = ['taux', 'tauT', 'dynamic']

    # Define traited attributes for this model, these represent possible kwargs.
    taux = arrays.FloatArray(
        label=":math:`\\tau_{x}`",
        default=numpy.array([1.]),
        range=basic.Range(lo=0.01, hi=100., step=0.01),
        doc="""The fast time-scale for potential calculus :math:`x`, state-variable of the model.""",
        order=1)

    tauT = arrays.FloatArray(
        label=":math:`\\tau_{\\theta}`",
        default=numpy.array([5.]),
        range=basic.Range(lo = 0.01, hi = 100., step = 0.01),
        doc="""The slow time-scale for threshold calculus :math:`\theta`, state-variable of the model.""",
        order=2)

    dynamic = arrays.IntegerArray(
        label="Dynamic",
        default=numpy.array([0, ]),
        range=basic.Range(lo=0, hi=1., step=1),
        doc="""Boolean value for static/dynamic threshold theta for (0/1).""",
        order=3)

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = basic.Dict(
        label="State Variable ranges [lo, hi]",
        default={"x": numpy.array([-1., 2.]),
                   "theta": numpy.array([0., 1.])},
        doc="""The values for each state-variable should be set to encompass
            the expected dynamic range of that state-variable for the current
            parameters, it is used as a mechanism for bounding random inital
            conditions when the simulation isn't started from an explicit
            history, it is also provides the default range of phase-plane plots.""",
        order = 4)

    variables_of_interest = basic.Enumerate(
        label="Variables watched by Monitors",
        options=["x", "theta"],
        default=["x"],
        select_multiple=True,
        doc="""The values for each state-variable should be set to encompass
            the expected dynamic range of that state-variable for the current
            parameters, it is used as a mechanism for bounding random initial
            conditions when the simulation isn't started from an explicit
            history, it is also provides the default range of phase-plane plots.""",
        order=5)

    def __init__(self, **kwargs):
        """Initialize the Hopfield model's traited attributes, any provided as
        keywords will overide their traited default.
        """

        LOG.info("%s: initing..." % str(self))
        super(Hopfield, self).__init__(**kwargs)

        self._nvar = 2
        self.cvar = numpy.array([0], dtype=numpy.int32)

        LOG.debug("%s: inited." % repr(self))

    def configure(self):
        """Set the threshold as a state variable for a dynamical threshold."""
        super(Hopfield, self).configure()

        if self.dynamic:
            self.dfun = self.dfunDyn
            self._nvar = 2
            self.cvar = numpy.array([0, 1], dtype=numpy.int32)
            # self.variables_of_interest = ["x", "theta"]

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        The fast, :math:`x`, and slow, :math:`\theta`, state variables are typically
        considered to represent a membrane potentials of nodes and the global inhibition term,
        respectively:

            .. math::
                \dot{x_{i}} &= 1 / \tau_{x} (-x_{i} + c_0)

        """

        x = state_variables[0, :]
        dx = (- x + coupling[0]) / self.taux

        # We return 2 arrays here, because we have 2 possible state Variable, even if not dynamic
        # Otherwise the phase-plane display will fail.
        derivative = numpy.array([dx, dx])
        return derivative

    def dfunDyn(self, state_variables, coupling, local_coupling=0.0):
        r"""
        The fast, :math:`x`, and slow, :math:`\theta`, state variables are typically
        considered to represent a membrane potentials of nodes and the inhibition term(s),
        respectively:

            .. math::
                \dot{x_{i}} &= 1 / \tau_{x} (-x_{i} + c_0(i)) \\
                \dot{\theta_{i}} &= 1 / \tau_{\theta_{i}} (-\theta + c_1(i))

        where c_0 is the coupling term and c_1 should be the direct output.

        """

        x = state_variables[0, :]
        theta = state_variables[1, :]
        dx = (- x + coupling[0]) / self.taux
        dtheta = (- theta + coupling[1]) / self.tauT

        derivative = numpy.array([dx, dtheta])
        return derivative


class Epileptor(Model):
    r"""
    The Epileptor is a composite neural mass model of six dimensions which
    has been crafted to model the phenomenology of epileptic seizures.
    (see [Jirsaetal_2014]_)

    Equations and default parameters are taken from [Jirsaetal_2014]_.

          +------------------------------------------------------+
          |                         Table 1                      |
          +----------------------+-------------------------------+
          |        Parameter     |           Value               |
          +======================+===============================+
          |         I_rest1      |              3.1              |
          +----------------------+-------------------------------+
          |         I_rest2      |              0.45             |
          +----------------------+-------------------------------+
          |         r            |            0.00035            |
          +----------------------+-------------------------------+
          |         x_0          |             -1.6              |
          +----------------------+-------------------------------+
          |         slope        |              0.0              |
          +----------------------+-------------------------------+
          |             Integration parameter                    |
          +----------------------+-------------------------------+
          |           dt         |              0.1              |
          +----------------------+-------------------------------+
          |  simulation_length   |              4000             |
          +----------------------+-------------------------------+
          |                    Noise                             |
          +----------------------+-------------------------------+
          |         nsig         | [0., 0., 0., 1e-3, 1e-3, 0.]  |
          +----------------------+-------------------------------+
          |              Jirsa et al. 2014                       |
          +------------------------------------------------------+


    .. figure :: img/Epileptor_01_mode_0_pplane.svg
        :alt: Epileptor phase plane

    .. [Jirsaetal_2014] Jirsa, V. K.; Stacey, W. C.; Quilichini, P. P.;
        Ivanov, A. I.; Bernard, C. *On the nature of seizure dynamics.* Brain,
        2014.

    .. automethod:: Epileptor.__init__

    Variables of interest to be used by monitors: -y[0] + y[3]

        .. math::
            \dot{x_{1}} &=& y_{1} - f_{1}(x_{1}, x_{2}) - z + I_{ext1} \\
            \dot{y_{1}} &=& c - d x_{1}^{2} - y{1} \\
            \dot{z} &=&
            \begin{cases}
            r(4 (x_{1} - x_{0}) - z-0.1 z^{7}) & \text{if } x<0 \\
            r(4 (x_{1} - x_{0}) - z) & \text{if } x \geq 0
            \end{cases} \\
            \dot{x_{2}} &=& -y_{2} + x_{2} - x_{2}^{3} + I_{ext2} + 0.002 g - 0.3 (z-3.5) \\
            \dot{y_{2}} &=& 1 / \tau (-y_{2} + f_{2}(x_{2}))\\
            \dot{g} &=& -0.01 (g - 0.1 x_{1})

    where:
        .. math::
            f_{1}(x_{1}, x_{2}) =
            \begin{cases}
            a x_{1}^{3} - b x_{1}^2 & \text{if } x_{1} <0\\
            -(slope - x_{2} + 0.6(z-4)^2) x_{1} &\text{if }x_{1} \geq 0
            \end{cases}

    and:

        .. math::
            f_{2}(x_{2}) =
            \begin{cases}
            0 & \text{if } x_{2} <-0.25\\
            a_{2}(x_{2} + 0.25) & \text{if } x_{2} \geq -0.25
            \end{cases}
    """

    _ui_name = "Epileptor"
    ui_configurable_parameters = ["Iext", "Iext2", "r", "x0", "slope"]

    a = arrays.FloatArray(
        label="a",
        default=numpy.array([1]),
        doc="Coefficient of the cubic term in the first state variable",
        order=-1)

    b = arrays.FloatArray(
        label="b",
        default=numpy.array([3]),
        doc="Coefficient of the squared term in the first state variabel",
        order=-1)

    c = arrays.FloatArray(
        label="c",
        default=numpy.array([1]),
        doc="Additive coefficient for the second state variable, \
        called :math:`y_{0}` in Jirsa paper",
        order=-1)

    d = arrays.FloatArray(
        label="d",
        default=numpy.array([5]),
        doc="Coefficient of the squared term in the second state variable",
        order=-1)

    r = arrays.FloatArray(
        label="r",
        range=basic.Range(lo=0.0, hi=0.001, step=0.00005),
        default=numpy.array([0.00035]),
        doc="Temporal scaling in the third state variable, \
        called :math:`1/\\tau_{0}` in Jirsa paper",
        order=4)

    s = arrays.FloatArray(
        label="s",
        default=numpy.array([4]),
        doc="Linear coefficient in the third state variable",
        order=-1)

    x0 = arrays.FloatArray(
        label="x0",
        range=basic.Range(lo=-3.0, hi=-1.0, step=0.1),
        default=numpy.array([-1.6]),
        doc="Epileptogenicity parameter",
        order=3)

    Iext = arrays.FloatArray(
        label="Iext",
        range=basic.Range(lo=1.5, hi=5.0, step=0.1),
        default=numpy.array([3.1]),
        doc="External input current to the first population",
        order=1)

    slope = arrays.FloatArray(
        label="slope",
        range=basic.Range(lo=-16.0, hi=6.0, step=0.1),
        default=numpy.array([0.]),
        doc="Linear coefficient in the first state variable",
        order=5)

    Iext2 = arrays.FloatArray(
        label="Iext2",
        range=basic.Range(lo=0.0, hi=1.0, step=0.05),
        default=numpy.array([0.45]),
        doc="External input current to the second population",
        order=2)

    tau = arrays.FloatArray(
        label="tau",
        default=numpy.array([10]),
        doc="Temporal scaling coefficient in fifth state variable",
        order=-1)

    aa = arrays.FloatArray(
        label="aa",
        default=numpy.array([6]),
        doc="Linear coefficient in fifth state variable",
        order=-1)

    Kvf = arrays.FloatArray(
        label="K_vf",
        default=numpy.array([0.0]),
        range=basic.Range(lo=0.0, hi=4.0, step=0.5),
        doc="Coupling scaling on a very fast time scale.",
        order=-1)

    Kf = arrays.FloatArray(
        label="K_f",
        default=numpy.array([0.0]),
        range=basic.Range(lo=0.0, hi=4.0, step=0.5),
        doc="Correspond to the coupling scaling on a fast time scale.",
        order=-1)

    state_variable_range = basic.Dict(
        label="State variable ranges [lo, hi]",
        default={"y0": numpy.array([-2., 1.]),
                 "y1": numpy.array([-20., 2.]),
                 "y2": numpy.array([2.0, 5.0]),
                 "y3": numpy.array([-2., 0.]),
                 "y4": numpy.array([0., 2.]),
                 "y5": numpy.array([-1., 1.])},
        doc="n/a",
        order=-1
        )

    variables_of_interest = basic.Enumerate(
        label="Variables watched by Monitors",
        options=["y0", "y1", "y2", "y3", "y4", "y5"],
        default=["y0", "y3"],
        select_multiple=True,
        doc="""default state variables to be monitored""",
        order=-1)

    def __init__(self, **kwargs):
        """
        """

        LOG.info("%s: init'ing..." % (str(self),))

        super(Epileptor, self).__init__(**kwargs)

        self._nvar = 6
        self.cvar = numpy.array([0, 3], dtype=numpy.int32)

        LOG.info("%s: init'ed." % (repr(self),))

    def dfun(self, state_variables, coupling, local_coupling=0.0,
             array=numpy.array, where=numpy.where, concat=numpy.concatenate):
        r"""
        Computes the derivatives of the state variables of the Epileptor
        with respect to time.

        Implementation note: we expect this version of the Epileptor to be used
        in a vectorized manner. Concretely, y has a shape of (6, n) where n is
        the number of nodes in the network. An consequence is that
        the original use of if/else is translated by calculated both the true
        and false forms and mixing them using a boolean mask.

        Variables of interest to be used by monitors: -y[0] + y[3]

            .. math::
                \dot{x_{1}} &=& y_{1} - f_{1}(x_{1}, x_{2}) - z + I_{ext1} \\
                \dot{y_{1}} &=& c - d x_{1}^{2} - y{1} \\
                \dot{z} &=&
                \begin{cases}
                r(4 (x_{1} - x_{0}) - z-0.1 z^{7}) & \text{if } x<0 \\
                r(4 (x_{1} - x_{0}) - z) & \text{if } x \geq 0
                \end{cases} \\
                \dot{x_{2}} &=& -y_{2} + x_{2} - x_{2}^{3} + I_{ext2} + 0.002 g - 0.3 (z-3.5) \\
                \dot{y_{2}} &=& 1 / \tau (-y_{2} + f_{2}(x_{2}))\\
                \dot{g} &=& -0.01 (g - 0.1 x_{1})

        where:
            .. math::
                f_{1}(x_{1}, x_{2}) =
                \begin{cases}
                a x_{1}^{3} - b x_{1}^2 & \text{if } x_{1} <0\\
                -(slope - x_{2} + 0.6(z-4)^2) x_{1} &\text{if }x_{1} \geq 0
                \end{cases}

            .. math::
                f_{2}(x_{2}) =
                \begin{cases}
                0 & \text{if } x_{2} <-0.25\\
                a_{2}(x_{2} + 0.25) & \text{if } x_{2} \geq -0.25
                \end{cases}

        """

        y = state_variables

        Iext = self.Iext + local_coupling * y[0]
        c_pop1 = coupling[0, :]
        c_pop2 = coupling[1, :]

        # population 1
        if_ydot0 = y[1] - self.a*y[0]**3 + self.b*y[0]**2 - y[2] + Iext + self.Kvf*c_pop1
        else_ydot0 = y[1] + (self.slope - y[3] + 0.6*(y[2]-4.0)**2)*y[0] - y[2] + Iext + self.Kvf*c_pop1
        ydot0 = where(y[0] < 0., if_ydot0, else_ydot0)
        ydot1 = self.c - self.d*y[0]**2 - y[1]

        # energy
        if_ydot2 = self.r*(4*(y[0] - self.x0) - y[2] - 0.1*y[2]**7)
        else_ydot2 = self.r*(4*(y[0] - self.x0) - y[2])
        ydot2 = where(y[2] < 0., if_ydot2, else_ydot2)

        # population 2
        ydot3 = -y[4] + y[3] - y[3]**3 + self.Iext2 + 2*y[5] - 0.3*(y[2] - 3.5) + self.Kf*c_pop2
        if_ydot4 = -y[4]/self.tau
        else_ydot4 = (-y[4] + self.aa*(y[3] + 0.25))/self.tau
        ydot4 = where(y[3] < -0.25, if_ydot4, else_ydot4)

        # filter
        ydot5 = -0.01*(y[5] - 0.1*y[0])

        #
        ydot = numpy.array([ydot0, ydot1, ydot2, ydot3, ydot4, ydot5])

        return ydot


class EpileptorPermittivityCoupling(Model):
    r"""
    Modified version of the Epileptor model:

    - The third state variable equation is modified to account for the time
      difference between interictal and ictal states
    - The equations are scales in time to have realist time lengths
    - A seventh state variable is added to directly calculate the correct
      output of the model for the monitors (not strictly correct
      mathematically except for Euler integration method)
    - There is a possible coupling between fast and slow time scales, call the
      permittivity coupling.

    .. figure :: img/EpileptorPermittivityCoupling_01_mode_0_pplane.svg
        :alt: Epileptor phase plane

    .. automethod:: EpileptorPermittivityCoupling.__init__

    Variables of interest to be used by monitors: y[6] = -y[0] + y[3]

        .. math::
            \dot{x_{1}} &=& y_{1} - f_{1}(x_{1}, x_{2}) - z + I_{ext1} \\
            \dot{y_{1}} &=& c - d x_{1}^{2} - y{1} \\
            \dot{z} &=& r(x_{0} + 3/(1 + \exp((-x_{1}-0.5)/0.1)) - z)\\
            \dot{x_{2}} &=& -y_{2} + x_{2} - x_{2}^{3} + I_{ext2} + 0.002 g(x_{1}) - 0.3 (z-3.5) \\
            \dot{y_{2}} &=& 1 / \tau (-y_{2} + f_{2}(x_{1}, x_{2}))\\
            \dot{g} &=& -0.01 (g - 0.1 x_{1})

    where:
        .. math::
            f_{1}(x_{1}, x_{2}) =
            \begin{cases}
            a x_{1}^{3} - b x_{1}^2 & \text{if } x_{1} <0\\
            -(slope - x_{2} + 0.6(z-4)^2) x_{1} &\text{if }x_{1} \geq 0
            \end{cases}

    and:
        .. math::
            f_{2}(x_{2}) =
            \begin{cases}
            0 & \text{if } x_{2} <-0.25\\
            a_{2}(x_{2} + 0.25) & \text{if } x_{2} \geq -0.25
            \end{cases}
    """

    _ui_name = "Epileptor with Permittivity Coupling"
    ui_configurable_parameters = ["Iext", "Iext2", "r", "x0", "slope"]

    a = arrays.FloatArray(
        label="a",
        default=numpy.array([1]),
        doc="Coefficient of the cubic term in the first state variable",
        order=-1)

    b = arrays.FloatArray(
        label="b",
        default=numpy.array([3]),
        doc="Coefficient of the squared term in the first state variabel",
        order=-1)

    c = arrays.FloatArray(
        label="c",
        default=numpy.array([1]),
        doc="Additive coefficient for the second state variable, \
        called :math:`y_{0}` in Jirsa paper",
        order=-1)

    d = arrays.FloatArray(
        label="d",
        default=numpy.array([5]),
        doc="Coefficient of the squared term in the second state variable",
        order=-1)

    r = arrays.FloatArray(
        label="r",
        range=basic.Range(lo=0.0, hi=0.001, step=0.00005),
        default=numpy.array([0.00035]),
        doc="Temporal scaling in the third state variable, \
        called :math:`1/\\tau_{0}` in Jirsa paper",
        order=4)

    s = arrays.FloatArray(
        label="s",
        default=numpy.array([4]),
        doc="Linear coefficient in the third state variable",
        order=-1)

    x0 = arrays.FloatArray(
        label="x0",
        range=basic.Range(lo=2.0, hi=6.0, step=0.1),
        default=numpy.array([2.5]),
        doc="Epileptogenicity parameter",
        order=3)

    Iext = arrays.FloatArray(
        label="Iext",
        range=basic.Range(lo=1.5, hi=5.0, step=0.1),
        default=numpy.array([3.1]),
        doc="External input current to the first population",
        order=1)

    slope = arrays.FloatArray(
        label="slope",
        range=basic.Range(lo=-16.0, hi=6.0, step=0.1),
        default=numpy.array([0.]),
        doc="Linear coefficient in the first state variable",
        order=5)

    Iext2 = arrays.FloatArray(
        label="Iext2",
        range=basic.Range(lo=0.0, hi=1.0, step=0.05),
        default=numpy.array([0.45]),
        doc="External input current to the second population",
        order=2)

    tau = arrays.FloatArray(
        label="tau",
        default=numpy.array([10]),
        doc="Temporal scaling coefficient in fifth state variable",
        order=-1)

    aa = arrays.FloatArray(
        label="aa",
        default=numpy.array([6]),
        doc="Linear coefficient in fifth state variable",
        order=-1)

    Kvf = arrays.FloatArray(
        label="K_vf",
        default=numpy.array([0.0]),
        range=basic.Range(lo=0.0, hi=4.0, step=0.5),
        doc="Coupling scaling on a very fast time scale.",
        order=-1)

    Kf = arrays.FloatArray(
        label="K_f",
        default=numpy.array([0.0]),
        range=basic.Range(lo=0.0, hi=4.0, step=0.5),
        doc="Correspond to the coupling scaling on a fast time scale.",
        order=-1)

    Ks = arrays.FloatArray(
        label="K_s",
        default=numpy.array([0.0]),
        range=basic.Range(lo=0.0, hi=4.0, step=0.5),
        doc="Coupling scaling on a slow time scale.",
        order=-1)

    tt = arrays.FloatArray(
        label="tt",
        default=numpy.array([1.0/(2**4)]),
        range=basic.Range(lo=0.001, hi=1.0, step=0.001),
        doc="Time scaling of the whole system to the system in real time",
        order=6)

    state_variable_range = basic.Dict(
        label="State variable ranges [lo, hi]",
        default={"y0": numpy.array([-2., 1.]),
                 "y1": numpy.array([-20, 2.]),
                 "y2": numpy.array([2.0, 5.0]),
                 "y3": numpy.array([-2., -1.]),
                 "y4": numpy.array([0., 2.]),
                 "y5": numpy.array([-1., 1.]),
                 "y6": numpy.array([-5., 5.])},
        doc="n/a",
        order=-1
        )

    variables_of_interest = basic.Enumerate(
        label="Variables watched by Monitors",
        options=["y0", "y1", "y2", "y3", "y4", "y5", "y6"],
        default=["y0", "y3", "y6"],
        select_multiple=True,
        doc="""default state variables to be monitored""",
        order=-1)

    def __init__(self, **kwargs):
        """
        """

        LOG.info("%s: init'ing..." % (str(self),))

        super(EpileptorPermittivityCoupling, self).__init__(**kwargs)

        self._nvar = 7
        self.cvar = numpy.array([0, 3], dtype=numpy.int32)

        LOG.info("%s: init'ed." % (repr(self),))

    def dfun(self, state_variables, coupling, local_coupling=0.0,
             array=numpy.array, where=numpy.where, concat=numpy.concatenate):
        r"""
        Computes the derivatives of the state variables of the Epileptor
        with respect to time.

        Implementation note: we expect this version of the Epileptor to be used
        in a vectorized manner. Concretely, y has a shape of (6, n) where n is
        the number of nodes in the network. An consequence is that
        the original use of if/else is translated by calculated both the true
        and false forms and mixing them using a boolean mask.

        Variables of interest to be used by monitors: y[6] = -y[0] + y[3]

            .. math::
                \dot{x_{1}} &=& y_{1} - f_{1}(x_{1}, x_{2}) - z + I_{ext1} \\
                \dot{y_{1}} &=& c - d x_{1}^{2} - y{1} \\
                \dot{z} &=& r(x_{0} + 3/(1 + \exp((-x_{1}-0.5)/0.1)) - z)\\
                \dot{x_{2}} &=& -y_{2} + x_{2} - x_{2}^{3} + I_{ext2} + 0.002 g(x_{1}) - 0.3 (z-3.5) \\
                \dot{y_{2}} &=& 1 / \tau (-y_{2} + f_{2}(x_{1}, x_{2}))\\
                \dot{g} &=& -0.01 (g - 0.1 x_{1})

        where:
            .. math::
                f_{1}(x_{1}, x_{2}) =
                \begin{cases}
                a x_{1}^{3} - b x_{1}^2 & \text{if } x_{1} <0\\
                -(slope - x_{2} + 0.6(z-4)^2) x_{1} &\text{if }x_{1} \geq 0
                \end{cases}

        and:

            .. math::
                f_{2}(x_{2}) =
                \begin{cases}
                0 & \text{if } x_{2} <-0.25\\
                a_{2}(x_{2} + 0.25) & \text{if } x_{2} \geq -0.25
                \end{cases}

"""

        y = state_variables

        Iext = self.Iext + local_coupling * y[0]
        c_pop1 = coupling[0, :]
        c_pop2 = coupling[1, :]

        # population 1
        if_ydot0 = self.tt*(y[1] - self.a*y[0]**3 + self.b*y[0]**2 - y[2] + Iext + self.Kvf*c_pop1)
        else_ydot0 = self.tt*(y[1] + (self.slope - y[3] + 0.6*(y[2]-4.0)**2)*y[0] - y[2] + Iext + self.Kvf*c_pop1)
        ydot0 = where(y[0] < 0., if_ydot0, else_ydot0)
        ydot1 = self.tt*(self.c - self.d*y[0]**2 - y[1])

        # energy
        ydot2 = self.tt*(self.r*(3./(1.+numpy.exp(-(y[0]+0.5)/0.2)) + self.x0 - y[2] - self.Ks*c_pop1))

        # population 2
        ydot3 = self.tt*(-y[4] + y[3] - y[3]**3 + self.Iext2 + 2*y[5] - 0.3*(y[2] - 3.5) + self.Kf*c_pop2)
        if_ydot4 = self.tt*(-y[4]/self.tau)
        else_ydot4 = self.tt*((-y[4] + self.aa*(y[3] + 0.25))/self.tau)
        ydot4 = where(y[3] < -0.25, if_ydot4, else_ydot4)

        # filter
        ydot5 = self.tt*(-0.01*(y[5] - 0.1*y[0]))

        # output time series
        ydot6 = -ydot0 + ydot3

        ydot = numpy.array([ydot0, ydot1, ydot2, ydot3, ydot4, ydot5, ydot6])

        return ydot
