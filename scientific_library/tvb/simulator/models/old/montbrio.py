from tvb.simulator.models.base import Model
from tvb.basic.neotraits.api import NArray, List, Range, Final

import numpy as np


class Theta2D(Model):
	r"""
	2D model describing the Ott-Antonsen reduction of infinitely all-to-all coupled QIF neurons (Theta-neurons).
	Depending on the parameter choice one finds the system equations as in
	i) Montbrio, Pazo, Roxin, 2015.
	ii) Coombes, Byrne, 2016.

	The two state variables :math:`r` and :math:`V` represent the average firing rate and
	the average membrane potential of our QIF neurons.

	The equations of the infinite QIF 2D population model read
	.. math::
			\dot{r} &= Delta/pi + 2 V r - k r^2,
			\dot{V} &= (V^2 - pi^2 r^2 + eta + (k s + J) r - k V r + gamma I ), \\

	"""

	# Define traited attributes for this model, these represent possible kwargs.
	I = NArray(
		label=":math:`I_{ext}`",
		default=np.array([0.0]),
		domain=Range(lo=-10.0, hi=10.0, step=0.01),
		doc="""???""",
	)

	Delta = NArray(
		label=r":math:`\Delta`",
		default=np.array([1.0]),
		domain=Range(lo=0.0, hi=10.0, step=0.01),
		doc="""Vertical shift of the configurable nullcline""",
	)

	alpha = NArray(
		label=r":math:`\alpha`",
		default=np.array([1.0]),
		domain=Range(lo=0.0, hi=1.0, step=0.1),
		doc=""":math:`\alpha` ratio of effect between long-range and local connectivity""",
	)

	s = NArray(
		label=":math:`s`",
		default=np.array([0.0]),
		domain=Range(lo=-15.0, hi=15.0, step=0.01),
		doc="""QIF membrane reversal potential""",
	)

	k = NArray(
		label=":math:`k`",
		default=np.array([0.0]),
		domain=Range(lo=-15.0, hi=15.0, step=0.01),
		doc="""Switch for the terms specific to Coombes model""",
	)

	J = NArray(
		label=":math:`J`",
		default=np.array([15.0]),
		domain=Range(lo=-25.0, hi=25.0, step=0.0001),
		doc="""Constant parameter to scale the rate of feedback from the
            slow variable to the firing rate variable.""",
	)

	eta = NArray(
		label=r":math:`\eta`",
		default=np.array([-5.0]),
		domain=Range(lo=-10.0, hi=10.0, step=0.0001),
		doc="""Constant parameter to scale the rate of feedback from the
            firing rate variable to itself""",
	)

	# Informational attribute, used for phase-plane and initial()
	state_variable_range = Final(
		label="State Variable ranges [lo, hi]",
		default={"r": np.array([0., 2.0]),
				 "V": np.array([-2.0, 1.5])},
		doc="""Expected ranges of the state variables for initial condition generation and phase plane setup.""",
	)

	state_variable_boundaries = Final(
		label="State Variable boundaries [lo, hi]",
		default={
			"r": np.array([0.0, np.inf])
		},
	)

	Gamma = NArray(
		label=r":math:`\Gamma`",
		default=np.array([0.0]),
		domain=Range(lo=0., hi=10.0, step=0.1),
		doc="""Derived from eterogeneous currents and synaptic weights (see Montbrio p.12)""",
	)

	# This parameter is basically a hack to avoid having a negative lower boundary in the global coupling strength.
	gamma = NArray(
		label=r":math:`\gamma`",
		default=np.array([1.0]),
		domain=Range(lo=-2.0, hi=2.0, step=0.1),
		doc="""Constant parameter to reproduce FHN dynamics where
               excitatory input currents are negative.
               It scales both I and the long range coupling term.""",
	)

	variables_of_interest = List(
		of=str,
		label="Variables or quantities available to Monitors",
		choices=("r", "V"),
		default=("r", "V"),
		doc="The quantities of interest for monitoring for the Infinite QIF 2D oscillator.",
	)

	state_variables = ('r', 'V')
	_nvar = 2
	# Cvar is the coupling variable.
	cvar = np.array([0, 1], dtype=np.int32)

	def dfun(self, state_variables, coupling, local_coupling=0.0):
		r = state_variables[0, :]
		V = state_variables[1, :]

		# [State_variables, nodes]
		I = self.I
		Delta = self.Delta
		s = self.s
		k = self.k
		gamma = self.gamma
		Gamma = self.Gamma
		eta = self.eta
		J = self.J
		alpha = self.alpha

		Coupling_global = alpha * coupling[0, :]  # This zero refers to the first element of cvar (trivial in this case)
		Coupling_local = (1 - alpha) * local_coupling * r
		Coupling_Term = Coupling_global + Coupling_local

		derivative = np.empty_like(state_variables)

		derivative[0] = Delta / np.pi + 2 * V * r - k * r ** 2 + Gamma * r / np.pi
		derivative[1] = V ** 2 - np.pi ** 2 * r ** 2 + eta + (k * s + J) * r - k * V * r + gamma * I + Coupling_Term

		return derivative


class Montbrio(Theta2D):
	r"""
	Parametrization of the Theta 2D model corresponding to Montbrio, Pazo, Roxin, 2015.
	"""
	k = Final(
		label=":math:`k`",
		default=np.array([0.0]),
		doc="""Switch for the terms specific to Coombes model.  Equals 0 in the Montbrio et al., 2015.""",
	)

	s = Final(
		label=":math:`s`",
		default=np.array([0.0]),
		doc="""QIF membrane reversal potential. Equals 0 in the Montbrio et al., 2015.""",
	)


class Coombes(Theta2D):
	r"""
	Parametrization of the Theta 2D model corresponding to Coombes, Byrne, 2016.
	"""

	J = Final(
		label=":math:`J`",
		default=np.array([0.0]),
		doc="""Constant parameter to scale the rate of feedback from the
            slow variable to the firing rate variable. Equals 0 in the Coombes, Byrne, 2016.""",
	)

	Gamma = Final(
		label=":math:`\Gamma`",
		default=np.array([0.0]),
		doc="""Derived from eterogeneous currents and synaptic weights (see Montbrio p.12). Equals 0 in the Coombes, Byrne, 2016.""",
	)
