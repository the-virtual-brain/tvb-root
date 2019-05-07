# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
#
#

"""
The original Wong and Wang model

.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>
.. moduleauthor:: Marmaduke Woodman <Marmaduke@tvb.invalid>
.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""


# Third party python libraries
import numpy

#The Virtual Brain
from tvb.simulator.common import get_logger
LOG = get_logger(__name__)

import tvb.datatypes.arrays as arrays
import tvb.basic.traits.types_basic as basic 
import tvb.simulator.models as models

class WongWang(models.Model):
    """
    .. [WW_2006] Kong-Fatt Wong and Xiao-Jing Wang,  *A Recurrent Network 
                Mechanism of Time Integration in Perceptual Decisions*. 
                Journal of Neuroscience 26(4), 1314-1328, 2006.

    .. [WW_2006_SI] Supplementary Information
                
    .. [WW_2007] Kong-Fatt Wong, Alexander C. Huk2, Michael N. Shadlen,
                Xiao-Jing Wang, *Neural circuit dynamics underlying accumulation
                of time-varying evidence during perceptual decision making*.
                Front. Comput. Neurosci., 2007.

    A reduced model by Wong and Wang: A reduced two-variable neural model 
    that offers a simple yet biophysically plausible framework for studying 
    perceptual decision making in general.

    S is the NMDA gating variable. Since its decay time is much longer that those
    corresponding to AMPAand GABA gating variables, it is assumed that is 
    :math:`S_{NMDA}` that dominates the time evolution of the system.

    The model (:math:`Sl`, :math:`Sr`) phase-plane, including a representation 
    of the vector field as well as its nullclines, using default parameters, 
    can be seen below:

    Notation and parameter selection follows _Materials and methods_ from [WW_2007].

    To reproduce the phase plane in Figure 5B, page 1320:
        Jll = Jrr = 0.3725
        Jlr = Jrl = 0.1137
        J_ext = 1.1e-3
        I_o = 0.3297
        mu_o = 30
        c = 6.4 

    To reproduce C & D vary c parameter respectively.

    .. automethod:: __init__

    """
    _ui_name = "Wong-Wang (2D)"

    #Define traited attributes for this model, these represent possible kwargs.
    a = arrays.FloatArray(
        label=":math:`a`",
        default=numpy.array([270., ]),
        range=basic.Range(lo=0.0, hi=1000.0),
        doc="""[Hz/nA] Parameter chosen to ﬁt numerical solutions.""")

    b = arrays.FloatArray(
        label=":math:`b`",
        default=numpy.array([108, ]),
        range=basic.Range(lo=0.0, hi=1.0),
        doc="""[Hz]. Parameter chosen to ﬁt numerical solutions.""")

    d = arrays.FloatArray(
        label=":math:`d`",
        default=numpy.array([0.154, ]),
        range=basic.Range(lo=0.0, hi=200.0),
        doc="""[s]. Parameter chosen to ﬁt numerical solutions.""")

    gamma = arrays.FloatArray(
        label=r":math:`\gamma`",
        default=numpy.array([0.641, ]),
        range=basic.Range(lo=0.0, hi=1.0),
        doc="""Kinetic parameter""")

    tau_s = arrays.FloatArray(
        label=r":math:`\tau_S`",
        default=numpy.array([60., ]),
        range=basic.Range(lo=50.0, hi=150.0),
        doc="""[ms] Kinetic parameter. NMDA decay time constant.""")

    tau_noise = arrays.FloatArray(
        label=r":math:`\tau_{noise}`",
        default=numpy.array([2., ]),
        range=basic.Range(lo=1.0, hi=10.0),
        doc="""[ms] Noise decay time constant.""",
        order=-1)

    Jll = arrays.FloatArray(
        label=":math:`J_{ll}`",
        default=numpy.array([0.3725, ]),
        range=basic.Range(lo=0.0, hi=1.0),
        doc="""Synaptic coupling""")

    Jrr = arrays.FloatArray(
        label=":math:`J_{rr}`",
        default=numpy.array([0.3725, ]),
        range=basic.Range(lo=0.0, hi=1.0),
        doc="""Synaptic coupling""")

    Jlr = arrays.FloatArray(
        label=":math:`J_{lr}`",
        default=numpy.array([0.1137, ]),
        range=basic.Range(lo=0.0, hi=1.0),
        doc="""Synaptic coupling""")

    Jrl = arrays.FloatArray(
        label=":math:`J_{rl}`",
        default=numpy.array([0.1137, ]),
        range=basic.Range(lo=0.0, hi=1.0),
        doc="""Synaptic coupling""")

    J_N = arrays.FloatArray(
        label=":math:`J_{N}`",
        default=numpy.array([0.1137, ]),
        range=basic.Range(lo=0.0, hi=1.0),
        doc="""External synaptic coupling""")

    J_ext = arrays.FloatArray(
        label=":math:`J_{ext}`",
        default=numpy.array([1.1e-3, ]),
        range=basic.Range(lo=0.0, hi=0.01),
        doc="""[nA/Hz] Synaptic coupling""")

    I_o = arrays.FloatArray(
        label=":math:`I_{o}`",
        default=numpy.array([0.3297, ]),
        range=basic.Range(lo=0.0, hi=1.0),
        doc="""Effective external input""")

    sigma_noise = arrays.FloatArray(
        label=r":math:`\sigma_{noise}`",
        default=numpy.array([0.009, ]),
        range=basic.Range(lo=0.0, hi=1.0),
        doc="""Noise amplitude. Take this value into account for stochatic
        integration schemes.""")

    mu_o = arrays.FloatArray(
        label=r":math:`\mu_{0}`",
        default=numpy.array([30, ]),
        range=basic.Range(lo=0.0, hi=50.0),
        doc="""[Hz] Stimulus amplitude""")

    c = arrays.FloatArray(
        label=":math:`c`",
        default=numpy.array([51.0, ]),
        range=basic.Range(lo=0.0, hi=100.0),
        doc="""[%].  Percentage coherence or motion strength. This parameter
        comes from experiments in MT cells.""")

    f = arrays.FloatArray(
        label=":math:`f`",
        default=numpy.array([1., ]), #0.45
        range=basic.Range(lo=0.0, hi=100.0),
        doc=""" Gain of MT firing rates""")

    state_variable_range = basic.Dict(
        label="State variable ranges [lo, hi]",
        default={"Sl": numpy.array([0., 1.]),
                 "Sr": numpy.array([0., 1.])},
        doc="n/a",
        order=-1
    )

    variables_of_interest = basic.Enumerate(
        label="Variables watched by Monitors",
        options=["Sl", "Sr"],
        default=["Sl"],
        select_multiple=True,
        doc="""default state variables to be monitored""",
        order=10)

    state_variables = ['Sl', 'Sr']

    def __init__(self, **kwargs):
        """
        .. May need to put kwargs back if we can't get them from trait...

        """

        super(WongWang, self).__init__(**kwargs)

        self._nvar = 2
        self.cvar = numpy.array([0], dtype=numpy.int32)

        #derived parameters
        self.I_mot_l = None
        self.I_mot_r = None

        LOG.debug('%s: inited.' % repr(self))

    def configure(self):
        """  """
        super(WongWang, self).configure()
        self.update_derived_parameters()


    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        The notation of those dynamic equations follows [WW_2007].
        Derivatives of s are multiplied by 0.001 constant to match ms time scale.
        """
        # add global coupling?
        sl = state_variables[0, :]
        sr = state_variables[1, :]

        c_0 = coupling[0, :]
        lc_0_l = local_coupling * sl
        lc_0_r = local_coupling * sr

        I_l = self.Jll * sl - self.Jlr*sr + self.I_mot_l + self.I_o + self.J_N * c_0 + self.J_N * lc_0_l
        I_r = self.Jrr * sr - self.Jrl*sl + self.I_mot_r + self.I_o + self.J_N * c_0 + self.J_N * lc_0_r

        r = lambda I_i: (self.a*I_i - self.b)*1./(1 - numpy.exp(-self.d*(self.a*I_i - self.b)))

        ds1 = -sl*1./ self.tau_s + (1 - sl) * self.gamma * r(I_l) * 0.001 # to ms
        ds2 = -sr*1./ self.tau_s + (1 - sr) * self.gamma * r(I_r) * 0.001 # to ms

        derivative = numpy.array([ds1, ds2])
        return derivative


    def update_derived_parameters(self):
        """
        Derived parameters
        """
        # Additional parameter g_stim introduced that controls I_mot strength
        self.I_mot_l = self.J_ext * self.mu_o * (1 + self.f * self.c *1. / 100)
        self.I_mot_r = self.J_ext * self.mu_o * (1 - self.f * self.c *1. / 100)
        if len(self.I_mot_l) > 1:
            self.I_mot_l = numpy.expand_dims(self.I_mot_l, -1)
            self.I_mot_r = numpy.expand_dims(self.I_mot_r, -1)



if __name__ == "__main__":
    # Do some stuff that tests or makes use of this module...
    LOG.info("Testing %s module..." % __file__)
    
    # Check that the docstring examples, if there are any, are accurate.
    import doctest
    doctest.testmod()
    
    #Initialise Models in their default state:
    WW = WongWang()
    WW.c = 11
    WW.configure()
        
    LOG.info("Model initialised in its default state without error...")
    
    LOG.info("Testing phase plane interactive ... ")
    
    # Check the Phase Plane
    from tvb.simulator.plot.phase_plane_interactive import PhasePlaneInteractive
    import tvb.simulator.integrators
        
    INTEGRATOR = tvb.simulator.integrators.HeunDeterministic(dt=0.1)
    ppi_fig = PhasePlaneInteractive(model=WW, integrator=INTEGRATOR)
    ppi_fig.show()

#EoF