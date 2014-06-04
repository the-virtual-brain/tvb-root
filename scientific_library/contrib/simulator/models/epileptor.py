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
The Epileptor model

.. moduleauthor:: Marmaduke Woodman <mw@eml.cc>

"""

# Third party python libraries
import numpy
import numexpr

#The Virtual Brain
from tvb.simulator.common import psutil, get_logger
LOG = get_logger(__name__)

import tvb.datatypes.arrays as arrays
import tvb.basic.traits.types_basic as basic 
import tvb.simulator.models as models


class HMJEpileptor(models.Model):
    """
    The Epileptor is a composite neural mass model of six dimensions which 
    has be crafted to model the phenomenology of epileptic seizures.

    This model, its motivation and derivation are currently in preparation
    for publication (Jirsa et al, 2013)

    .. automethod:: HMJEpileptor.__init__
    .. automethod:: HMJEpileptor.dfun
    """

    _ui_name = "Epileptor"
    ui_configurable_parameters = ["Iext", "Iext2", "r", "x0"]

    a = arrays.FloatArray(
        label="a",
        default=numpy.array([1]),
        doc="n/a",
        order=-1)

    b = arrays.FloatArray(
        label="b",
        default=numpy.array([3]),
        doc="n/a",
        order=-1)

    c = arrays.FloatArray(
        label="c",
        default=numpy.array([1]),
        doc="n/a",
        order=-1)

    d = arrays.FloatArray(
        label="d",
        default=numpy.array([5]),
        doc="n/a",
        order=-1)

    r = arrays.FloatArray(
        label="r",
        range=basic.Range(lo=0.0, hi=0.001, step=0.00005),
        default=numpy.array([0.00035]),
        doc="n/a",
        order=4)

    s = arrays.FloatArray(
        label="s",
        default=numpy.array([4]),
        doc="n/a",
        order=-1)

    x0 = arrays.FloatArray(
        label="x0",
        range=basic.Range(lo=-3.0, hi=0.0, step=0.1),
        default=numpy.array([-1.6]),
        doc="n/a",
        order=3)

    Iext = arrays.FloatArray(
        label="Iext",
        range=basic.Range(lo=1.5, hi=5.0, step=0.1),
        default=numpy.array([3.1]),
        doc="n/a",
        order=1)

    omega2 = arrays.FloatArray(
        label="omega2",
        default=numpy.array([0.1]),
        doc="n/a",
        order=-1)

    slope = arrays.FloatArray(
        label="slope",
        default=numpy.array([0.]),
        doc="n/a",
        order=-1)

    Iext2 = arrays.FloatArray(
        label="Iext2",
        range=basic.Range(lo=0.0, hi=1.0, step=0.05),
        default=numpy.array([0.45]),
        doc="n/a",
        order=2)

    tau = arrays.FloatArray(
        label="tau",
        default=numpy.array([10]),
        doc="n/a",
        order=-1)

    aa = arrays.FloatArray(
        label="aa",
        default=numpy.array([6]),
        doc="n/a",
        order=-1)
        
    Kpop1 = arrays.FloatArray(
        label="K_11",
        default=numpy.array([0.5]),
        range=basic.Range(lo=0.0, hi=4.0, step=0.5),
        doc='''Test parameter. Correspond to the coupling scaling. Move outside to be
        consistent with the general TVB implementation.''',
        order=-1)
        
    Kpop2 = arrays.FloatArray(
        label="K_22",
        default=numpy.array([0.2]),
        range=basic.Range(lo=0.0, hi=1.0, step=0.5),
        doc='''Test parameter. Correspond to the coupling scaling. Move outside to be
        consistent with the general TVB implementation.''',
        order=-1)

    state_variable_range = basic.Dict(
        label="State variable ranges [lo, hi]",
        default = {"y0": numpy.array([0., 1e-10]),
                   "y1": numpy.array([-5., 0.]),
                   "y2": numpy.array([3., 4.]),
                   "y3": numpy.array([0., 1e-10]),
                   "y4": numpy.array([0., 1e-10]),
                   "y5": numpy.array([0., 1e-2]) },
        doc = "n/a",
        order=-1
        )

    variables_of_interest = basic.Enumerate(
                              label = "Variables watched by Monitors",
                              options = ["y0", "y1", "y2", "y3", "y4", "y5"],
                              default = ["y0", "y3"],
                              select_multiple = True,
                              doc = """default state variables to be monitored""",
                              order = 10)

#    variables_of_interest = arrays.IntegerArray(
#        label="Variables watched by Monitors",
#        range=basic.Range(lo=0.0, hi=6.0, step=1.0),
#        default=numpy.array([0], dtype=numpy.int32),
#        doc="default state variables to be monitored",
#        order=10)


    def __init__(self, **kwargs):
        """
        """

        LOG.info("%s: init'ing..." % (str(self),))

        super(HMJEpileptor, self).__init__(**kwargs)

        #self._state_variables = ["y%d" % i for i in range(6)]
        #self._state_variables = ["y%d" % i for i in range(6)]
        self._nvar = 6
        self.cvar = numpy.array([0,3], dtype=numpy.int32)


        LOG.debug("%s: init'ed." % (repr(self),))

    def dfun(self, state_variables, coupling, local_coupling=0.0,
             array=numpy.array, where=numpy.where, concat=numpy.concatenate):
        """
        Computes the derivatives of the state variables of the Epileptor 
        with respect to time. 

        Implementation note: we expect this version of the Epileptor to be used
        in a vectorized manner. Concretely, y has a shape of (6, n) where n is 
        the number of nodes in the network. An consequence is that
        the original use of if/else is translated by calculated both the true and
        false forms and mixing them using a boolean mask.

        Variables of interest to be used by monitors: -y[0] + y[3]

        """

        """
        First population with high frequency burst and baseline jump - mechanisms
        is similar to a Hindmarsh-Rose scenario with two régimes connected by a
        slow trajectory (here y(3)).
        """

        y = state_variables
        n = y.shape[1]
        Iext = self.Iext + coupling[0, :] + local_coupling
        c_pop1 = coupling[0, :]
        c_pop2 = coupling[1, :]

        # if y(1)<0.
        #     ydot1 = y(2)-a*y(1)^3 + b*y(1)^2-y(3)+iext; 
        #     ydot2 = c-d*y(1)^2-y(2); 
        #     ydot3 =  r*(s*(y(1)-x0)  - y(3));   % energy consumption = 1 - available energy

        if_y1_lt_0 = concat([ (y[1] - self.a*y[0]**3 + self.b*y[0]**2 - y[2] + Iext).reshape((1, n, 1)),
                              (self.c - self.d*y[0]**2 - y[1]).reshape((1, n, 1)),
                              (self.r*(self.s*(y[0] - self.x0) - y[2] - self.Kpop1 * (c_pop1 - y[0]) )).reshape((1, n, 1)) ])

         # else
        # %    ydot1 = y(2) + (slope - y(4) -1.0*(y(3)-4))*y(1) - y(3)+iext; % this is just an
        # %    alternative representation, which worked well
        #     ydot1 = y(2) + (slope - y(4) + 0.6*(y(3)-4)^2)*y(1) -y(3)+iext; 
        # %   here the high energy burst is being generated through variation of the slope: 
        # %               1. via y(4) within the epileptic spike complex;         
        # %               2. via the expression with y(3), which causes more
        # %               oscillations at the beginning of the seizure (effect of the
        # %               energy available)
        #     ydot2 = c-d*y(1)^2-y(2);  
        #     ydot3 =   r*(s*(y(1)-x0)  - y(3));
        # end

        else_pop1 = concat([ (y[1] + (self.slope - y[3] + 0.6*(y[2]-4.0)**2)*y[0] - y[2] + Iext).reshape((1, n, 1)),
                         (self.c - self.d*y[0]**2 - y[1]).reshape((1, n, 1)),
                         (self.r*(self.s*(y[0] - self.x0) - y[2] - self.Kpop1 * (c_pop1 - y[0]))).reshape((1, n, 1)) ])

        pop1 = where(y[0] < 0., if_y1_lt_0, else_pop1)

        # % istim= 0*block(t,150,1);
        # 
        # % this is the second population that generates the big spike-wave complex
        # % preictally and within the seizure via a morris-lecar-jirsa (mlj) structure
        # 
        # if y(4)<-0.25
        #     ydot4 = -y(5)+ y(4)-y(4)^3 + iext2 + 2*y(6)-0.3*(y(3)-3.5) ; % these last two terms
        #     % put the population dynamics into the critical regime. in particular,
        #     % y(6) turns the oscillator on and off, whereas the y(3) term helps it to become precritical (critical fluctuations). 
        #     ydot5 = -y(5)/tau ;

        if_ = concat([ (-y[4] + y[3] - y[3]**3 + self.Iext2 + 2*y[5] - 0.3*(y[2] - 3.5) + self.Kpop2 * (c_pop2 - y[3])).reshape((1, n, 1)), (-y[4]/self.tau).reshape((1, n, 1)) ])
        # else
        #     ydot4 = -y(5)+ y(4)-y(4)^3 + iext2+ 2*y(6)-0.3*(y(3)-3.5); 
        #     ydot5 = (-y(5) + aa*(y(4)+0.25))/tau;   % here is the mlj structure
        # end

        else_pop2 = concat([ (-y[4] + y[3] - y[3]**3 + self.Iext2 + 2*y[5] - 0.3*(y[2] - 3.5) + self.Kpop2 * (c_pop2 - y[3])).reshape((1, n, 1)), ((-y[4] + self.aa*(y[3] + 0.25))/self.tau).reshape((1, n, 1)) ])


        pop2 = where(y[3] < -0.25, if_, else_pop2)

        # 
        #  ydot6 = -0.01*(y(6)-0.1*y(1)) ;

        energy = array([ -0.01*(y[5] - 0.1*y[0])])
        
        # 
        # ydot = [ydot1;ydot2;ydot3;ydot4;ydot5;ydot6];

        return concat((pop1, pop2, energy))