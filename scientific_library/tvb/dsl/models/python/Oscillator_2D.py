from tvb.basic.neotraits.api import NArray, Final, List, Range, HasTraits
import numpy

class Oscillator_2D:

    def __init__(self):

    # Define traited attributes for this model, these represent possible kwargs.
            
        tau = NArray(/
            label=":math:`tau`",
            default=numpy.array([1.0]),
            domain = Range(lo=1.0, hi=5.0, step=0.01),
            doc = """A time-scale hierarchy can be introduced for the state variables :math:`V` and :math:`W`. Default parameter is 1, which means no time-scale hierarchy."""
        )
        self.tau = tau
                    
        I = NArray(/
            label=":math:`I`",
            default=numpy.array([0.0]),
            domain = Range(lo=-5.0, hi=5.0, step=0.01),
            doc = """Baseline shift of the cubic nullcline"""
        )
        self.I = I
                    
        a = NArray(/
            label=":math:`a`",
            default=numpy.array([-2.0]),
            domain = Range(lo=-5.0, hi=5.0, step=0.01),
            doc = """Vertical shift of the configurable nullcline"""
        )
        self.a = a
                    
        b = NArray(/
            label=":math:`b`",
            default=numpy.array([-10.0]),
            domain = Range(lo=-20.0, hi=15.0, step=0.01),
            doc = """Linear slope of the configurable nullcline"""
        )
        self.b = b
                    
        c = NArray(/
            label=":math:`c`",
            default=numpy.array([0]),
            domain = Range(lo=-10.0, hi=10.0, step=0.01),
            doc = """Parabolic term of the configurable nullcline"""
        )
        self.c = c
                    
        d = NArray(/
            label=":math:`d`",
            default=numpy.array([0.02]),
            domain = Range(lo=0.0001, hi=1.0, step=0.0001),
            doc = """Temporal scale factor. Warning: do not use it unless you know what you are doing and know about time tides."""
        )
        self.d = d
                    
        e = NArray(/
            label=":math:`e`",
            default=numpy.array([3.0]),
            domain = Range(lo=-5.0, hi=5.0, step=0.0001),
            doc = """Coefficient of the quadratic term of the cubic nullcline."""
        )
        self.e = e
                    
        f = NArray(/
            label=":math:`f`",
            default=numpy.array([1.0]),
            domain = Range(lo=-5.0, hi=5.0, step=0.0001),
            doc = """Coefficient of the cubic term of the cubic nullcline."""
        )
        self.f = f
                    
        g = NArray(/
            label=":math:`g`",
            default=numpy.array([0.0]),
            domain = Range(lo=-5.0, hi=5.0, step=0.5),
            doc = """Coefficient of the linear term of the cubic nullcline."""
        )
        self.g = g
                    
        alpha = NArray(/
            label=":math:`alpha`",
            default=numpy.array([1.0]),
            domain = Range(lo=-5.0, hi=5.0, step=0.0001),
            doc = """Constant parameter to scale the rate of feedback from the slow variable to the fast variable."""
        )
        self.alpha = alpha
                    
        beta = NArray(/
            label=":math:`beta`",
            default=numpy.array([1.0]),
            domain = Range(lo=-5.0, hi=5.0, step=0.0001),
            doc = """Constant parameter to scale the rate of feedback from the slow variable to itself"""
        )
        self.beta = beta
                    
        gamma = NArray(/
            label=":math:`gamma`",
            default=numpy.array([1.0]),
            domain = Range(lo=-1.0, hi=1.0, step=0.1),
            doc = """Constant parameter to reproduce FHN dynamics where excitatory input currents are negative. It scales both I and the long range coupling term.."""
        )
        self.gamma = gamma
        
        state_variable_range = Final(
            label="State Variable ranges [lo, hi]",
            default={    "V": numpy.array([-2.0, 4.0]), 
				     "W": numpy.array([-6.0, 6.0])},
            doc="""V"""
        )

        state_variables = ('V', 'W')

        _nvar = 2
        cvar = numpy.array([0], dtype=numpy.int32)


    def _numpy_dfun(self, state_variables, coupling, local_coupling=0.0):

        V = state_variables[0, :]
        W = state_variables[1, :]

        #[State_variables, nodes]
        c_0 = coupling[0, :]

        # TODO why does it not default auto to default
        tau = self.tau.default
        I = self.I.default
        a = self.a.default
        b = self.b.default
        c = self.c.default
        d = self.d.default
        e = self.e.default
        f = self.f.default
        g = self.g.default
        alpha = self.alpha.default
        beta = self.beta.default
        gamma = self.gamma.default

        lc_0 = local_coupling * V
        derivative = numpy.empty_like(state_variables)

        # TODO fixed the acceptance of ** but it will process *** now as well. However not as an operand but as a value or node
        ev('d * tau * (alpha * W - f * V**3 + e * V**2 + g * V + gamma * I + gamma * c_0 + lc_0 * V)', out=derivative[0])
        ev('d * (a + b * V + c * V**2 - beta * W) / tau', out=derivative[1])

        return derivative

                                                        