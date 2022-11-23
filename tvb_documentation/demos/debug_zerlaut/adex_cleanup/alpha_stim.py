from tvb.basic.neotraits.api import Final, Attr
from tvb.datatypes.equations import TemporalApplicableEquation
import numpy as np

class AlphaFunction(TemporalApplicableEquation):
    equation = Final(
        label="Alpha Function Equation",
        default="NumExpr doesn't support heaviside function",
        doc=""":math:`amp * ( heaviside(t_0 - var ,0.5)*exp(-(var - t_0) ** 2 / (T_1 ** 2)) + heaviside(var - t_0, 0.5)*nexp(-(var - t_0) ** 2 / (T_2 ** 2))) + b` """)

    parameters = Attr(
        field_type=dict,
        label="Alpha Function Parameters",
        default=lambda: {"amp": 1.0, "t_0": 1000., "T_1": 100.0, "T_2": 100.0, "b":0.})

    def evaluate(self, var):
        amp, t_0, T_1, T_2, b = self.parameters['amp'],self.parameters['t_0'], self.parameters['T_1'], self.parameters['T_2'], self.parameters['b']
        return amp * ( np.heaviside(t_0 - var ,0.5)*np.exp(-(var - t_0) ** 2 / (T_1 ** 2)) + np.heaviside(var - t_0, 0.5)*np.exp(-(var - t_0) ** 2 / (T_2 ** 2))) + b
