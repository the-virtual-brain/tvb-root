import numpy as np
import tomli
import scipy.stats as stats
from tvb.simulator.lab import *

from tvb.basic.neotraits.api import Final, Attr
from tvb.datatypes.equations import TemporalApplicableEquation

def load_tvb_model_toml(model, path):
    with open(path, mode="rb") as fp:
        cfg = tomli.load(fp)
    
    model = model(
        **{k:np.r_[v] for k,v in cfg['parameters'].items()}
    )
    
    # this would be nicer with pattern matching
    stvar = cfg['attributes'].get('stvar', None)
    if stvar:
        model.stvar = np.r_[stvar]
    
    voi = cfg['attributes'].get('variables_of_interest', None)
    if voi:
        model.variables_of_interest = voi
    
    stvar_range = cfg['attributes'].get('state_variable_range', None)
    if stvar_range:
        for k,val in stvar_range.items():
            model.state_variable_range[k] = val
    for k in cfg['attributes'].keys():
        if k not in ['stvar', 'state_variable_range', 'variables_of_interest']:
            raise NotImplementedError(f'unsupported attribute: {k}')
    return model

def plot_normal(mu, sigma, ax, **kwds):
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    ax.plot( x, stats.norm.pdf(x, mu, sigma), **kwds )

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
