import numpy as np
import tomli
from alpha_stim import AlphaFunction
from tvb.simulator.lab import *

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


def make_alpha_stim(conn, stim_params=None):
    if stim_params is None:
        stim_params = AlphaFunction.parameters.default()

    stim = patterns.StimuliRegion(
        temporal=AlphaFunction(parameters=stim_params),
        connectivity=conn,
        weight=np.ones(len(conn.weights)))

    return stim

def configure_sim(path, stim=False, stim_params=None, with_noise=False):
    conn = connectivity.Connectivity.from_file()
    conn.speed = np.r_[np.inf]

    if stim:
        if stim_params is None:
            stim_params = AlphaFunction.parameters.default()

        stim = patterns.StimuliRegion(
            temporal=AlphaFunction(parameters=stim_params),
            connectivity=conn,
            weight=np.ones(len(conn.weights)))
    else:
        stim=None

    if with_noise:
        integrator = integrators.HeunStochastic(
                noise=noise.Additive(
                    nsig=np.r_[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] ),
                dt = 0.1,
        )
    else:
        integrator = integrators.HeunDeterministic( dt = 0.1,)



    sim = simulator.Simulator(
        model=load_tvb_model_toml(models.ZerlautAdaptationSecondOrder,path),
        connectivity=conn,
        conduction_speed=conn.speed.item(),
        coupling=coupling.Linear(a=np.r_[0.0]),
        integrator=integrator,
        monitors=[monitors.TemporalAverage(period=1.0)],
        stimulus=stim,
    ).configure()

    return sim

import scipy.stats as stats

def plot_normal(mu, sigma, ax, **kwds):
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    ax.plot( x, stats.norm.pdf(x, mu, sigma), **kwds )
