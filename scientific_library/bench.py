
"""
This file benchmarks isolated components in the scientific library.

"""

import sys
import time
import importlib
import numpy

# util {{{

try:
    import cpuinfo
    print 'CPU is a %s' % (cpuinfo.get_cpu_info()['brand'], )
except ImportError:
    print 'consider `pip py-cpuinfo` to include CPU info in report'

def get_subclasses(submodname, baseclassname):
    modname = 'tvb.simulator.' + submodname
    mod = importlib.import_module(modname, modname) 
    baseclass = getattr(mod, baseclassname)
    ignore_names = baseclass._base_classes
    for key in dir(mod):
        attr = getattr(mod, key)
        if (key != baseclassname
            and isinstance(attr, type)
            and issubclass(attr, baseclass)
            and key not in ignore_names):
            yield attr

# }}}

# integrators {{{

def integrators():
    return get_subclasses('integrators', 'Integrator')

def nop_dfun(X, coupling, local_coupling):
    return X

def eps_for_Integrator(Integrator, n_node, time_limit=0.5):
    integ = Integrator()
    integ.configure()
    if 'Stochastic' in Integrator.__name__:
        integ.noise.dt = integ.dt
    X = numpy.random.randn(n_node)
    thunk = lambda : integ.scheme(X, nop_dfun, None, None, 0.0)
    thunk()
    # start timing
    tic = time.time()
    n_eval = 0
    while (time.time() - tic) < time_limit:
        thunk()
        n_eval += 1
    toc = time.time()
    return n_eval / (toc - tic)

# }}}

# models {{{ 

def models():
    return get_subclasses('models', 'Model')

def randn_state_for_model(model, n_node):
    shape = (model.nvar, n_node, model.number_of_modes)
    state = numpy.random.randn(*shape)
    return state


def zero_coupling_for_model(model, n_node):
    n_cvar = len(model.cvar)
    shape = (n_cvar, n_node, model.number_of_modes)
    coupling = numpy.zeros(shape)
    return coupling


def eps_for_Model(Model, n_node, time_limit=0.5):
    model = Model()
    model.configure()
    state = randn_state_for_model(model, n_node)
    coupling = zero_coupling_for_model(model, n_node)
    # throw one away in case of initialization
    model.dfun(state, coupling)
    # start timing
    tic = time.time()
    n_eval = 0
    while (time.time() - tic) < time_limit:
        model.dfun(state, coupling)
        n_eval += 1
    toc = time.time()
    return n_eval / (toc - tic)

# }}}

def eps_report_for_components(comps, eps_func):
    n_nodes = [2 << i for i in range(14)]
    sys.stdout.write('%30s' % ('n_node',))
    [sys.stdout.write('%06s' % (n, )) for n in n_nodes]
    sys.stdout.write('\n')
    sys.stdout.flush()
    for comp in comps:
        name = comp.__name__ if isinstance(comp, type) else comp.__class__.__name__
        sys.stdout.write('%30s' % (name, ))
        for n_node in n_nodes:
            deps = eps_func(comp, n_node)
            sdeps = '%0.1f' % (deps/1e3,)
            sys.stdout.write('%06s' % (sdeps, ))
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()


if __name__ == '__main__':
    print 'units in kHz'
    print 'benchmarking models'
    eps_report_for_components(models(), eps_for_Model)
    print 'benchmarking integrators'
    from tvb.simulator.integrators import RungeKutta4thOrderDeterministic
    integs = list(integrators()) + [RungeKutta4thOrderDeterministic]
    eps_report_for_components(integs, eps_for_Integrator)

# vim: sw=4 sts=4 ai et foldmethod=marker
