from sympy import symbols, cse, Matrix, exp
Vector = lambda *args: Matrix([arg for arg in args])
syms = 'nu_max, r, v0, y1, y2, a_1, J, y0, a_3, y3, y4, y5, lrc, short_range_coupling, A, B, a, b, mu, a_2, a_4'
nu_max, r, v0, y1, y2, a_1, J, y0, a_3, y3, y4, y5, lrc, short_range_coupling, A, B, a, b, mu, a_2, a_4 = symbols(syms)
sigm_y1_y2 = 2 * nu_max / (1 + exp(r * (v0 - (y1 - y2))))
sigm_y0_1  = 2 * nu_max / (1 + exp(r * (v0 - (a_1 * J * y0))))
sigm_y0_3  = 2 * nu_max / (1 + exp(r * (v0 - (a_3 * J * y0))))
state = Vector(y0, y1, y2, y3, y4, y5)
param = Vector(nu_max, r, v0, a_1, J, a_3, lrc, short_range_coupling, A, B, a, b, mu, a_2, a_4)
drift = Vector(
    y3,
    y4,
    y5,
    A * a * sigm_y1_y2 - 2.0 * a * y3 - a ** 2 * y0,
    A * a * (mu + a_2 * J * sigm_y0_1 + lrc + short_range_coupling)
        - 2.0 * a * y4 - a ** 2 * y1,
    B * b * (a_4 * J * sigm_y0_3) - 2.0 * b * y5 - b ** 2 * y2,
)
n = 6
flatjac = lambda x, y: x.diff(y).reshape(len(x)*len(y)).tolist()
terms = drift.reshape(1, n).tolist()[0] + flatjac(drift, state) + flatjac(drift, param)

from tvb.simulator.models.jansen_rit import JansenRit
defaults = {}
for (p,) in param.tolist():
    key = str(p)
    if hasattr(JansenRit, key):
        defaults[key] = getattr(JansenRit, key).default.item()
use_defaults = True
if use_defaults:
    terms = [_.subs(defaults).simplify() for _ in terms]

do_cse = True
if do_cse:
    reps, rexs = cse(terms)
else:
    reps, rexs = [], terms

drift_rex = rexs[:n]
jac_state = rexs[n:n+n*n]
jac_param = rexs[n+n*n:]

lines = []

for i, var in enumerate(state):
    lines.append(f'{var} = state[{i}]')

for lhs, rhs in reps:
    lines.append(f'{lhs} = {rhs}')

for i, par in enumerate(param):
    if str(par) not in defaults:
        lines.append(f'{par} = param[{i}]')

for i, ex in enumerate(drift_rex):
    lines.append(f'xt[{i}] = {ex}')

for i, (js, jp) in enumerate(zip(jac_state, jac_param)):
    if js != 0:
        lines.append(f"jxtx[{i//n}, {i%n}] = {js}")
    if jp != 0:
        lines.append(f"jxtp[{i//n}, {i%n}] = {jp}")

# embed Euler and update multiple at once

header = 'def gufunc(state, param, xt, jxtx, jxtp):\n'

code = header + '\n'.join(['    ' + line for line in lines])

ns = {}
exec('from math import *', ns)
exec(code, ns)
gufunc = ns['gufunc']
from numba import njit
import numpy as np
jit_gufunc = njit(gufunc)
state = np.random.randn(n)
param = np.random.randn(len(param))
xt = np.zeros((n, ))
jxtx = np.zeros((n, n))
jxtp = np.zeros((n, len(param)))
jit_gufunc(state, param, xt, jxtx, jxtp)