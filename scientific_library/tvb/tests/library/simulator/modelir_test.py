# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
Tests for template code generation using the simulator instance as an
intermediate representation. 

Several TestCases here are related to ensuring generation proceeds
correctly and is verified against the simulator result itself, so
these can later be refactored as per-backend conformance tests.

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import math
import unittest
import pytest
from mako.template import Template
from mako.lookup import TemplateLookup
from mako.exceptions import text_error_template
import numpy as np

# maybe import pycuda
try:
    import pycuda
    import pycuda.autoinit
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule
    from pycuda.driver import Out, In, InOut
except Exception as exc:
    pycuda = None
    pycuda_why_not = exc

from tvb.simulator import templates
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.coupling import Sigmoidal, Linear
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.integrators import EulerDeterministic
from tvb.simulator.monitors import Raw
from tvb.simulator.simulator import Simulator

@unittest.skipUnless(pycuda, 'requires working PyCUDA & GPU')
class TestPyCUDABasics(unittest.TestCase):
    "Tests validating basic usage of PyCUDA and working GPU."

    def test_demo(self):
        "Test basic PyCUDA example."
        mod = SourceModule("""
        __global__ void multiply_them(float *dest, float *a, float *b)
        {
          const int i = threadIdx.x;
          dest[i] = a[i] * b[i];
        }
        """)
        multiply_them = mod.get_function("multiply_them")
        a = np.random.randn(400).astype(np.float32)
        b = np.random.randn(400).astype(np.float32)
        dest = np.zeros_like(a)
        multiply_them(
            drv.Out(dest), drv.In(a), drv.In(b),
            block=(400,1,1), grid=(1,1))
        np.testing.assert_allclose(dest, a*b)

    @staticmethod
    def flow(dX, X, Z):
        dt = 0.1
        sqrt_dt = math.sqrt(0.1)
        x, y = X
        dX[0] = (x - x**3/3 + y)*3
        dX[1] = (1.01 - x) / 3
        X += dt * dX + sqrt_dt * 0.1 * Z

    def test_array(self):
        "Test use of GPU arrays as NumPy look-alike."
        import pycuda.gpuarray as gpu
        X = np.random.rand(2,512).astype('f')
        Z = np.random.randn(2,512).astype('f')
        dX = np.zeros_like(X)
        gX = gpu.to_gpu(X)
        gdX = gpu.zeros_like(gX)
        gZ = gpu.to_gpu(Z)
        self.flow(dX, X, Z)
        self.flow(gdX, gX, gZ)
        np.testing.assert_allclose(gX.get(), X, 1e-5, 1e-6)


class MakoUtilMix:

    @property
    def lookup(self):
        lookup = TemplateLookup(directories=[templates.__path__[0]])
        return lookup

    def _render_template(self, source, content):
        template = Template(source, lookup=self.lookup, strict_undefined=True)
        try:
            source = template.render(**content)
        except Exception as exc:
            print(text_error_template().render())
            raise exc
        return source

    def _build_py_func(self, template_source, content, name='kernel'):
        "Build and retrieve a Python function from template."
        source = self._render_template(template_source, content)
        globals_ = {}
        try:
            exec(source, globals_)
        except Exception as exc:
            print(source)
            raise exc
        return globals_[name]

    def _insert_line_numbers(self, source):
        lines = source.split('\n')
        numbers = range(1, len(lines) + 1)
        nu_lines = ['%03d\t%s' % (nu, li) for (nu, li) in zip(numbers, lines)]
        nu_source = '\n'.join(nu_lines)
        return nu_source

    def _build_cu_func(self, template_source, content, name='kernel', print_source=False):
        "Build and retrieve a Python function from template."
        source = self._render_template(template_source, content)
        if print_source:
            print(source)
        try:
            module = SourceModule(source)
        except pycuda.driver.CompileError as exc:
            print(self._insert_line_numbers(source))
            raise exc
        func = module.get_function(name)
        return func


class TestMako(unittest.TestCase, MakoUtilMix):
    "Test basic Mako usage."

    def _assert_flow_ok(self, template_source, content):
        cg_flow = self._build_py_func(template_source, content, "flow")
        dX, X, Z = np.random.randn(3,2,10)
        cg_X = X.copy()
        cg_flow(dX, cg_X, Z)
        TestPyCUDABasics.flow(dX, X, Z)
        np.testing.assert_allclose(X, cg_X)

    def test_template(self):
        "Test basic use of Mako."
        template = """def flow(dX, X, Z):
    x, y = X
    dX[0] = ${drift_X}
    dX[1] = ${drift_Y}
    X += ${dt} * dX + ${math.sqrt(dt)} * ${sigma} * Z     
"""
        content = dict(
            math=math,
            drift_X="(x - x**3/3 + y)*3",
            drift_Y="(1.01 - x) / 3",
            dt=0.1,
            sigma=0.1,
        )
        self._assert_flow_ok(template, content)

    def test_template_loop(self):
        "Test a Mako loop over dfuns"
        template = """def flow(dX, X, Z):
    ${','.join(svars)} = X
    % for rhs in dfuns:
    dX[${loop.index}] = ${rhs}
    % endfor
    X += ${dt} * dX + ${math.sqrt(dt)} * ${sigma} * Z
"""
        content = dict(
            math=math,
            svars="x y".split(),
            dfuns=["(x - x**3/3 + y)*3",
                   "(1.01 - x) / 3"],
            dt=0.1,
            sigma=0.1,
        )
        self._assert_flow_ok(template, content)

    def test_defs(self):
        "Test use of Mako defs to better structure template."
        # NB. this isn't really structured since defs can access args
        template = """${flow()}

<%def name="flow()">
def flow(dX, X, Z):
    ${','.join(svars)} = X
    ${diffeqs()}
    ${em_update()}
    # done
</%def>

<%def name="diffeqs()">
% for rhs in dfuns:
    dX[${loop.index}] = ${rhs}
% endfor
</%def>

<%def name="em_update()">
    X += ${dt} * dX + ${math.sqrt(dt)} * ${sigma} * Z
</%def>
"""
        content = dict(
            math=math,
            svars="x y".split(),
            dfuns=["(x - x**3/3 + y)*3",
                   "(1.01 - x) / 3"],
            dt=0.1,
            sigma=0.1,
        )
        self._assert_flow_ok(template, content)

    def test_mpr_dfun(self):
        "Test MPR dfun against built-in model dfun."
        mpr = MontbrioPazoRoxin()
        state, coupling = np.random.rand(2,2,32).astype('f')
        state[1] -= 2.0
        coupling[1] -= 2.0
        drift = mpr.dfun(state, coupling)
        template = """import numpy as np
pi = np.pi
def flow(state, coupling):
    dX = np.zeros_like(state)
    % for par in model.parameter_names:
    ${par} = ${getattr(model,par)[0]}
    % endfor
    ${','.join(svars)} = state
    ${','.join(cterms)} = coupling
    % for svar in svars:
    dX[${loop.index}] = ${dfuns[svar]}
    % endfor
    return dX
"""
        content = dict(
            math=math,
            np=np,
            model=mpr,
            svars=mpr.state_variables,
            dfuns=mpr.state_variable_dfuns,
            cterms=mpr.coupling_terms,
            dt=0.1,
            sigma=0.1,
        )
        cg_flow = self._build_py_func(template, content, 'flow')
        cg_drift = cg_flow(state, coupling)
        self.assertTrue(np.isfinite(cg_drift).all())
        self.assertTrue(np.isfinite(drift).all())
        np.testing.assert_allclose(cg_drift, drift, 1e-5, 1e-6)

    def test_sigmoidal_cfun(self):
        "Test cfun code gen against builtin sigmoidal cfun."
        cfun = Sigmoidal()
        template = """
import numpy as np
exp = np.exp
def pre_post(X):
    gX = np.zeros_like(X)
    % for par in cfun.parameter_names:
    ${par} = ${getattr(cfun,par)[0]}
    % endfor
    for i in range(X.shape[0]):
        gx = 0
        x_i = X[i]
        for j in range(X.shape[0]):
            x_j = X[j]
            gx += ${cfun.pre_expr}
        gX[i] = ${cfun.post_expr}
    return gX
"""
        content = dict(
            cfun=cfun
        )
        cg_pre_post = self._build_py_func(template, content, 'pre_post')
        def pre_post(X):
            x_i, x_j = X, X.reshape((-1, 1))+0*X
            gx = cfun.pre(x_i, x_j).sum(axis=0)
            return cfun.post(gx)
        # now evaluate
        X = np.random.randn(32).astype('f')
        cg_gX = cg_pre_post(X)
        gX = pre_post(X)
        np.testing.assert_allclose(cg_gX, gX, 1e-5, 1e-6)


@unittest.skipUnless(pycuda, 'requires working PyCUDA')
class TestPyCUDAModel(unittest.TestCase, MakoUtilMix):
    "Test model definitions in form of generated CUDA kernels."

    def test_mpr_dfun(self):
        "Test generated CUDA MPR dfun against built-in."
        mpr = MontbrioPazoRoxin()
        state, coupling = np.random.rand(2, 2, 32).astype('f')
        state[1] -= 2.0
        coupling[1] -= 2.0
        drift = mpr.dfun(state, coupling).astype('f')
        template = '<%include file="test_cu_mpr_dfun.mako"/>'
        content = dict(np=np, model=mpr, debug=False)
        cg_flow = self._build_cu_func(template, content, 'mpr_dfun')
        cg_drift = np.zeros_like(drift)
        # TODO unit test higher level driver separately
        # TODO use prepared calls
        cg_flow(np.uintc(cg_drift.shape[1]),
                InOut(cg_drift), InOut(state), InOut(coupling),
                grid=(1,1), block=(state.shape[-1],1,1))
        self.assertTrue(np.isfinite(cg_drift).all())
        self.assertTrue(np.isfinite(drift).all())
        np.testing.assert_allclose(cg_drift, drift, 1e-5, 1e-6)

    def test_mpr_traj(self):
        "Test generated time stepping against built in."
        mpr = MontbrioPazoRoxin()
        state, coupling = np.random.rand(2, 2, 32).astype('f')
        state[1] -= 2.0
        coupling[1] -= 2.0
        state_copy, coupling_copy = state.copy(), coupling.copy()
        nt = 100
        t, y = mpr.stationary_trajectory(
            initial_conditions=state, coupling=coupling,
            n_step=nt-1, n_skip=1, dt=0.01)
        nt, nsvar, nmode, nnode = y.shape
        self.assertEqual(y.shape, (nt, 2, 1, coupling.shape[1]))
        template = '<%include file="test_cu_mpr_traj.mako"/>'
        content = dict(np=np, model=mpr, dt=0.01, nt=nt, debug=False)
        cg_traj = self._build_cu_func(template, content, 'mpr_traj', print_source=True)
        cg_drift = np.empty_like(state)
        cg_trace = np.empty((nt,2,coupling.shape[1]),'f')
        cg_traj(np.uintc(cg_drift.shape[1]),
                Out(cg_drift), In(state_copy), In(coupling_copy),
                Out(cg_trace),
                grid=(1,1), block=(state.shape[-1],1,1))
        self.assertTrue(np.isfinite(cg_trace).all())
        np.testing.assert_allclose(cg_trace, y[:,:,0], 1e-5, 1e-6)


class TestSimODE(unittest.TestCase, MakoUtilMix):

    def _create_sim(self, inhom_mmpr=False):
        mpr = MontbrioPazoRoxin()
        conn = Connectivity.from_file()
        if inhom_mmpr:
            dispersion = 1 + np.random.randn(conn.weights.shape[0])*0.1
            mpr = MontbrioPazoRoxin(eta=mpr.eta*dispersion)
        conn.speed = np.r_[np.inf]
        dt = 0.01
        integrator = EulerDeterministic(dt=dt)
        sim = Simulator(connectivity=conn, model=mpr, integrator=integrator, 
            monitors=[Raw()],
            simulation_length=0.1)  # 10 steps
        sim.configure()
        self.assertTrue((conn.idelays == 0).all())
        state = sim.current_state.copy()[:,:,0].astype('f')
        self.assertEqual(state.shape[0], 2)
        self.assertEqual(state.shape[1], conn.weights.shape[0])
        (t,y), = sim.run()
        return sim, state, t, y

    def _check_match(self, expected, actual):
        # check we don't have numerical errors
        self.assertTrue(np.isfinite(actual).all())
        # check tolerances
        maxtol = np.max(np.abs(actual[0] - expected[0,:,:,0]))
        for t in range(1, len(actual)):
            print(t, 'tol:', np.max(np.abs(actual[t] - expected[t,:,:,0])))
            np.testing.assert_allclose(actual[0], expected[0, :, :, 0], 2e-5*t, 1e-5*t)      

    def test_mpr_cu1(self):
        "Test generated time stepping network without delay."
        sim, state, t, y = self._create_sim()
        template = '<%include file="test_cu_mpr_net_no_delay.mako"/>'
        content = dict(kernel_name='mpr_net',
            np=np, model=sim.model,
            dt=sim.integrator.dt, nt=len(t), 
            cfun_a=sim.coupling.a[0], debug=False)
        cu_loop = self._build_cu_func(template, content, 'mpr_net')
        dX = state.copy()
        weights = sim.connectivity.weights.T.copy().astype('f')
        yh = np.empty((len(t),)+state.shape, 'f')
        with self.assertRaises(AssertionError):
            self._check_match(y, yh)
        cu_loop(np.uintc(state.shape[1]),
            Out(dX), In(state), In(weights), Out(yh), 
            grid=(1,1), block=(128,1,1))
        self._check_match(y, yh)

    # a functionally identical test which generates directly from a
    # configured simulator instance, and uses the coupling function
    # template instead of monolithically generating everything.

    def test_mpr_cu2(self):
        "Test generated CUDA kernel directly from Simulator instance."
        sim, state, t, y = self._create_sim(inhom_mmpr=True)
        template = '<%include file="cu-sim-ode.mako"/>'
        kernel = self._build_cu_func(template, dict(sim=sim, pi=np.pi))
        dX = state.copy()
        weights = sim.connectivity.weights.T.copy().astype('f')
        eta = sim.model.eta.astype('f')
        yh = np.empty((len(t),)+state.shape, 'f')
        kernel(
            In(state), In(weights), Out(yh), In(eta),
            grid=(1,1), block=(128,1,1))
        self._check_match(y, yh)


# each component has it's main template like cu-coupling.mako
# TODO inject a namespace "target" with concrete rendering functions

class TestCoupling(unittest.TestCase, MakoUtilMix):     

    def _eval_cfun_no_delay(self, cfun, weights, X):
        nsvar, nnode = X.shape
        x_i, x_j = X.reshape((nsvar, 1, nnode)), X.reshape((nsvar, nnode, 1))
        gx = (weights * cfun.pre(x_i+x_j*0, x_j+x_i*0)).sum(axis=1)
        return cfun.post(gx)

    def _test_cu_cfun(self, cfun):
        class sim:  # dummy
            model = MontbrioPazoRoxin()
            coupling = cfun
        template = '''
<%include file="cu-coupling.mako"/>
__global__ void kernel(float *state, float *weights, float *cX) {
    coupling(threadIdx.x, ${n_node}, cX, weights, state);
}
'''
        content = dict(n_node=128, sim=sim)
        kernel = self._build_cu_func(template, content)
        state = np.random.rand(2, content['n_node']).astype('f')
        weights = np.random.randn(state.shape[1], state.shape[1]).astype('f')
        cX = np.zeros_like(state)
        kernel(In(state), In(weights), Out(cX), 
            grid=(1,1), block=(content['n_node'],1,1))
        expected = self._eval_cfun_no_delay(sim.coupling, weights, state)
        np.testing.assert_allclose(cX, expected, 1e-5, 1e-6)

    def test_cu_linear(self): self._test_cu_cfun(Linear())
    # def test_cu_sigmoidal(self): self._test_cu_cfun(Sigmoidal())

    def _test_nb_cfun(self, cfun):
        class sim:  # dummy
            model = MontbrioPazoRoxin()
            coupling = cfun
        template = '<%include file="nb-coupling.mako"/>'
        kernel = self._build_py_func(template, dict(sim=sim), name='coupling')
        state = np.random.rand(2, 128).astype('f')
        weights = np.random.randn(state.shape[1], state.shape[1]).astype('f')
        cX = np.zeros_like(state)
        kernel(cX, weights.T, state)
        expected = self._eval_cfun_no_delay(sim.coupling, weights, state)
        np.testing.assert_allclose(cX, expected, 1e-5, 1e-6)

    def test_nb_linear(self): self._test_nb_cfun(Linear())
    # def test_nb_sigmoidal(self): self._test_nb_cfun(Sigmoidal())

    def _test_np_cfun(self, cfun):
        class sim:  # dummy
            model = MontbrioPazoRoxin()
            coupling = cfun
        template = '<%include file="np-coupling.mako"/>'
        kernel = self._build_py_func(template, dict(sim=sim), name='coupling')
        state = np.random.rand(2, 128).astype('f')
        weights = np.random.randn(state.shape[1], state.shape[1]).astype('f')
        cX = np.zeros_like(state)
        kernel(cX, weights.T, state)
        expected = self._eval_cfun_no_delay(sim.coupling, weights, state)
        np.testing.assert_allclose(cX, expected, 1e-5, 1e-6)

    def test_np_linear(self): self._test_np_cfun(Linear())
    # def test_nb_sigmoidal(self): self._test_nb_cfun(Sigmoidal())