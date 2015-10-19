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
#   The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

if __name__ == "__main__":
    from tvb.tests.library import setup_test_console_env
    setup_test_console_env()

import numpy
import unittest
from tvb.tests.library.base_testcase import BaseTestCase
from tvb.simulator.monitors import MonitorTransforms
from tvb.simulator import models, coupling, integrators, simulator
from tvb.datatypes import connectivity
from tvb.simulator.monitors import Raw, TemporalAverage


class MonitorTransformsTests(BaseTestCase):

    def test_split(self):
        mt = MonitorTransforms('a,c', '1,2', delim=',')
        self.assertEqual(len(mt.pre), 2)
        self.assertEqual(len(mt.post), 2)
        mt = MonitorTransforms('a;c', 'exp(x);2.234')
        self.assertEqual(len(mt.pre), 2)
        self.assertEqual(len(mt.post), 2)

    def test_pre_1(self):
        mt = MonitorTransforms('a', 'b;c')
        self.assertEqual(1, len(mt.pre))

    def test_post_1(self):
        mt = MonitorTransforms('a;b', 'c')
        self.assertEqual(1, len(mt.post))

    # def _shape_fail(self):
    #     MonitorTransforms('1,2,3', '2;3', delim=',')
    #
    # def test_shape_fail(self):
    #     self.assertRaises(Exception, self._shape_fail)

    def _syntax_fail(self, pre, post):
        MonitorTransforms(pre, post)

    def test_syntax(self):
        self.assertRaises(SyntaxError, self._syntax_fail, 'a=3', '23.234')
        self.assertRaises(SyntaxError, self._syntax_fail, 'a+b/c*f(a,b)', 'f=23')

    def test_noop_post(self):
        mt = MonitorTransforms('a;b;c', '2.34*(pre+1.5);;')
        self.assertEqual(len(mt.post), 3)

    def _fail_noop_pre(self):
        MonitorTransforms(';;', ';;')

    def test_noop_pre_fail(self):
        self.assertRaises(SyntaxError, self._fail_noop_pre)

    def test_pre(self):
        state = numpy.r_[:4].reshape((1, -1, 1))
        # check expr correctly evaluated
        mt = MonitorTransforms('x0**2', '')
        out = mt.apply_pre(state)
        self.assertEqual(out[0, -1, 0], 9)
        # check correct shape
        n_expr = numpy.random.randint(5, 10)
        pre_expr = ';'.join([str(i) for i in range(n_expr)])
        mt = MonitorTransforms(pre_expr, '')
        out = mt.apply_pre(state)
        self.assertEqual(n_expr, out.shape[0])

    def test_post(self):
        state = numpy.tile(numpy.r_[:4], (2, 1)).reshape((2, -1, 1))
        state[1] *= 2
        # check expr eval correct
        mt = MonitorTransforms('0;0', 'mon;')
        _, out = mt.apply_post((0.0, state))
        self.assertEqual(3, out.flat[3])
        self.assertEqual(6, out.flat[7])
        mt = MonitorTransforms('0;0', 'mon;mon**2-1')
        _, out = mt.apply_post((0.0, state))
        self.assertEqual(3, out.flat[3])
        self.assertEqual(35, out.flat[7])
        # check correct shape
        n_expr = numpy.random.randint(5, 10)
        state = numpy.tile(numpy.r_[:4], (n_expr, 1)).reshape((n_expr, -1, 1))
        post_expr = ';'.join([str(i) for i in range(n_expr)])
        mt = MonitorTransforms('0', post_expr)
        _, out = mt.apply_post((0.0, state))
        self.assertEqual(n_expr, out.shape[0])

    def test_user_tags(self):
        pre = '0;0'
        post = 'mon;mon**2-1'
        raw = Raw(pre_expr=pre, post_expr=post)
        tags = raw._transform_user_tags()
        self.assertIn('user_tag_1', tags)
        self.assertIn('user_tag_2', tags)
        self.assertEqual(tags['user_tag_1'], pre)
        self.assertEqual(tags['user_tag_2'], post)


class MonitorTransformsInSimTest(BaseTestCase):

    def _run_sim(self, length, model, *mons):
        sim = simulator.Simulator(
            model=model,
            connectivity=connectivity.Connectivity(load_default=True),
            coupling=coupling.Linear(),
            integrator=integrators.EulerDeterministic(),
            monitors=mons)
        sim.configure()
        ys = []
        for (t, y), in sim(simulation_length=length):
            ys.append(y)
        return sim, numpy.array(ys)

    def test_expr_pre(self):
        sim, ys = self._run_sim(5, models.Generic2dOscillator(), Raw(pre_expr='V;W;V**2;W-V',
                                                                     post_expr='mon;mon;mon;mon'))
        self.assertTrue(hasattr(sim.monitors[0], '_transforms'))
        v, w, v2, wmv = ys.transpose((1, 0, 2, 3))
        self.assertTrue(numpy.allclose(v ** 2, v2))
        self.assertTrue(numpy.allclose(w - v, wmv))

    def test_expr_post(self):
        sim, ys = self._run_sim(5, models.Generic2dOscillator(),
                                Raw(pre_expr='V;W;V;W', post_expr=';;mon**2; exp(mon)'))
        self.assertTrue(hasattr(sim.monitors[0], '_transforms'))
        v, w, v2, ew = ys.transpose((1, 0, 2, 3))
        self.assertTrue(numpy.allclose(v ** 2, v2))
        self.assertTrue(numpy.allclose(numpy.exp(w), ew))

    def test_expr_tim(self):
        sim, ys = self._run_sim(5, models.Epileptor(), Raw(pre_expr='-y0+y3;y2', post_expr='mon;mon'))
        self.assertTrue(hasattr(sim.monitors[0], '_transforms'))
        lfp, slow = ys.transpose((1, 0, 2, 3))

    def test_period_handling(self):
        """Test that expression application working for monitors with a period."""
        sim, ys = self._run_sim(5, models.Generic2dOscillator(), TemporalAverage(pre_expr='V+W'))



def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(MonitorTransformsInSimTest))
    test_suite.addTest(unittest.makeSuite(MonitorTransformsTests))
    return test_suite



if __name__ == "__main__":
    #So you can run tests from this package individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE)
