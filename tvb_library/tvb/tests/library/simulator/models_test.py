# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

"""
Test for tvb.simulator.models module

.. moduleauthor:: Paula Sanz Leon <paula@tvb.invalid>

"""
import numpy
from tvb.basic.neotraits.api import Final, List
from tvb.simulator import models
from tvb.simulator.models.base import Model
from tvb.tests.library.base_testcase import BaseTestCase


class ModelTestBounds(Model):
    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={"x1": numpy.array([0.0, 1.0]),
                 "x2": numpy.array([None, 1.0]),
                 "x3": numpy.array([0.0, None]),
                 "x4": numpy.array([-numpy.inf, numpy.inf])
                 },
        doc="""The values for each state-variable should be set to encompass
            the boundaries of the dynamic range of that state-variable. Set None for one-sided boundaries""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=('x1', 'x2', 'x3', 'x4', 'x5'),
        default=('x1', 'x2', 'x3', 'x4', 'x5'),
        doc="""default state variables to be monitored""")

    state_variables = ['x1', 'x2', 'x3', 'x4', 'x5']

    state_variable_range = Final(
        default={
            "x1": numpy.array([-1.0, 2.0]),
            "x2": numpy.array([-1.0, 2.0]),
            "x3": numpy.array([-1.0, 2.0]),
            "x4": numpy.array([-1.0, 2.0]),
            "x5": numpy.array([-1.0, 2.0])
        })
    _nvar = 5
    cvar = numpy.array([0], dtype=numpy.int32)

    def dfun(self, state, node_coupling, local_coupling=0.0):
        return 0.0 * state


class ModelTestUpdateVariables(Model):
    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=('x1', 'x2', 'x3', 'x4', 'x5'),
        default=('x1', 'x2', 'x3', 'x4', 'x5'),
        doc="""default state variables to be monitored""")

    state_variables = ['x1', 'x2', 'x3', 'x4', 'x5']

    non_integrated_variables = ['x4', 'x5']

    state_variable_range = Final(
        default={
            "x1": numpy.array([-1.0, 2.0]),
            "x2": numpy.array([-1.0, 2.0]),
            "x3": numpy.array([-1.0, 2.0]),
            "x4": numpy.array([-1.0, 2.0]),
            "x5": numpy.array([-1.0, 2.0])
        })
    _nvar = 5
    cvar = numpy.array([0], dtype=numpy.int32)

    def dfun(self, integrated_variables, node_coupling, local_coupling=0.0):
        return 0.0 * integrated_variables

    def update_state_variables_before_integration(self, state, coupling, local_coupling=0.0, stimulus=0.0):
        new_state = numpy.copy(state)
        new_state[3] = state[3] + state[0]
        new_state[4] = state[4] + state[1] + state[2]
        return state

    def update_state_variables_after_integration(self, state):
        new_state = numpy.copy(state)
        new_state[3] = state[3] - state[0]
        new_state[4] = state[4] - state[1] - state[2]
        return state


class ModelTestUpdateVariablesBounds(ModelTestUpdateVariables, ModelTestBounds):
    pass


class TestModels(BaseTestCase):
    """
    Define test cases for models:
        - initialise each class
        - check that initial conditions are always in range

    """

    @staticmethod
    def _validate_initialization(model, expected_sv, expected_modes=1):

        model.configure()
        dt = 2 ** -4
        history_shape = (1, model._nvar, 1, model.number_of_modes)
        model_ic = model.initial(dt, history_shape)
        assert expected_sv == model._nvar
        assert expected_modes == model.number_of_modes

        svr = model.state_variable_range
        sv = model.state_variables
        for i, (lo, hi) in enumerate([svr[sv[i]] for i in range(model._nvar)]):
            for val in model_ic[:, i, :].flatten():
                assert (lo < val < hi)

        state = numpy.zeros((expected_sv, 10, model.number_of_modes))
        obser = model.observe(state)
        assert (len(model.variables_of_interest), 10, model.number_of_modes) == obser.shape
        return state, obser

    def test_sv_boundaries_setup(self):
        model = ModelTestBounds()
        model.configure()
        min_float = numpy.finfo("double").min
        max_float = numpy.finfo("double").max
        min_positive = 1.0 / numpy.finfo("single").max
        state_variable_boundaries = \
            {"x1": numpy.array([0.0, 1.0]),
             "x2": numpy.array([min_float, 1.0]),
             "x3": numpy.array([0.0, max_float]),
             "x4": numpy.array([min_float, max_float])}
        for sv, sv_bounds in state_variable_boundaries.items():
            assert numpy.allclose(sv_bounds, model.state_variable_boundaries[sv], min_positive)

    def test_stvar_init(self):
        model = ModelTestBounds()
        model.configure()
        numpy.testing.assert_array_equal(model.stvar, model.cvar)

        model = ModelTestBounds()
        model.stvar=numpy.r_[1,3]
        model.configure()
        numpy.testing.assert_array_equal(model.stvar, numpy.r_[1,3])

    def _test_steady_state(self, model, fp):
        "test that model produces a given fixed point"
        if fp.ndim == 1:
            fp = fp.reshape((-1, 1))
        init = numpy.zeros(fp.shape)
        for i, (lo, hi) in enumerate(model.state_variable_range.values()):
            init[i, :] = (hi + lo) / 2.0
        t, y = model.stationary_trajectory(initial_conditions=init, n_step=2000, dt=0.01)
        assert y.shape == (201, fp.shape[0], 1, fp.shape[1])
        y = y[:, :, 0]
        # test we're converging to fixed point
        rfp = numpy.sqrt(numpy.sum(numpy.sum((y[-1] - y)**2,axis=1),axis=1))
        dr = rfp[-20:] / (rfp[-21:-1] + 1e-9)
        assert (dr < 1.0).all()
        # and it should be close to what we expect / fix
        numpy.testing.assert_allclose(y[-1], fp, 1e-6, 1e-3)

    def test_wilson_cowan(self):
        """
        Default parameters are taken from figure 4 of [WC_1972]_, pag. 10
        """
        model = models.WilsonCowan()
        self._validate_initialization(model, 2)
        self._test_steady_state(model, numpy.r_[0.461, 0.243])

    def test_g2d(self):
        """
        Default parameters:

        +---------------------------+
        |  SanzLeonetAl  2013       |
        +--------------+------------+
        |Parameter     |  Value     |
        +==============+============+
        | a            |    - 0.5   |
        +--------------+------------+
        | b            |    -10.0   |
        +--------------+------------+
        | c            |      0.0   |
        +--------------+------------+
        | d            |      0.02  |
        +--------------+------------+
        | I            |      0.0   |
        +--------------+------------+
        |* excitable regime if      |
        |* intrinsic frequency is   |
        |  approx 10 Hz             |
        +---------------------------+


        """
        model = models.Generic2dOscillator()
        state, obser = self._validate_initialization(model, 2)
        numpy.testing.assert_allclose(obser[0], state[0])
        self._test_steady_state(model, numpy.r_[ 1.048535, -4.507975])

    def test_g2d_voi(self):
        model = models.Generic2dOscillator(
            variables_of_interest=('W', 'V - W')
        )
        (V, W), (voi_W, voi_WmV) = self._validate_initialization(model, 2)
        numpy.testing.assert_allclose(voi_W, W)
        numpy.testing.assert_allclose(voi_WmV, W - V)

    def test_jansen_rit(self):
        """
        """
        model = models.JansenRit()
        self._validate_initialization(model, 6)
        self._test_steady_state(model,
                numpy.r_[1.105518e-02, 4.790847e+00, 8.826628e-01, 
                         1.308187e-03, 2.367387e-01, 6.615113e-02])

    def test_sj2d(self):
        """
        """
        model = models.ReducedSetFitzHughNagumo()
        self._validate_initialization(model, 4, 3)
        fp = numpy.array([
            [-1.228146,  0.677939,  1.541479],
            [-0.953419,  0.384293,  0.907564],
            [-1.346946, -0.351683,  1.762915],
            [-1.03536 , -0.293471,  0.536003]])
        self._test_steady_state(model, fp)

    def test_sj3d(self):
        """
        """
        model = models.ReducedSetHindmarshRose()
        self._validate_initialization(model, 6, 3)
        fp = numpy.array([
            [-2.154452, -1.387575, -2.018075],
            [-24.581927, -18.522624, -21.449834],
            [5.094243, 5.094284, 5.161934],
            [-2.145466, -1.405172, -2.000391],
            [-24.353387, -19.002245, -21.028078],
            [5.119597, 5.096642, 5.199867],
            ])
        self._test_steady_state(model, fp)

    def test_reduced_wong_wang(self):
        """
        """
        model = models.ReducedWongWang()
        self._validate_initialization(model, 1)
        self._test_steady_state(model, numpy.r_[0.452846])

    def test_deco_balanced_exc_inh(self):
        """
        """
        model = models.DecoBalancedExcInh()
        self._validate_initialization(model, 2)
        self._test_steady_state(model, numpy.r_[0.416673, 0.078865])

    def test_zetterberg_jansen(self):
        """
        """
        model = models.ZetterbergJansen()
        self._validate_initialization(model, 12)

    def test_epileptor(self):
        """
        """
        model = models.Epileptor()
        self._validate_initialization(model, 6)

    def test_hopfield(self):
        """
        """
        model = models.Hopfield()
        self._validate_initialization(model, 2)

    def test_kuramoto(self):
        """
        """
        model = models.Kuramoto()
        self._validate_initialization(model, 1)

    def test_larter(self):
        """
        """
        model = models.LarterBreakspear()
        self._validate_initialization(model, 3)

    def test_linear(self):
        model = models.Linear()
        self._validate_initialization(model, 1)

    def test_ww(self):
        model = models.ReducedWongWang()
        self._validate_initialization(model, 1)

        model = models.ReducedWongWangExcInh()
        self._validate_initialization(model, 2)

    def test_infinite_theta(self):
        model = models.MontbrioPazoRoxin()
        self._validate_initialization(model, 2)

        model = models.CoombesByrne()
        self._validate_initialization(model, 4)
        
        model = models.CoombesByrne2D()
        self._validate_initialization(model, 2)
        
        model = models.GastSchmidtKnosche_SD()
        self._validate_initialization(model, 4)

        model = models.GastSchmidtKnosche_SF()
        self._validate_initialization(model, 4)

        model = models.DumontGutkin()
        self._validate_initialization(model, 8)
