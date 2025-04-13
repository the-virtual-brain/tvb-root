import numpy as np
import time
from tvb.tests.library.base_testcase import BaseTestCase
from tvb.simulator.models.stefanescu_jirsa import ReducedSetFitzHughNagumo
from tvb.simulator.models.stefanescu_jirsa import ReducedSetHindmarshRose
from tvb.simulator.models.base import Model


class TestBenchmarkModels(BaseTestCase):
    """Test cases that benchmark the performance of models' implementation
    """

    def randn_state_for_model(self, model: Model, n_node):
        shape = (model.nvar, n_node, model.number_of_modes)
        state = np.random.randn(*shape)
        return state


    def randn_coupling_for_model(self, model: Model, n_node):
        n_cvar = len(model.cvar)
        shape = (n_cvar, n_node, model.number_of_modes)
        coupling = np.random.randn(*shape)
        return coupling

    def eps_for_model(self, model: Model, n_node, time_limit=0.5, state=None, coupling=None):
        model.configure()
        if state is None:
            state = self.randn_state_for_model(model, n_node)
        if coupling is None:
            coupling = self.randn_coupling_for_model(model, n_node)
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

    def test_rsfhn_numba(self):
        #  create, & initialize numba & no-numba models
        rs_fhn_model = ReducedSetFitzHughNagumo()
        n_node = 10
        state = self.randn_state_for_model(rs_fhn_model, n_node)
        coupling = self.randn_coupling_for_model(rs_fhn_model, n_node)

        rs_fhn_model.use_numba = False
        no_numba_performance = self.eps_for_model(rs_fhn_model, 10, state=state, coupling=coupling)
        no_numba_dfun_derivative = rs_fhn_model.dfun(state, coupling)
        rs_fhn_model.use_numba = True
        numba_performance = self.eps_for_model(rs_fhn_model, 10, state=state, coupling=coupling)
        numba_dfun_derivative = rs_fhn_model.dfun(state, coupling)
        speedup = numba_performance / no_numba_performance
        print(f"speedup: {speedup}")
        print(f"no_numba_performance: {no_numba_performance}, numba_performance: {numba_performance}")
        assert speedup > 1
        assert np.allclose(no_numba_dfun_derivative, numba_dfun_derivative)

    def test_rshr_numba(self):
        #  create, & initialize numba & no-numba models
        rs_hr_model = ReducedSetHindmarshRose()
        n_node = 10
        state = self.randn_state_for_model(rs_hr_model, n_node)
        coupling = self.randn_coupling_for_model(rs_hr_model, n_node)

        rs_hr_model.use_numba = False
        no_numba_performance = self.eps_for_model(rs_hr_model, 10, state=state, coupling=coupling)
        no_numba_dfun_derivative = rs_hr_model.dfun(state, coupling)
        rs_hr_model.use_numba = True
        numba_performance = self.eps_for_model(rs_hr_model, 10, state=state, coupling=coupling)
        numba_dfun_derivative = rs_hr_model.dfun(state, coupling)
        speedup = numba_performance / no_numba_performance
        print(f"speedup: {speedup}")
        print(f"no_numba_performance: {no_numba_performance}, numba_performance: {numba_performance}")
        assert speedup > 1
        assert np.allclose(no_numba_dfun_derivative, numba_dfun_derivative)
