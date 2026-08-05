# -*- coding: utf-8 -*-
"""Validate that ``state_variable_dfuns`` declarative expressions match the
actual model ``dfun`` output.
"""
import numpy as np
import pytest
from tvb.simulator.models import (
    Generic2dOscillator, Kuramoto, SupHopf,
    Epileptor, EpileptorCodim3, Hopfield,
    WilsonCowan, ReducedWongWang, DecoBalancedExcInh,
    JansenRit, ZetterbergJansen, LarterBreakspear,
)
from tvb.tests.library.simulator.hybrid.test_validation_base import ValidationTestBase


class TestModelCodegenParity(ValidationTestBase):
    """Verify codegen expressions match numpy dfun output."""

    RTOL = 1e-4
    ATOL = 1e-5

    def _check(self, model_cls, init_kwargs=None, **kwargs):
        model = model_cls(**(init_kwargs or {}))
        model.configure()
        n_nodes = model.nvar
        n_modes = model.number_of_modes
        exprs = getattr(model, "state_variable_dfuns", None)
        if exprs is None:
            pytest.skip("no state_variable_dfuns")

        state = np.random.RandomState(42).randn(len(model.state_variables), n_nodes, n_modes)
        coupling = np.random.RandomState(43).randn(len(model.cvar), n_nodes, n_modes)
        ref = model.dfun(state, coupling)

        # Simple eval: build namespace from parameters only
        ns = {"np": np, "math": np, "nb": np.float32}
        ns["Coupling_Term"] = coupling[0] if len(model.cvar) > 0 else np.zeros((n_nodes, n_modes))
        for pname in getattr(model, "parameter_names", []):
            ns[pname] = float(np.asarray(getattr(model, pname)).flat[0])

        codegen = np.zeros_like(ref)
        for i in range(n_nodes):
            for m in range(n_modes):
                ln = dict(ns)
                for j, sv in enumerate(model.state_variables):
                    ln[sv] = float(state[j, i, m])
                for ci in range(len(model.cvar)):
                    ln[f"Coupling_Term_{ci}"] = float(coupling[ci, i, m])
                for sv_name, expr in exprs.items():
                    if sv_name in model.state_variables:
                        idx = list(model.state_variables).index(sv_name)
                        try:
                            expr_clean = expr.replace("nb.float32(", "").replace(")", "")
                            codegen[idx, i, m] = float(eval(expr_clean, {"__builtins__": {}}, ln))
                        except Exception:
                            codegen[idx, i, m] = np.nan

        for idx, sv_name in enumerate(model.state_variables):
            valid = ~np.isnan(codegen[idx])
            if valid.any():
                np.testing.assert_allclose(
                    codegen[idx][valid], ref[idx][valid],
                    rtol=self.RTOL, atol=self.ATOL,
                    err_msg=f"{model_cls.__name__}/{sv_name}",
                )

    def test_generic2d(self):
        self._check(Generic2dOscillator)

    def test_suphopf(self):
        self._check(SupHopf)

    def test_epileptor(self):
        self._check(Epileptor)

    def test_wilsoncowan(self):
        self._check(WilsonCowan)

    def test_reducedwongwang(self):
        self._check(ReducedWongWang)

    def test_jansenrit(self):
        self._check(JansenRit)

    def test_larterbreakspear(self):
        self._check(LarterBreakspear)

    def test_kuramoto(self):
        self._check(Kuramoto)

    def test_epileptorcodim3(self):
        self._check(EpileptorCodim3)

    def test_hopfield(self):
        self._check(Hopfield)

    def test_decobalanced(self):
        self._check(DecoBalancedExcInh)

    def test_zetterbergjansen(self):
        self._check(ZetterbergJansen)
