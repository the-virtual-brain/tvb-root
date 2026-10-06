import time
import numpy as np
from tvb.simulator.models.stefanescu_jirsa import (
    ReducedSetFitzHughNagumo,
    ReducedSetHindmarshRose,
)
from tvb.simulator.models.wong_wang_exc_inh import ReducedWongWangExcInh
from tvb.simulator import integrators

def old_fhn_dfun(state, coupling, model):
    xi = state[0, :]
    eta = state[1, :]
    alpha = state[2, :]
    beta = state[3, :]
    c_0 = coupling[0, :].sum(axis=1)[:, np.newaxis]

    dot_xi_A = xi @ model.Aik
    dot_alpha_B = alpha @ model.Bik
    dot_xi_C = xi @ model.Cik

    deriv = np.empty_like(state)
    deriv[0] = (
        model.tau * (xi - model.e_i * (xi**3) / 3.0 - eta)
        + model.K11 * (dot_xi_A - xi)
        - model.K12 * (dot_alpha_B - xi)
        + model.tau * (model.IE_i + c_0)
    )
    deriv[1] = (xi - model.b * eta + model.m_i) / model.tau
    deriv[2] = (
        model.tau * (alpha - model.f_i * (alpha**3) / 3.0 - beta)
        + model.K21 * (dot_xi_C - alpha)
        + model.tau * (model.II_i + c_0)
    )
    deriv[3] = (alpha - model.b * beta + model.n_i) / model.tau
    return deriv

def old_hr_dfun(state, coupling, model):
    xi = state[0, :]
    eta = state[1, :]
    tau = state[2, :]
    alpha = state[3, :]
    beta = state[4, :]
    gamma = state[5, :]
    c_0 = coupling[0, :].sum(axis=1)[:, np.newaxis]

    dot_xi_A = xi @ model.A_ik
    dot_alpha_B = alpha @ model.B_ik
    dot_xi_C = xi @ model.C_ik

    xi_sq = xi * xi
    alpha_sq = alpha * alpha

    deriv = np.empty_like(state)
    deriv[0] = (
        eta - model.a_i * (xi_sq * xi) + model.b_i * xi_sq - tau
        + model.K11 * (dot_xi_A - xi)
        - model.K12 * (dot_alpha_B - xi)
        + model.IE_i + c_0
    )
    deriv[1] = model.c_i - model.d_i * xi_sq - eta
    deriv[2] = model.r * model.s * xi - model.r * tau - model.m_i
    deriv[3] = (
        beta - model.e_i * (alpha_sq * alpha) + model.f_i * alpha_sq - gamma
        + model.K21 * (dot_xi_C - alpha)
        + model.II_i + c_0
    )
    deriv[4] = model.h_i - model.p_i * alpha_sq - beta
    deriv[5] = model.r * model.s * alpha - model.r * gamma - model.n_i
    return deriv

def main():
    nodes_list = [68, 100, 200, 360]
    configs = [
        ("FHN", ReducedSetFitzHughNagumo, 4, 3, lambda m: (lambda s, c, *a: old_fhn_dfun(s, c, m))),
        ("HR",  ReducedSetHindmarshRose,    6, 3, lambda m: (lambda s, c, *a: old_hr_dfun(s, c, m))),
        ("WW",  ReducedWongWangExcInh,      2, 1, lambda m: m._numpy_dfun),
    ]

    print(f"{'Model':5} | {'Nodes':5} | {'Old dfun':>11} | {'New dfun':>11} | {'Old kHz':>8} | {'New kHz':>8} | {'dfun Speedup':>12} | {'Old Sim (ms)':>12} | {'New Sim (ms)':>12} | {'Sim Speedup':>11}")
    print("-" * 115)

    # Use dt=0.01 for stiff numerical stability over 100 steps
    integ = integrators.HeunDeterministic(dt=0.01)

    for name, model_cls, n_vars, n_modes, old_dfun_factory in configs:
        model = model_cls()
        model.configure()
        old_dfun = old_dfun_factory(model)
        new_dfun = model.dfun

        for n in nodes_list:
            np.random.seed(42)
            # Confine initial states within typical phase-space ranges to avoid polynomial divergence
            if name == "WW":
                state = np.random.uniform(0.1, 0.4, size=(n_vars, n, n_modes)).astype(np.float64)
            else:
                state = np.random.uniform(-0.5, 0.5, size=(n_vars, n, n_modes)).astype(np.float64)

            coupling = np.zeros((1, n, n_modes), dtype=np.float64)

            # Warmup
            for _ in range(10):
                old_dfun(state, coupling, 0.0)
                new_dfun(state, coupling, 0.0)

            # Isolated dfun timing: 5 trials of 200 calls
            repeats = 200
            n_trials = 5

            old_dfun_times = []
            new_dfun_times = []
            for _ in range(n_trials):
                t0 = time.perf_counter()
                for _ in range(repeats):
                    old_dfun(state, coupling, 0.0)
                old_dfun_times.append((time.perf_counter() - t0) / repeats)

                t0 = time.perf_counter()
                for _ in range(repeats):
                    new_dfun(state, coupling, 0.0)
                new_dfun_times.append((time.perf_counter() - t0) / repeats)

            t_old_us = np.mean(old_dfun_times) * 1e6
            t_new_us = np.mean(new_dfun_times) * 1e6
            khz_old = (1.0 / np.mean(old_dfun_times)) / 1e3
            khz_new = (1.0 / np.mean(new_dfun_times)) / 1e3
            dfun_speedup = t_old_us / t_new_us

            # Full simulation timing: 5 trials of 100 Heun steps
            old_sim_times = []
            new_sim_times = []
            for _ in range(n_trials):
                s_cur = state.copy()
                t0 = time.perf_counter()
                for _ in range(100):
                    s_cur = integ.scheme(s_cur, old_dfun, coupling, 0.0, 0.0)
                old_sim_times.append(time.perf_counter() - t0)

                s_cur = state.copy()
                t0 = time.perf_counter()
                for _ in range(100):
                    s_cur = integ.scheme(s_cur, new_dfun, coupling, 0.0, 0.0)
                new_sim_times.append(time.perf_counter() - t0)

            sim_old_ms = np.mean(old_sim_times) * 1e3
            sim_new_ms = np.mean(new_sim_times) * 1e3
            sim_speedup = sim_old_ms / sim_new_ms

            print(f"{name:5} | {n:5} | {t_old_us:8.2f} us | {t_new_us:8.2f} us | {khz_old:7.1f}  | {khz_new:7.1f}  | {dfun_speedup:11.2f}x | {sim_old_ms:11.2f}  | {sim_new_ms:11.2f}  | {sim_speedup:10.2f}x")

if __name__ == "__main__":
    main()