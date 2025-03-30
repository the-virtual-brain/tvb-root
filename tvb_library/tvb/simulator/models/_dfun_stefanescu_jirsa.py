import numba as nb
import numpy as np


def dfun_fitzhughnaguma(state_variables, coupling, local_coupling, tau, e_i, K11, Aik, K12, Bik, IE_i, 
               b, m_i, K21, Cik, II_i, f_i, n_i):
    xi = state_variables[0, :]
    eta = state_variables[1, :]
    alpha = state_variables[2, :]
    beta = state_variables[3, :]
    derivative = np.empty_like(state_variables)

    # Sum the activity from the modes
    c_0 = coupling[0, :].sum(axis=1)[:, np.newaxis]

    # Compute derivatives
    derivative[0] = (
        tau * (xi - e_i * xi ** 3 / 3.0 - eta)
        + K11 * (np.dot(xi, Aik) - xi)
        - K12 * (np.dot(alpha, Bik) - xi)
        + tau * (IE_i + c_0 + local_coupling * xi)
    )

    derivative[1] = (xi - b * eta + m_i) / tau

    derivative[2] = (
        tau * (alpha - f_i * alpha ** 3 / 3.0 - beta)
        + K21 * (np.dot(xi, Cik) - alpha)
        + tau * (II_i + c_0 + local_coupling * xi)
    )

    derivative[3] = (alpha - b * beta + n_i) / tau

    return derivative

_dfun_fitzhughnaguma_numba = nb.njit(dfun_fitzhughnaguma)
def dfun_fitzhughnaguma_numba(state_variables, coupling, local_coupling, tau, e_i, K11, Aik, K12, Bik, IE_i, 
               b, m_i, K21, Cik, II_i, f_i, n_i):
    return _dfun_fitzhughnaguma_numba(state_variables, coupling, local_coupling,
                                      tau, e_i, K11, Aik, K12, Bik, IE_i, b,
                                      m_i, K21, Cik, II_i, f_i, n_i)

def dfun_hindmarshrose(state_variables, coupling, local_coupling, r, a_i, b_i, c_i, d_i, s, K11, A_ik, K12, B_ik, IE_i, m_i, K21, C_ik, II_i, e_i, f_i, h_i, p_i, n_i):
    xi = state_variables[0, :]
    eta = state_variables[1, :]
    tau = state_variables[2, :]
    alpha = state_variables[3, :]
    beta = state_variables[4, :]
    gamma = state_variables[5, :]
    derivative = np.empty_like(state_variables)

    c_0 = coupling[0, :].sum(axis=1)[:, np.newaxis]
    # c_1 = coupling[1, :]

    derivative[0] = (eta - a_i * xi ** 3 + b_i * xi ** 2 - tau +
            K11 * (np.dot(xi, A_ik) - xi) -
            K12 * (np.dot(alpha, B_ik) - xi) +
            IE_i + c_0 + local_coupling * xi)

    derivative[1] = c_i - d_i * xi ** 2 - eta

    derivative[2] = r * s * xi - r * tau - m_i

    derivative[3] = (beta - e_i * alpha ** 3 + f_i * alpha ** 2 - gamma +
                K21 * (np.dot(xi, C_ik) - alpha) +
                II_i + c_0 + local_coupling * xi)

    derivative[4] = h_i - p_i * alpha ** 2 - beta

    derivative[5] = r * s * alpha - r * gamma - n_i

    return derivative

_dfun_hindmarshrose_numba = nb.njit(dfun_hindmarshrose)
@nb.njit
def dfun_hindmarshrose_numba(state_variables, coupling, local_coupling, r, a_i, b_i, c_i, d_i, s, K11, A_ik, K12, B_ik, IE_i, m_i, K21, C_ik, II_i, e_i, f_i, h_i, p_i, n_i):
    return _dfun_hindmarshrose_numba(state_variables, coupling, local_coupling,
                                      r, a_i, b_i, c_i, d_i, s, K11, A_ik,
                                      K12, B_ik, IE_i, m_i, K21, C_ik, II_i,
                                      e_i, f_i, h_i, p_i, n_i)


