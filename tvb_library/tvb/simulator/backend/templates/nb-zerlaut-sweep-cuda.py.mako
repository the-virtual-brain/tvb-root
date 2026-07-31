##
## TheVirtualBrain-Scientific Package. This package holds all simulators, and
## analysers necessary to run brain-simulations. You can use it stand alone or
## in conjunction with TheVirtualBrain-Framework Package.
##
## (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
##
## This program is free software: you can redistribute it and/or modify it under the
## terms of the GNU General Public License as published by the Free Software Foundation,
## either version 3 of the License, or (at your option) any later version.
##
## nb-zerlaut-sweep-cuda.py.mako
##
## CUDA custom dfun generation for ZerlautAdaptationFirstOrder and
## ZerlautAdaptationSecondOrder models.  Included by nb-hybrid-sweep-cuda.py.mako.
##
## WARNING: This template is <%include%>ed from nb-hybrid-sweep-cuda.py.mako.
## The <%page args="..."> MUST include ALL args that the parent template passes,
## even if they are unused by Zerlaut models (is_heun, is_combined, dm_names,
## dm_data, dm_ops). If the parent adds new page args, this template must be
## updated accordingly.
##

<%page args="sn, debug_nojit, svars, cterms, svars_str, cterms_str, n_svars, dt_val, n_nodes, n_modes, int_type, lo_map, hi_map, i1svars_str, is_heun=False, is_combined=False, dm_names=[], dm_data={}, dm_ops=[]"/>

<%
    from tvb.simulator.models.zerlaut import ZerlautAdaptationSecondOrder

    _is_second_order = isinstance(sn.model, ZerlautAdaptationSecondOrder)

    # Extract all parameters as float scalars for baking into generated code.
    _zp = {}
    _zp_names = [
        'g_L', 'E_L_e', 'E_L_i', 'C_m', 'b_e', 'a_e', 'b_i', 'a_i',
        'tau_w_e', 'tau_w_i', 'E_e', 'E_i', 'Q_e', 'Q_i', 'tau_e', 'tau_i',
        'N_tot', 'p_connect_e', 'p_connect_i', 'g', 'K_ext_e', 'K_ext_i',
        'T', 'external_input_ex_ex', 'external_input_ex_in',
        'external_input_in_ex', 'external_input_in_in',
        'tau_OU', 'weight_noise',
    ]
    if _is_second_order:
        _zp_names.append('S_i')
    for _pn in _zp_names:
        _zp[_pn] = float(getattr(sn.model, _pn)[0])

    # Polynomial coefficients — 10 each for exc and inh
    import numpy as _np
    _P_e = [float(x) for x in _np.asarray(sn.model.P_e).ravel()[:10]]
    _P_i = [float(x) for x in _np.asarray(sn.model.P_i).ravel()[:10]]

    _fluct_params = ['g_L', 'C_m', 'Q_e', 'tau_e', 'E_e', 'Q_i', 'tau_i', 'E_i',
                     'N_tot', 'p_connect_e', 'p_connect_i', 'g', 'K_ext_e', 'K_ext_i']
    _dfun_params = ['T', 'weight_noise', 'external_input_ex_ex', 'external_input_ex_in',
                    'external_input_in_ex', 'external_input_in_in',
                    'tau_w_e', 'tau_w_i', 'b_e', 'a_e', 'b_i', 'a_i',
                    'E_L_e', 'E_L_i', 'tau_OU', 'K_ext_e']

    # 1e-7 is near the float32 noise floor for second derivatives;
    # 1e-4 gives a reasonable accuracy/stability trade-off.
    _df = "np.float32(1e-4)"
    _df_scale = "np.float32(1e3)"
    # NOTE: local_coupling is deprecated for hybrid subnetworks; coupling is
    # handled exclusively through the projection/cfun pipeline.
%>

## ------------------------------------------------------------------
## Helper: get_fluct_regime_vars (scalar, all params baked)
## ------------------------------------------------------------------
@cuda.jit(device=True)
def _zerlaut_fluct_${sn.name}(Fe, Fi, Fe_ext, Fi_ext, W, E_L):
% for _pn in _fluct_params:
    ${_pn} = np.float32(${_zp[_pn]})
% endfor
    fe = (Fe + np.float32(1.0e-6)) * (np.float32(1.0) - g) * p_connect_e * N_tot + Fe_ext * K_ext_e
    fi = (Fi + np.float32(1.0e-6)) * g * p_connect_i * N_tot + Fi_ext * K_ext_i

    mu_Ge = Q_e * tau_e * fe
    mu_Gi = Q_i * tau_i * fi
    mu_G = g_L + mu_Ge + mu_Gi
    T_m = C_m / mu_G

    mu_V = (mu_Ge * E_e + mu_Gi * E_i + g_L * E_L - W) / mu_G
    U_e = Q_e / mu_G * (E_e - mu_V)
    U_i = Q_i / mu_G * (E_i - mu_V)

    sigma_V = math.sqrt(
        fe * (U_e * tau_e) ** 2 / (np.float32(2.0) * (tau_e + T_m))
        + fi * (U_i * tau_i) ** 2 / (np.float32(2.0) * (tau_i + T_m)))

    T_V_num = fe * (U_e * tau_e) ** 2 + fi * (U_i * tau_i) ** 2
    T_V_den = (fe * (U_e * tau_e) ** 2 / (tau_e + T_m)
               + fi * (U_i * tau_i) ** 2 / (tau_i + T_m))
    T_V = T_V_num / T_V_den if T_V_den != np.float32(0.0) else np.float32(1.0)
    return mu_V, sigma_V, T_V


## ------------------------------------------------------------------
## Helper: threshold_func (order-9 polynomial, scalar)
## ------------------------------------------------------------------
@cuda.jit(device=True)
def _zerlaut_threshold_${sn.name}(muV, sigmaV, TvN, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9):
    muV0 = np.float32(-60.0)
    DmuV0 = np.float32(10.0)
    sV0 = np.float32(4.0)
    DsV0 = np.float32(6.0)
    TvN0 = np.float32(0.5)
    DTvN0 = np.float32(1.0)
    V = (muV - muV0) / DmuV0
    S = (sigmaV - sV0) / DsV0
    T = (TvN - TvN0) / DTvN0
    return (P0 + P1 * V + P2 * S + P3 * T + P4 * V * V
            + P5 * S * S + P6 * T * T + P7 * V * S + P8 * V * T + P9 * S * T)


## ------------------------------------------------------------------
## Helper: estimate_firing_rate (erfc-based, scalar)
## ------------------------------------------------------------------
@cuda.jit(device=True)
def _zerlaut_firing_rate_${sn.name}(muV, sigmaV, Tv, Vthre):
    return math.erfc((Vthre - muV) / (np.float32(1.4142135623730951) * sigmaV)) / (np.float32(2.0) * Tv)


## ------------------------------------------------------------------
## Helper: TF — full transfer function pipeline (scalar)
## E_L and P_* select excitatory vs inhibitory
## ------------------------------------------------------------------
@cuda.jit(device=True)
def _zerlaut_TF_${sn.name}(fe, fi, fe_ext, fi_ext, W, E_L,
                            P0, P1, P2, P3, P4, P5, P6, P7, P8, P9):
    g_L = np.float32(${_zp['g_L']})
    C_m = np.float32(${_zp['C_m']})
    mu_V, sigma_V, T_V = _zerlaut_fluct_${sn.name}(fe, fi, fe_ext, fi_ext, W, E_L)
    TvN = T_V * g_L / C_m
    V_thre = _zerlaut_threshold_${sn.name}(mu_V, sigma_V, TvN, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9)
    V_thre = V_thre * np.float32(1000.0)  # V→mV
    return _zerlaut_firing_rate_${sn.name}(mu_V, sigma_V, T_V, V_thre)


## ------------------------------------------------------------------
## Convenience wrappers: TF_e / TF_i with baked polynomial coefficients
## ------------------------------------------------------------------
@cuda.jit(device=True)
def _zerlaut_TF_e_${sn.name}(fe, fi, fe_ext, fi_ext, W):
    return _zerlaut_TF_${sn.name}(fe, fi, fe_ext, fi_ext, W,
        np.float32(${_zp['E_L_e']}),
        % for _k in range(10):
        np.float32(${_P_e[_k]})${',' if _k < 9 else ')'}
        % endfor


@cuda.jit(device=True)
def _zerlaut_TF_i_${sn.name}(fe, fi, fe_ext, fi_ext, W):
    return _zerlaut_TF_${sn.name}(fe, fi, fe_ext, fi_ext, W,
        np.float32(${_zp['E_L_i']}),
        % for _k in range(10):
        np.float32(${_P_i[_k]})${',' if _k < 9 else ')'}
        % endfor


% if _is_second_order:
## ------------------------------------------------------------------
## Numerical derivative helpers for second-order covariance dynamics
## ------------------------------------------------------------------

@cuda.jit(device=True)
def _zerlaut_diff_fe_e_${sn.name}(fe, fi, fe_ext, fi_ext, W):
    df = ${_df}
    return (_zerlaut_TF_e_${sn.name}(fe + df, fi, fe_ext, fi_ext, W)
            - _zerlaut_TF_e_${sn.name}(fe - df, fi, fe_ext, fi_ext, W)) / (np.float32(2.0) * df * ${_df_scale})


@cuda.jit(device=True)
def _zerlaut_diff_fe_i_${sn.name}(fe, fi, fe_ext, fi_ext, W):
    df = ${_df}
    return (_zerlaut_TF_i_${sn.name}(fe + df, fi, fe_ext, fi_ext, W)
            - _zerlaut_TF_i_${sn.name}(fe - df, fi, fe_ext, fi_ext, W)) / (np.float32(2.0) * df * ${_df_scale})


@cuda.jit(device=True)
def _zerlaut_diff_fi_e_${sn.name}(fe, fi, fe_ext, fi_ext, W):
    df = ${_df}
    return (_zerlaut_TF_e_${sn.name}(fe, fi + df, fe_ext, fi_ext, W)
            - _zerlaut_TF_e_${sn.name}(fe, fi - df, fe_ext, fi_ext, W)) / (np.float32(2.0) * df * ${_df_scale})


@cuda.jit(device=True)
def _zerlaut_diff_fi_i_${sn.name}(fe, fi, fe_ext, fi_ext, W):
    df = ${_df}
    return (_zerlaut_TF_i_${sn.name}(fe, fi + df, fe_ext, fi_ext, W)
            - _zerlaut_TF_i_${sn.name}(fe, fi - df, fe_ext, fi_ext, W)) / (np.float32(2.0) * df * ${_df_scale})


## Second derivatives
@cuda.jit(device=True)
def _zerlaut_diff2_fe_fe_e_${sn.name}(fe, fi, fe_ext, fi_ext, W):
    df = ${_df}
    TF_base = _zerlaut_TF_e_${sn.name}(fe, fi, fe_ext, fi_ext, W)
    return (_zerlaut_TF_e_${sn.name}(fe + df, fi, fe_ext, fi_ext, W)
            - np.float32(2.0) * TF_base
            + _zerlaut_TF_e_${sn.name}(fe - df, fi, fe_ext, fi_ext, W)) / (df * ${_df_scale}) ** 2


@cuda.jit(device=True)
def _zerlaut_diff2_fe_fe_i_${sn.name}(fe, fi, fe_ext, fi_ext, W):
    df = ${_df}
    TF_base = _zerlaut_TF_i_${sn.name}(fe, fi, fe_ext, fi_ext, W)
    return (_zerlaut_TF_i_${sn.name}(fe + df, fi, fe_ext, fi_ext, W)
            - np.float32(2.0) * TF_base
            + _zerlaut_TF_i_${sn.name}(fe - df, fi, fe_ext, fi_ext, W)) / (df * ${_df_scale}) ** 2


@cuda.jit(device=True)
def _zerlaut_diff2_fi_fi_e_${sn.name}(fe, fi, fe_ext, fi_ext, W):
    df = ${_df}
    TF_base = _zerlaut_TF_e_${sn.name}(fe, fi, fe_ext, fi_ext, W)
    return (_zerlaut_TF_e_${sn.name}(fe, fi + df, fe_ext, fi_ext, W)
            - np.float32(2.0) * TF_base
            + _zerlaut_TF_e_${sn.name}(fe, fi - df, fe_ext, fi_ext, W)) / (df * ${_df_scale}) ** 2


@cuda.jit(device=True)
def _zerlaut_diff2_fi_fi_i_${sn.name}(fe, fi, fe_ext, fi_ext, W):
    df = ${_df}
    TF_base = _zerlaut_TF_i_${sn.name}(fe, fi, fe_ext, fi_ext, W)
    return (_zerlaut_TF_i_${sn.name}(fe, fi + df, fe_ext, fi_ext, W)
            - np.float32(2.0) * TF_base
            + _zerlaut_TF_i_${sn.name}(fe, fi - df, fe_ext, fi_ext, W)) / (df * ${_df_scale}) ** 2


@cuda.jit(device=True)
def _zerlaut_diff2_fi_fe_e_${sn.name}(fe, fi, fe_ext, fi_ext, W):
    df = ${_df}
    dfi_at_fe_plus = (_zerlaut_TF_e_${sn.name}(fe + df, fi + df, fe_ext, fi_ext, W)
                      - _zerlaut_TF_e_${sn.name}(fe + df, fi - df, fe_ext, fi_ext, W)) / (np.float32(2.0) * df * ${_df_scale})
    dfi_at_fe_minus = (_zerlaut_TF_e_${sn.name}(fe - df, fi + df, fe_ext, fi_ext, W)
                       - _zerlaut_TF_e_${sn.name}(fe - df, fi - df, fe_ext, fi_ext, W)) / (np.float32(2.0) * df * ${_df_scale})
    return (dfi_at_fe_plus - dfi_at_fe_minus) / (np.float32(2.0) * df * ${_df_scale})


@cuda.jit(device=True)
def _zerlaut_diff2_fi_fe_i_${sn.name}(fe, fi, fe_ext, fi_ext, W):
    df = ${_df}
    dfi_at_fe_plus = (_zerlaut_TF_i_${sn.name}(fe + df, fi + df, fe_ext, fi_ext, W)
                      - _zerlaut_TF_i_${sn.name}(fe + df, fi - df, fe_ext, fi_ext, W)) / (np.float32(2.0) * df * ${_df_scale})
    dfi_at_fe_minus = (_zerlaut_TF_i_${sn.name}(fe - df, fi + df, fe_ext, fi_ext, W)
                       - _zerlaut_TF_i_${sn.name}(fe - df, fi - df, fe_ext, fi_ext, W)) / (np.float32(2.0) * df * ${_df_scale})
    return (dfi_at_fe_plus - dfi_at_fe_minus) / (np.float32(2.0) * df * ${_df_scale})


@cuda.jit(device=True)
def _zerlaut_diff2_fe_fi_e_${sn.name}(fe, fi, fe_ext, fi_ext, W):
    df = ${_df}
    dfe_at_fi_plus = (_zerlaut_TF_e_${sn.name}(fe + df, fi + df, fe_ext, fi_ext, W)
                      - _zerlaut_TF_e_${sn.name}(fe - df, fi + df, fe_ext, fi_ext, W)) / (np.float32(2.0) * df * ${_df_scale})
    dfe_at_fi_minus = (_zerlaut_TF_e_${sn.name}(fe + df, fi - df, fe_ext, fi_ext, W)
                       - _zerlaut_TF_e_${sn.name}(fe - df, fi - df, fe_ext, fi_ext, W)) / (np.float32(2.0) * df * ${_df_scale})
    return (dfe_at_fi_plus - dfe_at_fi_minus) / (np.float32(2.0) * df * ${_df_scale})


@cuda.jit(device=True)
def _zerlaut_diff2_fe_fi_i_${sn.name}(fe, fi, fe_ext, fi_ext, W):
    df = ${_df}
    dfe_at_fi_plus = (_zerlaut_TF_i_${sn.name}(fe + df, fi + df, fe_ext, fi_ext, W)
                      - _zerlaut_TF_i_${sn.name}(fe - df, fi + df, fe_ext, fi_ext, W)) / (np.float32(2.0) * df * ${_df_scale})
    dfe_at_fi_minus = (_zerlaut_TF_i_${sn.name}(fe + df, fi - df, fe_ext, fi_ext, W)
                       - _zerlaut_TF_i_${sn.name}(fe - df, fi - df, fe_ext, fi_ext, W)) / (np.float32(2.0) * df * ${_df_scale})
    return (dfe_at_fi_plus - dfe_at_fi_minus) / (np.float32(2.0) * df * ${_df_scale})

% endif

## ------------------------------------------------------------------
## dfun — the actual derivative function
## ------------------------------------------------------------------
@cuda.jit(device=True)
def dfun_${sn.name}(${svars_str}, ${cterms_str}, _sp, ni, sweep_params, tid):
% for _pn in _dfun_params:
    ${_pn} = np.float32(${_zp[_pn]})
% endfor
% if _is_second_order:
    S_i = np.float32(${_zp['S_i']})
    N_e = np.float32(${_zp['N_tot']} * (1.0 - ${_zp['g']}))
    N_i = np.float32(${_zp['N_tot']} * ${_zp['g']})
% endif

% if not _is_second_order:
    Fe_ext = Coupling_Term + weight_noise * ou_drift
    if Fe_ext * K_ext_e < np.float32(0.0):
        Fe_ext = np.float32(0.0)
    Fi_ext = np.float32(0.0)

    d_E = (_zerlaut_TF_e_${sn.name}(E, I, Fe_ext + external_input_ex_ex, Fi_ext + external_input_ex_in, W_e) - E) / T
    d_I = (_zerlaut_TF_i_${sn.name}(E, I, Fe_ext + external_input_in_ex, Fi_ext + external_input_in_in, W_i) - I) / T

    mu_V_e, _, _ = _zerlaut_fluct_${sn.name}(E, I, Fe_ext + external_input_ex_ex, Fi_ext + external_input_ex_in, W_e, E_L_e)
    d_W_e = -W_e / tau_w_e + b_e * E + a_e * (mu_V_e - E_L_e) / tau_w_e

    mu_V_i, _, _ = _zerlaut_fluct_${sn.name}(E, I, Fe_ext + external_input_in_ex, Fi_ext + external_input_in_in, W_i, E_L_i)
    d_W_i = -W_i / tau_w_i + b_i * I + a_i * (mu_V_i - E_L_i) / tau_w_i

    d_ou_drift = -ou_drift / tau_OU

    return (${', '.join(['d_' + s for s in svars])},)
% else:
    E_input_excitatory = Coupling_Term + external_input_ex_ex + weight_noise * ou_drift
    if E_input_excitatory < np.float32(0.0):
        E_input_excitatory = np.float32(0.0)

    E_input_inhibitory = S_i * Coupling_Term + external_input_in_ex + weight_noise * ou_drift
    if E_input_inhibitory < np.float32(0.0):
        E_input_inhibitory = np.float32(0.0)

    I_input_excitatory = external_input_ex_in
    I_input_inhibitory = external_input_in_in

    _TF_e = _zerlaut_TF_e_${sn.name}(E, I, E_input_excitatory, I_input_excitatory, W_e)
    _TF_i = _zerlaut_TF_i_${sn.name}(E, I, E_input_inhibitory, I_input_inhibitory, W_i)

    _dfe_TF_e = _zerlaut_diff_fe_e_${sn.name}(E, I, E_input_excitatory, I_input_excitatory, W_e)
    _dfe_TF_i = _zerlaut_diff_fe_i_${sn.name}(E, I, E_input_inhibitory, I_input_inhibitory, W_i)
    _dfi_TF_e = _zerlaut_diff_fi_e_${sn.name}(E, I, E_input_excitatory, I_input_excitatory, W_e)
    _dfi_TF_i = _zerlaut_diff_fi_i_${sn.name}(E, I, E_input_inhibitory, I_input_inhibitory, W_i)

    _d2fefe_e = _zerlaut_diff2_fe_fe_e_${sn.name}(E, I, E_input_excitatory, I_input_excitatory, W_e)
    _d2fefe_i = _zerlaut_diff2_fe_fe_i_${sn.name}(E, I, E_input_inhibitory, I_input_inhibitory, W_i)
    _d2fifi_e = _zerlaut_diff2_fi_fi_e_${sn.name}(E, I, E_input_excitatory, I_input_excitatory, W_e)
    _d2fifi_i = _zerlaut_diff2_fi_fi_i_${sn.name}(E, I, E_input_inhibitory, I_input_inhibitory, W_i)
    _d2fefi_e = _zerlaut_diff2_fe_fi_e_${sn.name}(E, I, E_input_excitatory, I_input_excitatory, W_e)
    _d2fife_e = _zerlaut_diff2_fi_fe_e_${sn.name}(E, I, E_input_excitatory, I_input_excitatory, W_e)
    _d2fefi_i = _zerlaut_diff2_fe_fi_i_${sn.name}(E, I, E_input_inhibitory, I_input_inhibitory, W_i)
    _d2fife_i = _zerlaut_diff2_fi_fe_i_${sn.name}(E, I, E_input_inhibitory, I_input_inhibitory, W_i)

    d_E = (_TF_e - E
           + np.float32(0.5) * C_ee * _d2fefe_e
           + np.float32(0.5) * C_ei * _d2fefi_e
           + np.float32(0.5) * C_ei * _d2fife_e
           + np.float32(0.5) * C_ii * _d2fifi_e
           ) / T

    d_I = (_TF_i - I
           + np.float32(0.5) * C_ee * _d2fefe_i
           + np.float32(0.5) * C_ei * _d2fefi_i
           + np.float32(0.5) * C_ei * _d2fife_i
           + np.float32(0.5) * C_ii * _d2fifi_i
           ) / T

    d_C_ee = (_TF_e * (np.float32(1.0) / T - _TF_e) / N_e
              + (_TF_e - E) ** 2
              + np.float32(2.0) * C_ee * _dfe_TF_e
              + np.float32(2.0) * C_ei * _dfi_TF_e
              - np.float32(2.0) * C_ee
              ) / T

    d_C_ei = ((_TF_e - E) * (_TF_i - I)
              + C_ee * _dfe_TF_e
              + C_ei * _dfe_TF_i
              + C_ei * _dfi_TF_e
              + C_ii * _dfi_TF_i
              - np.float32(2.0) * C_ei
              ) / T

    d_C_ii = (_TF_i * (np.float32(1.0) / T - _TF_i) / N_i
              + (_TF_i - I) ** 2
              + np.float32(2.0) * C_ii * _dfi_TF_i
              + np.float32(2.0) * C_ei * _dfe_TF_i
              - np.float32(2.0) * C_ii
              ) / T

    mu_V_e, _, _ = _zerlaut_fluct_${sn.name}(E, I, E_input_excitatory, I_input_excitatory, W_e, E_L_e)
    d_W_e = -W_e / tau_w_e + b_e * E + a_e * (mu_V_e - E_L_e) / tau_w_e

    mu_V_i, _, _ = _zerlaut_fluct_${sn.name}(E, I, E_input_inhibitory, I_input_inhibitory, W_i, E_L_i)
    d_W_i = -W_i / tau_w_i + b_i * I + a_i * (mu_V_i - E_L_i) / tau_w_i

    d_ou_drift = -ou_drift / tau_OU

    return (${', '.join(['d_' + s for s in svars])},)
% endif
