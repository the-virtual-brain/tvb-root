## -*- coding: utf-8 -*-
##
## nb-zerlaut-dfun.py.mako
##
## Custom dfun generation for ZerlautAdaptationFirstOrder and
## ZerlautAdaptationSecondOrder models.  Included by nb-hybrid-sim.py.mako
## when a Zerlaut model is detected on a subnetwork.
##
## The transfer function pipeline (get_fluct_regime_vars → threshold_func →
## erfc → estimate_firing_rate → TF) is emitted as composed @njit helper
## functions with all model parameters baked in as constants.
##
## For ZerlautAdaptationSecondOrder, numerical-derivative helpers (finite
## differences of TF) are also emitted for the covariance dynamics.
##
<%page args="sn, debug_nojit, svars, cterms, svars_str, cterms_str, n_svars, dt_val, n_nodes, n_modes, int_type, lo_map, hi_map, i1svars_str"/>
## Expected template-level variables (set by the including template):
##   sn          — SubnetworkInfo for the current subnetwork
##   debug_nojit — bool
##   _zerlaut_params — dict of {param_name: float_value} (extracted by caller)
##   _is_second_order — bool
##   svars, cterms, svars_str, cterms_str, n_svars — standard dfun vars
##   dt_val, n_nodes, n_modes, int_type — integration params
##   lo_map, hi_map — boundary maps
##   i1svars_str — for Heun predictor step

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

    # Population fractions
    _N_e_expr = f"nb.float32({_zp['N_tot']} * (1.0 - {_zp['g']}))"
    _N_i_expr = f"nb.float32({_zp['N_tot']} * {_zp['g']})"
%>

## ------------------------------------------------------------------
## Baked scalar parameters
## ------------------------------------------------------------------
% for _pn, _pv in _zp.items():
_zp_${sn.name}_${_pn} = nb.float32(${_pv})
% endfor

## Polynomial coefficients for transfer function
% for _k in range(10):
_zp_${sn.name}_P_e_${_k} = nb.float32(${_P_e[_k]})
% endfor
% for _k in range(10):
_zp_${sn.name}_P_i_${_k} = nb.float32(${_P_i[_k]})
% endfor

## Population sizes (second-order only)
% if _is_second_order:
_zp_${sn.name}_N_e = ${_N_e_expr}
_zp_${sn.name}_N_i = ${_N_i_expr}
% endif

## ------------------------------------------------------------------
## Helper: get_fluct_regime_vars (scalar, all params baked)
## ------------------------------------------------------------------
${'' if debug_nojit else '@nb.njit(inline="always", cache=True)'}
def _zerlaut_fluct_${sn.name}(Fe, Fi, Fe_ext, Fi_ext, W, E_L):
    g_L = _zp_${sn.name}_g_L
    C_m = _zp_${sn.name}_C_m
    Q_e = _zp_${sn.name}_Q_e
    tau_e = _zp_${sn.name}_tau_e
    E_e = _zp_${sn.name}_E_e
    Q_i = _zp_${sn.name}_Q_i
    tau_i = _zp_${sn.name}_tau_i
    E_i = _zp_${sn.name}_E_i
    N_tot = _zp_${sn.name}_N_tot
    p_connect_e = _zp_${sn.name}_p_connect_e
    p_connect_i = _zp_${sn.name}_p_connect_i
    g = _zp_${sn.name}_g
    K_ext_e = _zp_${sn.name}_K_ext_e
    K_ext_i = _zp_${sn.name}_K_ext_i

    fe = (Fe + nb.float32(1.0e-6)) * (nb.float32(1.0) - g) * p_connect_e * N_tot + Fe_ext * K_ext_e
    fi = (Fi + nb.float32(1.0e-6)) * g * p_connect_i * N_tot + Fi_ext * K_ext_i

    mu_Ge = Q_e * tau_e * fe
    mu_Gi = Q_i * tau_i * fi
    mu_G = g_L + mu_Ge + mu_Gi
    T_m = C_m / mu_G

    mu_V = (mu_Ge * E_e + mu_Gi * E_i + g_L * E_L - W) / mu_G
    U_e = Q_e / mu_G * (E_e - mu_V)
    U_i = Q_i / mu_G * (E_i - mu_V)

    sigma_V = math.sqrt(
        fe * (U_e * tau_e) ** 2 / (nb.float32(2.0) * (tau_e + T_m))
        + fi * (U_i * tau_i) ** 2 / (nb.float32(2.0) * (tau_i + T_m)))

    T_V_num = fe * (U_e * tau_e) ** 2 + fi * (U_i * tau_i) ** 2
    T_V_den = (fe * (U_e * tau_e) ** 2 / (tau_e + T_m)
               + fi * (U_i * tau_i) ** 2 / (tau_i + T_m))
    T_V = T_V_num / T_V_den if T_V_den != nb.float32(0.0) else nb.float32(1.0)
    return mu_V, sigma_V, T_V


## ------------------------------------------------------------------
## Helper: threshold_func (order-9 polynomial, scalar)
## ------------------------------------------------------------------
${'' if debug_nojit else '@nb.njit(inline="always", cache=True)'}
def _zerlaut_threshold_${sn.name}(muV, sigmaV, TvN, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9):
    muV0 = nb.float32(-60.0)
    DmuV0 = nb.float32(10.0)
    sV0 = nb.float32(4.0)
    DsV0 = nb.float32(6.0)
    TvN0 = nb.float32(0.5)
    DTvN0 = nb.float32(1.0)
    V = (muV - muV0) / DmuV0
    S = (sigmaV - sV0) / DsV0
    T = (TvN - TvN0) / DTvN0
    return (P0 + P1 * V + P2 * S + P3 * T + P4 * V * V
            + P5 * S * S + P6 * T * T + P7 * V * S + P8 * V * T + P9 * S * T)


## ------------------------------------------------------------------
## Helper: estimate_firing_rate (erfc-based, scalar)
## ------------------------------------------------------------------
${'' if debug_nojit else '@nb.njit(inline="always", cache=True)'}
def _zerlaut_firing_rate_${sn.name}(muV, sigmaV, Tv, Vthre):
    return math.erfc((Vthre - muV) / (nb.float32(1.4142135623730951) * sigmaV)) / (nb.float32(2.0) * Tv)


## ------------------------------------------------------------------
## Helper: TF — full transfer function pipeline (scalar)
## E_L and P_* select excitatory vs inhibitory
## ------------------------------------------------------------------
${'' if debug_nojit else '@nb.njit(inline="always", cache=True)'}
def _zerlaut_TF_${sn.name}(fe, fi, fe_ext, fi_ext, W, E_L,
                            P0, P1, P2, P3, P4, P5, P6, P7, P8, P9):
    mu_V, sigma_V, T_V = _zerlaut_fluct_${sn.name}(fe, fi, fe_ext, fi_ext, W, E_L)
    TvN = T_V * _zp_${sn.name}_g_L / _zp_${sn.name}_C_m
    V_thre = _zerlaut_threshold_${sn.name}(mu_V, sigma_V, TvN, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9)
    V_thre = V_thre * nb.float32(1000.0)  # V→mV
    return _zerlaut_firing_rate_${sn.name}(mu_V, sigma_V, T_V, V_thre)


## ------------------------------------------------------------------
## Convenience wrappers: TF_e / TF_i with baked polynomial coefficients
## ------------------------------------------------------------------
${'' if debug_nojit else '@nb.njit(inline="always", cache=True)'}
def _zerlaut_TF_e_${sn.name}(fe, fi, fe_ext, fi_ext, W):
    return _zerlaut_TF_${sn.name}(fe, fi, fe_ext, fi_ext, W,
        _zp_${sn.name}_E_L_e,
        % for _k in range(10):
        _zp_${sn.name}_P_e_${_k}${',' if _k < 9 else ')'}
        % endfor


${'' if debug_nojit else '@nb.njit(inline="always", cache=True)'}
def _zerlaut_TF_i_${sn.name}(fe, fi, fe_ext, fi_ext, W):
    return _zerlaut_TF_${sn.name}(fe, fi, fe_ext, fi_ext, W,
        _zp_${sn.name}_E_L_i,
        % for _k in range(10):
        _zp_${sn.name}_P_i_${_k}${',' if _k < 9 else ')'}
        % endfor


% if _is_second_order:
## ------------------------------------------------------------------
## Numerical derivative helpers for second-order covariance dynamics
## ------------------------------------------------------------------
<%
    _df = "nb.float32(1e-7)"
    _df_scale = "nb.float32(1e3)"  # 1/df in kHz units: 2*df*1e3
%>

${'' if debug_nojit else '@nb.njit(inline="always", cache=True)'}
def _zerlaut_diff_fe_${sn.name}(TF_fn, fe, fi, fe_ext, fi_ext, W):
    df = ${_df}
    return (TF_fn(fe + df, fi, fe_ext, fi_ext, W) - TF_fn(fe - df, fi, fe_ext, fi_ext, W)) / (nb.float32(2.0) * df * ${_df_scale})


${'' if debug_nojit else '@nb.njit(inline="always", cache=True)'}
def _zerlaut_diff_fi_${sn.name}(TF_fn, fe, fi, fe_ext, fi_ext, W):
    df = ${_df}
    return (TF_fn(fe, fi + df, fe_ext, fi_ext, W) - TF_fn(fe, fi - df, fe_ext, fi_ext, W)) / (nb.float32(2.0) * df * ${_df_scale})


## For second derivatives we need exc/inh-specific versions because
## they reference the base TF value (_TF_e or _TF_i) for the central term.

${'' if debug_nojit else '@nb.njit(inline="always", cache=True)'}
def _zerlaut_diff2_fe_fe_${sn.name}(TF_fn, fe, fi, fe_ext, fi_ext, W, TF_base):
    df = ${_df}
    return (TF_fn(fe + df, fi, fe_ext, fi_ext, W)
            - nb.float32(2.0) * TF_base
            + TF_fn(fe - df, fi, fe_ext, fi_ext, W)) / (df * ${_df_scale}) ** 2


${'' if debug_nojit else '@nb.njit(inline="always", cache=True)'}
def _zerlaut_diff2_fi_fi_${sn.name}(TF_fn, fe, fi, fe_ext, fi_ext, W, TF_base):
    df = ${_df}
    return (TF_fn(fe, fi + df, fe_ext, fi_ext, W)
            - nb.float32(2.0) * TF_base
            + TF_fn(fe, fi - df, fe_ext, fi_ext, W)) / (df * ${_df_scale}) ** 2


${'' if debug_nojit else '@nb.njit(inline="always", cache=True)'}
def _zerlaut_diff2_fe_fi_${sn.name}(TF_fn, fe, fi, fe_ext, fi_ext, W):
    df = ${_df}
    dfe_plus = (TF_fn(fe, fi + df, fe_ext, fi_ext, W) - TF_fn(fe, fi - df, fe_ext, fi_ext, W)) / (nb.float32(2.0) * df * ${_df_scale})
    ## Evaluate _diff_fe at fi+df and fi-df — but we approximate via _diff_fi at fe+df and fe-df
    return ((TF_fn(fe + df, fi, fe_ext, fi_ext, W) - TF_fn(fe - df, fi, fe_ext, fi_ext, W)) / (nb.float32(2.0) * df * ${_df_scale})
            - (TF_fn(fe + df, fi, fe_ext, fi_ext, W) - TF_fn(fe - df, fi, fe_ext, fi_ext, W)) / (nb.float32(2.0) * df * ${_df_scale}))


## Actually, the cross-derivatives match the original more faithfully
## using the _diff_fi(fe+df) - _diff_fi(fe-df) pattern:
${'' if debug_nojit else '@nb.njit(inline="always", cache=True)'}
def _zerlaut_diff2_fi_fe_${sn.name}(TF_fn, fe, fi, fe_ext, fi_ext, W):
    df = ${_df}
    dfi_at_fe_plus = (TF_fn(fe + df, fi + df, fe_ext, fi_ext, W) - TF_fn(fe + df, fi - df, fe_ext, fi_ext, W)) / (nb.float32(2.0) * df * ${_df_scale})
    dfi_at_fe_minus = (TF_fn(fe - df, fi + df, fe_ext, fi_ext, W) - TF_fn(fe - df, fi - df, fe_ext, fi_ext, W)) / (nb.float32(2.0) * df * ${_df_scale})
    return (dfi_at_fe_plus - dfi_at_fe_minus) / (nb.float32(2.0) * df * ${_df_scale})


${'' if debug_nojit else '@nb.njit(inline="always", cache=True)'}
def _zerlaut_diff2_fe_fi_correct_${sn.name}(TF_fn, fe, fi, fe_ext, fi_ext, W):
    df = ${_df}
    dfe_at_fi_plus = (TF_fn(fe + df, fi + df, fe_ext, fi_ext, W) - TF_fn(fe - df, fi + df, fe_ext, fi_ext, W)) / (nb.float32(2.0) * df * ${_df_scale})
    dfe_at_fi_minus = (TF_fn(fe + df, fi - df, fe_ext, fi_ext, W) - TF_fn(fe - df, fi - df, fe_ext, fi_ext, W)) / (nb.float32(2.0) * df * ${_df_scale})
    return (dfe_at_fi_plus - dfe_at_fi_minus) / (nb.float32(2.0) * df * ${_df_scale})

% endif

## ------------------------------------------------------------------
## dfun — the actual derivative function
## ------------------------------------------------------------------
${'' if debug_nojit else '@nb.njit(inline="always", cache=True)'}
% if not _is_second_order:
def dfun_${sn.name}(E, I, W_e, W_i, ou_drift, Coupling_Term):
    T = _zp_${sn.name}_T
    weight_noise = _zp_${sn.name}_weight_noise
    ext_ex_ex = _zp_${sn.name}_external_input_ex_ex
    ext_ex_in = _zp_${sn.name}_external_input_ex_in
    ext_in_ex = _zp_${sn.name}_external_input_in_ex
    ext_in_in = _zp_${sn.name}_external_input_in_in
    tau_w_e = _zp_${sn.name}_tau_w_e
    tau_w_i = _zp_${sn.name}_tau_w_i
    b_e = _zp_${sn.name}_b_e
    a_e = _zp_${sn.name}_a_e
    b_i = _zp_${sn.name}_b_i
    a_i = _zp_${sn.name}_a_i
    E_L_e = _zp_${sn.name}_E_L_e
    E_L_i = _zp_${sn.name}_E_L_i
    tau_OU = _zp_${sn.name}_tau_OU

    ## external input (exc)
    Fe_ext = Coupling_Term + weight_noise * ou_drift
    if Fe_ext * _zp_${sn.name}_K_ext_e < nb.float32(0.0):
        Fe_ext = nb.float32(0.0)
    Fi_ext = nb.float32(0.0)

    ## dE/dt
    d_E = (_zerlaut_TF_e_${sn.name}(E, I, Fe_ext + ext_ex_ex, Fi_ext + ext_ex_in, W_e) - E) / T
    ## dI/dt
    d_I = (_zerlaut_TF_i_${sn.name}(E, I, Fe_ext + ext_in_ex, Fi_ext + ext_in_in, W_i) - I) / T

    ## dW_e/dt — adaptation (excitatory)
    mu_V_e, _, _ = _zerlaut_fluct_${sn.name}(E, I, Fe_ext + ext_ex_ex, Fi_ext + ext_ex_in, W_e, E_L_e)
    d_W_e = -W_e / tau_w_e + b_e * E + a_e * (mu_V_e - E_L_e) / tau_w_e

    ## dW_i/dt — adaptation (inhibitory)
    mu_V_i, _, _ = _zerlaut_fluct_${sn.name}(E, I, Fe_ext + ext_in_ex, Fi_ext + ext_in_in, W_i, E_L_i)
    d_W_i = -W_i / tau_w_i + b_i * I + a_i * (mu_V_i - E_L_i) / tau_w_i

    ## dou_drift/dt
    d_ou_drift = -ou_drift / tau_OU

    return (d_E, d_I, d_W_e, d_W_i, d_ou_drift,)
% else:
def dfun_${sn.name}(E, I, C_ee, C_ei, C_ii, W_e, W_i, ou_drift, Coupling_Term):
    T = _zp_${sn.name}_T
    weight_noise = _zp_${sn.name}_weight_noise
    ext_ex_ex = _zp_${sn.name}_external_input_ex_ex
    ext_ex_in = _zp_${sn.name}_external_input_ex_in
    ext_in_ex = _zp_${sn.name}_external_input_in_ex
    ext_in_in = _zp_${sn.name}_external_input_in_in
    tau_w_e = _zp_${sn.name}_tau_w_e
    tau_w_i = _zp_${sn.name}_tau_w_i
    b_e = _zp_${sn.name}_b_e
    a_e = _zp_${sn.name}_a_e
    b_i = _zp_${sn.name}_b_i
    a_i = _zp_${sn.name}_a_i
    E_L_e = _zp_${sn.name}_E_L_e
    E_L_i = _zp_${sn.name}_E_L_i
    tau_OU = _zp_${sn.name}_tau_OU
    S_i = _zp_${sn.name}_S_i
    N_e = _zp_${sn.name}_N_e
    N_i = _zp_${sn.name}_N_i

    ## external input — exc and inh populations get different coupling
    E_input_excitatory = Coupling_Term + ext_ex_ex + weight_noise * ou_drift
    if E_input_excitatory < nb.float32(0.0):
        E_input_excitatory = nb.float32(0.0)

    E_input_inhibitory = S_i * Coupling_Term + ext_in_ex + weight_noise * ou_drift
    if E_input_inhibitory < nb.float32(0.0):
        E_input_inhibitory = nb.float32(0.0)

    I_input_excitatory = ext_ex_in
    I_input_inhibitory = ext_in_in

    ## Transfer function values
    _TF_e = _zerlaut_TF_e_${sn.name}(E, I, E_input_excitatory, I_input_excitatory, W_e)
    _TF_i = _zerlaut_TF_i_${sn.name}(E, I, E_input_inhibitory, I_input_inhibitory, W_i)

    ## First derivatives
    _dfe_TF_e = _zerlaut_diff_fe_${sn.name}(_zerlaut_TF_e_${sn.name}, E, I, E_input_excitatory, I_input_excitatory, W_e)
    _dfe_TF_i = _zerlaut_diff_fe_${sn.name}(_zerlaut_TF_i_${sn.name}, E, I, E_input_inhibitory, I_input_inhibitory, W_i)
    _dfi_TF_e = _zerlaut_diff_fi_${sn.name}(_zerlaut_TF_e_${sn.name}, E, I, E_input_excitatory, I_input_excitatory, W_e)
    _dfi_TF_i = _zerlaut_diff_fi_${sn.name}(_zerlaut_TF_i_${sn.name}, E, I, E_input_inhibitory, I_input_inhibitory, W_i)

    ## Second derivatives
    _d2fefe_e = _zerlaut_diff2_fe_fe_${sn.name}(_zerlaut_TF_e_${sn.name}, E, I, E_input_excitatory, I_input_excitatory, W_e, _TF_e)
    _d2fefe_i = _zerlaut_diff2_fe_fe_${sn.name}(_zerlaut_TF_i_${sn.name}, E, I, E_input_inhibitory, I_input_inhibitory, W_i, _TF_i)
    _d2fifi_e = _zerlaut_diff2_fi_fi_${sn.name}(_zerlaut_TF_e_${sn.name}, E, I, E_input_excitatory, I_input_excitatory, W_e, _TF_e)
    _d2fifi_i = _zerlaut_diff2_fi_fi_${sn.name}(_zerlaut_TF_i_${sn.name}, E, I, E_input_inhibitory, I_input_inhibitory, W_i, _TF_i)
    _d2fefi_e = _zerlaut_diff2_fe_fi_correct_${sn.name}(_zerlaut_TF_e_${sn.name}, E, I, E_input_excitatory, I_input_excitatory, W_e)
    _d2fife_e = _zerlaut_diff2_fi_fe_${sn.name}(_zerlaut_TF_e_${sn.name}, E, I, E_input_excitatory, I_input_excitatory, W_e)
    _d2fefi_i = _zerlaut_diff2_fe_fi_correct_${sn.name}(_zerlaut_TF_i_${sn.name}, E, I, E_input_inhibitory, I_input_inhibitory, W_i)
    _d2fife_i = _zerlaut_diff2_fi_fe_${sn.name}(_zerlaut_TF_i_${sn.name}, E, I, E_input_inhibitory, I_input_inhibitory, W_i)

    ## dE/dt — excitatory rate with second-order corrections
    d_E = (_TF_e - E
           + nb.float32(0.5) * C_ee * _d2fefe_e
           + nb.float32(0.5) * C_ei * _d2fefi_e
           + nb.float32(0.5) * C_ei * _d2fife_e
           + nb.float32(0.5) * C_ii * _d2fifi_e
           ) / T

    ## dI/dt — inhibitory rate with second-order corrections
    d_I = (_TF_i - I
           + nb.float32(0.5) * C_ee * _d2fefe_i
           + nb.float32(0.5) * C_ei * _d2fefi_i
           + nb.float32(0.5) * C_ei * _d2fife_i
           + nb.float32(0.5) * C_ii * _d2fifi_i
           ) / T

    ## dC_ee/dt
    d_C_ee = (_TF_e * (nb.float32(1.0) / T - _TF_e) / N_e
              + (_TF_e - E) ** 2
              + nb.float32(2.0) * C_ee * _dfe_TF_e
              + nb.float32(2.0) * C_ei * _dfi_TF_e
              - nb.float32(2.0) * C_ee
              ) / T

    ## dC_ei/dt
    d_C_ei = ((_TF_e - E) * (_TF_i - I)
              + C_ee * _dfe_TF_e
              + C_ei * _dfe_TF_i
              + C_ei * _dfi_TF_e
              + C_ii * _dfi_TF_i
              - nb.float32(2.0) * C_ei
              ) / T

    ## dC_ii/dt
    d_C_ii = (_TF_i * (nb.float32(1.0) / T - _TF_i) / N_i
              + (_TF_i - I) ** 2
              + nb.float32(2.0) * C_ii * _dfi_TF_i
              + nb.float32(2.0) * C_ei * _dfe_TF_i
              - nb.float32(2.0) * C_ii
              ) / T

    ## dW_e/dt — adaptation (excitatory)
    mu_V_e, _, _ = _zerlaut_fluct_${sn.name}(E, I, E_input_excitatory, I_input_excitatory, W_e, E_L_e)
    d_W_e = -W_e / tau_w_e + b_e * E + a_e * (mu_V_e - E_L_e) / tau_w_e

    ## dW_i/dt — adaptation (inhibitory)
    mu_V_i, _, _ = _zerlaut_fluct_${sn.name}(E, I, E_input_inhibitory, I_input_inhibitory, W_i, E_L_i)
    d_W_i = -W_i / tau_w_i + b_i * I + a_i * (mu_V_i - E_L_i) / tau_w_i

    ## dou_drift/dt
    d_ou_drift = -ou_drift / tau_OU

    return (d_E, d_I, d_C_ee, d_C_ei, d_C_ii, d_W_e, d_W_i, d_ou_drift,)
% endif
