## -*- coding: utf-8 -*-
##
## nb-cerebellar-dfun.py.mako
##
## Custom dfun generation for the CerebellarMF model.  Included by
## nb-hybrid-sim.py.mako when a CerebellarMF model is detected on a
## subnetwork.
##
## The transfer-function pipeline (get_fluct_regime_vars → threshold_func →
## estimate_firing_rate → TF) is emitted as composed @njit helper functions
## with scalar and per-node model parameters read from the runtime parameter matrix.
##
<%page args="sn, debug_nojit, svars, cterms, svars_str, cterms_str, n_svars, dt_val, n_nodes, n_modes, int_type, lo_map, hi_map, i1svars_str"/>
## Expected template-level variables (set by the including template):
##   sn          — SubnetworkInfo for the current subnetwork
##   debug_nojit — bool
##   svars, cterms, svars_str, cterms_str, n_svars — standard dfun vars
##   dt_val, n_nodes, n_modes, int_type — integration params
##   lo_map, hi_map — boundary maps
##   i1svars_str — for Heun predictor step

<%
    _cp_names = list(sn.model._nb_hybrid_runtime_parameter_names)
    _cp_indices = {name: index for index, name in enumerate(_cp_names)}

    # Legacy mode flags (baked as compile-time conditionals)
    _use_legacy_goc_e_e = bool(getattr(sn.model, 'use_legacy_goc_e_e')[0])
    _add_noise_mli_pc = bool(getattr(sn.model, 'add_noise_mli_pc')[0])

    # Polynomial coefficients — 5 each for the 4 populations
    import numpy as _np
    _P_grc = [float(x) for x in _np.asarray(sn.model.P_grc).ravel()[:5]]
    _P_goc = [float(x) for x in _np.asarray(sn.model.P_goc).ravel()[:5]]
    _P_mli = [float(x) for x in _np.asarray(sn.model.P_mli).ravel()[:5]]
    _P_pc  = [float(x) for x in _np.asarray(sn.model.P_pc).ravel()[:5]]
%>

## Legacy mode flags (compile-time)
_USE_LEGACY_GOC_EE_${sn.name} = ${_use_legacy_goc_e_e}
_ADD_NOISE_MLI_PC_${sn.name} = ${_add_noise_mli_pc}

## Polynomial coefficients for threshold function
% for _k in range(5):
_cp_${sn.name}_P_grc_${_k} = nb.float32(${_P_grc[_k]})
% endfor
% for _k in range(5):
_cp_${sn.name}_P_goc_${_k} = nb.float32(${_P_goc[_k]})
% endfor
% for _k in range(5):
_cp_${sn.name}_P_mli_${_k} = nb.float32(${_P_mli[_k]})
% endfor
% for _k in range(5):
_cp_${sn.name}_P_pc_${_k} = nb.float32(${_P_pc[_k]})
% endfor

## ------------------------------------------------------------------
## Helper: get_fluct_regime_vars (2D: GrC, MLI, PC)
## Two synaptic inputs: excitatory (Q_e, tau_e, Ee, Ke) and
## inhibitory (Q_i, tau_i, Ei, Ki).
## Returns (mu_V, sigma_V, T_V, muGn).
## ------------------------------------------------------------------
${'' if debug_nojit else '@nb.njit(inline="always", cache=True)'}
def _crbl_fluct_2d_${sn.name}(Fe, Fi, Fe_ext, Fi_ext, XX, Q_e, tau_e, Ee, Q_i, tau_i, Ei, Gl, Cm, El, Ke, Ki):
    fe = (Fe + nb.float32(1.0e-6)) + Fe_ext
    fi = (Fi + nb.float32(1.0e-6)) + Fi_ext

    mu_Ge = Q_e * tau_e * fe * Ke
    mu_Gi = Q_i * tau_i * fi * Ki
    mu_G = Gl + mu_Ge + mu_Gi

    mu_V = (nb.float32(2.718281828459045) * (mu_Ge * Ee + mu_Gi * Ei + Gl * El) - XX) / mu_G
    muGn = mu_G / Gl
    Tm = Cm / mu_G

    Ue = Q_e / mu_G * (Ee - mu_V)
    Ui = Q_i / mu_G * (Ei - mu_V)

    sVe = (nb.float32(2.0) * Tm + tau_e) * ((nb.float32(2.718281828459045) * Ue * tau_e) / (nb.float32(2.0) * (tau_e + Tm))) ** 2 * Ke * fe
    sVi = (nb.float32(2.0) * Tm + tau_i) * ((nb.float32(2.718281828459045) * Ui * tau_i) / (nb.float32(2.0) * (tau_i + Tm))) ** 2 * Ki * fi
    sigma_V = math.sqrt(sVe + sVi)

    fe = fe + nb.float32(1.0e-9)
    fi = fi + nb.float32(1.0e-9)

    Tv_num = (Ke * fe * Ue ** 2 * tau_e ** 2 * nb.float32(2.718281828459045) ** 2
              + Ki * fi * Ui ** 2 * tau_i ** 2 * nb.float32(2.718281828459045) ** 2)
    Tv_den = (sigma_V + nb.float32(1.0e-20)) ** 2
    Tv = nb.float32(0.5) * Tv_num / Tv_den

    T_V = Tv * Gl / Cm
    return mu_V, sigma_V, T_V, muGn


## ------------------------------------------------------------------
## Helper: get_fluct_regime_vars_goc (3D: GoC)
## Three synaptic inputs: grc→goc excitatory, mossy→goc excitatory,
## goc→goc inhibitory.
## Returns (mu_V, sigma_V, T_V, muGn).
## ------------------------------------------------------------------
${'' if debug_nojit else '@nb.njit(inline="always", cache=True)'}
def _crbl_fluct_3d_${sn.name}(Fe, Fi, Fe_ext, XX, Qe_gr, Te_gr, Ee, Qi, Ti, Ei, Gl, Cm, El, Ke_grc, Ki, Ke_ext, Qe_ext, Te_ext):
    fe_g = Fe + nb.float32(1.0e-6)
    fe_m = Fe_ext
    fi = Fi + nb.float32(1.0e-6)

    muGe_g = Qe_gr * Ke_grc * Te_gr * fe_g
    muGe_m = Qe_ext * Ke_ext * Te_ext * fe_m
    muGi = Qi * Ki * Ti * fi
    mu_G = Gl + muGe_g + muGe_m + muGi

    mu_V = (nb.float32(2.718281828459045) * (muGe_g * Ee + muGe_m * Ee + muGi * Ei + Gl * El) - XX) / mu_G
    muGn = mu_G / Gl
    Tm = Cm / mu_G

    Ue_g = Qe_gr / mu_G * (Ee - mu_V)
    Ue_m = Qe_ext / mu_G * (Ee - mu_V)
    Ui = Qi / mu_G * (Ei - mu_V)

    sVe_g = (nb.float32(2.0) * Tm + Te_gr) * ((nb.float32(2.718281828459045) * Ue_g * Te_gr) / (nb.float32(2.0) * (Te_gr + Tm))) ** 2 * Ke_grc * fe_g
    sVe_m = (nb.float32(2.0) * Tm + Te_ext) * ((nb.float32(2.718281828459045) * Ue_m * Te_ext) / (nb.float32(2.0) * (Te_ext + Tm))) ** 2 * Ke_ext * fe_m
    sVi = (nb.float32(2.0) * Tm + Ti) * ((nb.float32(2.718281828459045) * Ui * Ti) / (nb.float32(2.0) * (Ti + Tm))) ** 2 * Ki * fi
    sigma_V = math.sqrt(sVe_g + sVe_m + sVi)

    fe_m = fe_m + nb.float32(1.0e-15)
    fe_g = fe_g + nb.float32(1.0e-15)
    fi = fi + nb.float32(1.0e-15)

    Tv_num = (Ke_grc * fe_g * Ue_g ** 2 * Te_gr ** 2 * nb.float32(2.718281828459045) ** 2
              + Ke_ext * fe_m * Ue_m ** 2 * Te_ext ** 2 * nb.float32(2.718281828459045) ** 2
              + Ki * fi * Ui ** 2 * Ti ** 2 * nb.float32(2.718281828459045) ** 2)
    Tv_den = (sigma_V + nb.float32(1.0e-20)) ** 2
    Tv = nb.float32(0.5) * Tv_num / Tv_den
    T_V = Tv * Gl / Cm
    return mu_V, sigma_V, T_V, muGn


## ------------------------------------------------------------------
## Helper: threshold_func (4th-order polynomial + log(muGn) term)
## ------------------------------------------------------------------
${'' if debug_nojit else '@nb.njit(inline="always", cache=True)'}
def _crbl_threshold_${sn.name}(muV, sigmaV, TvN, muGn, P0, P1, P2, P3, P4):
    muV0 = nb.float32(-60.0)
    DmuV0 = nb.float32(10.0)
    sV0 = nb.float32(4.0)
    DsV0 = nb.float32(6.0)
    TvN0 = nb.float32(0.5)
    DTvN0 = nb.float32(1.0)
    V = (muV - muV0) / DmuV0
    S = (sigmaV - sV0) / DsV0
    T = (TvN - TvN0) / DTvN0
    return P0 + P1 * V + P2 * S + P3 * T + P4 * math.log(muGn)


## ------------------------------------------------------------------
## Helper: estimate_firing_rate (erfc-based, Escalón et al. 2018 Eq. 10)
## ------------------------------------------------------------------
${'' if debug_nojit else '@nb.njit(inline="always", cache=True)'}
def _crbl_firing_rate_${sn.name}(muV, sigmaV, TvN, Vthre, Gl, Cm, alpha):
    return nb.float32(0.5) / TvN * Gl / Cm * math.erfc((Vthre - muV) / (nb.float32(1.4142135623730951) * sigmaV)) * alpha


## ------------------------------------------------------------------
## Composite TF for 2-input populations (GrC, MLI, PC)
## ------------------------------------------------------------------
${'' if debug_nojit else '@nb.njit(inline="always", cache=True)'}
def _crbl_TF_2d_${sn.name}(Fe, Fi, Fe_ext, Fi_ext, W, Q_e, tau_e, Ee, Q_i, tau_i, Ei, Gl, Cm, El, Ke, Ki, alpha, P0, P1, P2, P3, P4):
    mu_V, sigma_V, T_V, muGn = _crbl_fluct_2d_${sn.name}(Fe, Fi, Fe_ext, Fi_ext, nb.float32(0.0), Q_e, tau_e, Ee, Q_i, tau_i, Ei, Gl, Cm, El, Ke, Ki)
    V_thre = _crbl_threshold_${sn.name}(mu_V, sigma_V, T_V, muGn, P0, P1, P2, P3, P4)
    V_thre = V_thre * nb.float32(1000.0)  # V → mV
    return _crbl_firing_rate_${sn.name}(mu_V, sigma_V, T_V, V_thre, Gl, Cm, alpha)


## ------------------------------------------------------------------
## Composite TF for 3-input GoC population
## ------------------------------------------------------------------
${'' if debug_nojit else '@nb.njit(inline="always", cache=True)'}
def _crbl_TF_3d_${sn.name}(Fe, Fi, Fe_ext, W, Qe_gr, Te_gr, Ee, Qi, Ti, Ei, Gl, Cm, El, Ke_grc, Ki, Ke_ext, Qe_ext, Te_ext, alpha, P0, P1, P2, P3, P4):
    mu_V, sigma_V, T_V, muGn = _crbl_fluct_3d_${sn.name}(Fe, Fi, Fe_ext, nb.float32(0.0), Qe_gr, Te_gr, Ee, Qi, Ti, Ei, Gl, Cm, El, Ke_grc, Ki, Ke_ext, Qe_ext, Te_ext)
    V_thre = _crbl_threshold_${sn.name}(mu_V, sigma_V, T_V, muGn, P0, P1, P2, P3, P4)
    V_thre = V_thre * nb.float32(1000.0)  # V → mV
    return _crbl_firing_rate_${sn.name}(mu_V, sigma_V, T_V, V_thre, Gl, Cm, alpha)


## ------------------------------------------------------------------
## dfun — the actual derivative function
##
## Signature matches the non-combined integrate function call:
##   dfun_${sn.name}(GrC, GoC, MLI, PC, noise, mossy, parallel, _sp, ni)
## ------------------------------------------------------------------
${'' if debug_nojit else '@nb.njit(inline="always", cache=True)'}
def dfun_${sn.name}(GrC, GoC, MLI, PC, noise, mossy, parallel, _sp, ni):
% for _pn, _pi in _cp_indices.items():
    ${_pn} = _sp[${_pi}, ni]
% endfor
    goc_Ee = E_i if _USE_LEGACY_GOC_EE_${sn.name} else E_e

    ## Anatomical routing of coupling signals
    ## frac_mossy / frac_parallel split the coupling into pathways,
    ## then mf_to_* / pf_to_* distribute to each population.
    Fe_ext_tod1 = mossy * frac_mossy * mf_to_grc + weight_noise * noise
    Fe_ext_tod2 = mossy * frac_mossy * mf_to_goc + parallel * frac_parallel * pf_to_goc + weight_noise * noise

    % if _add_noise_mli_pc:
    Fe_ext_tod3 = parallel * frac_parallel * pf_to_mli + weight_noise * noise
    Fe_ext_tod4 = parallel * frac_parallel * pf_to_pc + weight_noise * noise
    % else:
    Fe_ext_tod3 = parallel * frac_parallel * pf_to_mli
    Fe_ext_tod4 = parallel * frac_parallel * pf_to_pc
    % endif

    ## Clamp negative inputs
    if Fe_ext_tod1 * K_mossy_grc < nb.float32(0.0):
        Fe_ext_tod1 = nb.float32(0.0)
    if Fe_ext_tod2 * K_mossy_goc < nb.float32(0.0):
        Fe_ext_tod2 = nb.float32(0.0)
    if Fe_ext_tod3 * K_grc_mli < nb.float32(0.0):
        Fe_ext_tod3 = nb.float32(0.0)
    if Fe_ext_tod4 * K_grc_pc < nb.float32(0.0):
        Fe_ext_tod4 = nb.float32(0.0)

    Fi_ext = nb.float32(0.0)

    ## GrC — excitatory TF
    d_GrC = (_crbl_TF_2d_${sn.name}(
        Fe_ext_tod1 + external_input_ex_ex, GoC, nb.float32(0.0),
        Fi_ext + external_input_ex_in, nb.float32(0.0),
        Q_mf_grc, tau_mf_grc, E_e, Q_goc_grc, tau_goc_grc, E_i,
        g_L_grc, C_m_grc, E_L_grc, K_mossy_grc, K_mossy_goc, alpha_grc,
        _cp_${sn.name}_P_grc_0, _cp_${sn.name}_P_grc_1, _cp_${sn.name}_P_grc_2,
        _cp_${sn.name}_P_grc_3, _cp_${sn.name}_P_grc_4) - GrC) / T

    ## GoC — inhibitory TF (3-input: GrC, GoC, external)
    ## GoC uses external_input_in_ex (matching the monolithic model)
    d_GoC = (_crbl_TF_3d_${sn.name}(
        GrC, GoC, Fe_ext_tod2 + external_input_in_ex, nb.float32(0.0),
        Q_grc_goc, tau_grc_goc, goc_Ee, Q_goc_goc, tau_goc_goc, E_i,
        g_L_goc, C_m_goc, E_L_goc, K_grc_goc, K_goc_goc,
        K_mossy_goc, Q_mf_goc, tau_mf_goc, alpha_goc,
        _cp_${sn.name}_P_goc_0, _cp_${sn.name}_P_goc_1, _cp_${sn.name}_P_goc_2,
        _cp_${sn.name}_P_goc_3, _cp_${sn.name}_P_goc_4) - GoC) / T

    ## MLI — inhibitory TF
    d_MLI = (_crbl_TF_2d_${sn.name}(
        GrC, MLI, Fe_ext_tod3, Fi_ext, nb.float32(0.0),
        Q_grc_mli, tau_grc_mli, E_e, Q_mli_mli, tau_mli_mli, E_i,
        g_L_mli, C_m_mli, E_L_mli, K_grc_mli, K_mli_mli, alpha_mli,
        _cp_${sn.name}_P_mli_0, _cp_${sn.name}_P_mli_1, _cp_${sn.name}_P_mli_2,
        _cp_${sn.name}_P_mli_3, _cp_${sn.name}_P_mli_4) - MLI) / T

    ## PC — inhibitory TF
    d_PC = (_crbl_TF_2d_${sn.name}(
        GrC, MLI, Fe_ext_tod4, Fi_ext, nb.float32(0.0),
        Q_grc_pc, tau_grc_pc, E_e, Q_mli_pc, tau_mli_pc, E_i,
        g_L_pc, C_m_pc, E_L_pc, K_grc_pc, K_mli_pc, alpha_pc,
        _cp_${sn.name}_P_pc_0, _cp_${sn.name}_P_pc_1, _cp_${sn.name}_P_pc_2,
        _cp_${sn.name}_P_pc_3, _cp_${sn.name}_P_pc_4) - PC) / T

    ## OU noise
    d_noise = -noise / tau_OU

    return (d_GrC, d_GoC, d_MLI, d_PC, d_noise,)
