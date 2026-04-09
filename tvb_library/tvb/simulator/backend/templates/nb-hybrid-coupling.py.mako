## -*- coding: utf-8 -*-
##
## Numba code-generation template for a single hybrid projection (inter or intra).
##
## Required template variables (set in the <%def> caller or passed via content dict):
##   proj_name   : str   — unique function-suffix for this projection
##   n_tgt_nodes : int   — number of target nodes (constant at codegen time)
##   n_src_modes : int   — source mode count
##   n_tgt_modes : int   — target mode count
##   cvar_mapping : str  — one of "1_to_1", "n_to_n", "many_to_1", "1_to_many"
##   n_src_cvar  : int   — number of source coupling variables
##   n_tgt_cvar  : int   — number of target coupling variables
##   cfun_type   : str   — "none", "linear", or "scaling"
##   has_mode_map: bool  — True for inter-projections, False for intra (identity)
##
## This single template handles both inter and intra projections:
##   - intra: pass has_mode_map=False  (identity; n_src_modes == n_tgt_modes)
##   - inter: pass has_mode_map=True   (explicit mode_map argument)

import numba as nb
import numpy as np


<%def name="render_coupling(proj_name, n_tgt_nodes, n_src_modes, n_tgt_modes,
                             cvar_mapping, n_src_cvar, n_tgt_cvar,
                             cfun_type, has_mode_map)">
@nb.njit(inline="always")
def compute_coupling_${proj_name}(
    buf,            # (n_vars_src, n_nodes_src, n_src_modes, horizon)
    w_data,         # (nnz,) CSR data
    w_indices,      # (nnz,) CSR column indices
    w_indptr,       # (n_tgt_nodes+1,)
    idelays,        # (nnz,) int
    % if has_mode_map:
    mode_map,       # (n_src_modes, n_tgt_modes)
    % endif
    source_cvar,    # (n_src_cvar,) int
    target_cvar,    # (n_tgt_cvar,) int
    scale,          # float
    target_scales,  # (n_tgt_cvar,) — may be length-0 empty array if unused
    cfun_a,         # float — coupling param a
    cfun_b,         # float — coupling param b
    horizon,        # int
    t,              # int current step
    tgt,            # (n_cvar_tgt, n_nodes_tgt, n_tgt_modes) — INPLACE output
):
    n_src_cvar = source_cvar.shape[0]
    n_tgt_cvar = target_cvar.shape[0]
    has_ts = target_scales.shape[0] > 0

    for j in range(${n_tgt_nodes}):
        row_start = w_indptr[j]
        row_end = w_indptr[j + 1]

        for ic in range(n_src_cvar):
            cv = source_cvar[ic]
            % if has_mode_map:
            # Accumulate weighted-delayed sum in src-mode space: (n_src_modes,)
            # then apply mode_map to get (n_tgt_modes,)
            wsum = np.zeros(${n_src_modes})
            % else:
            wsum = np.zeros(${n_src_modes})
            % endif

            for ptr in range(row_start, row_end):
                w = w_data[ptr]
                src_node = w_indices[ptr]
                buf_idx = (t - 1 - idelays[ptr] + horizon) % horizon
                for m in range(${n_src_modes}):
                    wsum[m] += w * buf[cv, src_node, m, buf_idx]

            # Apply cfun.pre() — identity for "none"
            % if cfun_type == "none":
            # no pre transform
            % elif cfun_type == "linear":
            # Linear.pre is identity
            # no pre transform
            % elif cfun_type == "scaling":
            # Scaling.pre is identity
            # no pre transform
            % endif

            # Apply scale
            for m in range(${n_src_modes}):
                wsum[m] *= scale

            # Apply cfun.post()
            % if cfun_type == "none":
            # no post transform
            % elif cfun_type == "linear":
            for m in range(${n_src_modes}):
                wsum[m] = cfun_a * wsum[m] + cfun_b
            % elif cfun_type == "scaling":
            for m in range(${n_src_modes}):
                wsum[m] = cfun_a * wsum[m]
            % endif

            # Apply mode map and accumulate
            % if has_mode_map:
            # inter: mode_map (n_src_modes, n_tgt_modes)
            for m_tgt in range(${n_tgt_modes}):
                contrib = np.float32(0.0)
                for m_src in range(${n_src_modes}):
                    contrib += wsum[m_src] * mode_map[m_src, m_tgt]
                % if cvar_mapping in ("1_to_1", "n_to_n"):
                ts = target_scales[ic] if has_ts else np.float32(1.0)
                tgt[target_cvar[ic], j, m_tgt] += ts * contrib
                % elif cvar_mapping == "many_to_1":
                ts = target_scales[0] if has_ts else np.float32(1.0)
                tgt[target_cvar[0], j, m_tgt] += ts * contrib
                % elif cvar_mapping == "1_to_many":
                for itc in range(n_tgt_cvar):
                    ts = target_scales[itc] if has_ts else np.float32(1.0)
                    tgt[target_cvar[itc], j, m_tgt] += ts * contrib
                % endif
            % else:
            # intra: identity mode map (n_src_modes == n_tgt_modes)
            for m in range(${n_src_modes}):
                contrib = wsum[m]
                % if cvar_mapping in ("1_to_1", "n_to_n"):
                ts = target_scales[ic] if has_ts else np.float32(1.0)
                tgt[target_cvar[ic], j, m] += ts * contrib
                % elif cvar_mapping == "many_to_1":
                ts = target_scales[0] if has_ts else np.float32(1.0)
                tgt[target_cvar[0], j, m] += ts * contrib
                % elif cvar_mapping == "1_to_many":
                for itc in range(n_tgt_cvar):
                    ts = target_scales[itc] if has_ts else np.float32(1.0)
                    tgt[target_cvar[itc], j, m] += ts * contrib
                % endif
            % endif

</%def>
