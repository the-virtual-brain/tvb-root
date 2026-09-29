# -*- coding: utf-8 -*-
"""Expression-driven C++ dfun generator for the CppHybrid backend.

TVB models declare their derivatives as Python expression strings
(``state_variable_dfuns``) plus optional ``dfun_intermediates``,
``dfun_helpers`` and ``dfun_constants``.  This module translates those
expressions into small per-model C++ dfun functions (one function per model,
SIMD-lane layout identical to the hand-written kernels), compiles them into a
shared library at cache time and registers them with the C++ core.

This is NOT monolithic kernel generation: only per-model derivative
functions are emitted; all control flow (integration, coupling, monitors,
sweeps) stays in the prebuilt C++ core.
"""

import ast
import importlib
import inspect
import os
import subprocess
import textwrap
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# expression translation
# ---------------------------------------------------------------------------

_FN_MAP = {
    "sin": "sin", "cos": "cos", "tan": "tan", "exp": "exp", "log": "log",
    "log10": "log10", "sqrt": "sqrt", "tanh": "tanh", "sinh": "sinh",
    "cosh": "cosh", "atan": "atan", "asin": "asin", "acos": "acos",
    "abs": "fabs", "fabs": "fabs", "floor": "floor", "ceil": "ceil",
    "sign": "cph_sign", "minimum": "fmin", "maximum": "fmax",
    "pow": "pow", "arctan": "atan", "arcsin": "asin", "arccos": "acos",
    "log1p": "log1p", "expm1": "expm1", "hypot": "hypot", "cbrt": "cbrt",
}

_PRELUDE = r"""
#include <cmath>
#include <cstdint>
#include <complex>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline double cph_sign(double v) { return v > 0 ? 1.0 : (v < 0 ? -1.0 : 0.0); }
static inline double cph_pow(double b, double e) {
  // python ** semantics; complex when the base would be negative
  if (b >= 0.0 || std::floor(e) == e) return std::pow(b, e);
  return std::pow(std::complex<double>(b, 0.0), e).real();
}
static inline std::complex<double> cph_pow(std::complex<double> b, double e) {
  return std::pow(b, e);
}
static inline std::complex<double> cplx(double v) { return {v, 0.0}; }
static inline std::complex<double> cplx(const std::complex<double> &v) { return v; }
"""


class Expr2Cpp(ast.NodeVisitor):
    def __init__(self, names, ctx, helpers=(), allow_mode=False):
        self.names = names          # known identifiers
        self.ctx = ctx              # 'complex' if complex ops used
        self.helpers = set(helpers)
        self.allow_mode = allow_mode
        self.out = []

    def __call__(self, node):
        self.out.append(self.visit(node))
        return self.out[-1]

    def visit_Name(self, node):
        n = node.id
        if n in self.names:
            return n
        if self.allow_mode and n.endswith("_m"):
            base = n[:-2]
            if base in self.names:
                return base
        raise ValueError(f"unknown identifier {n!r}")

    def visit_Constant(self, node):
        v = node.value
        if isinstance(v, complex):
            self.ctx["complex"] = True
            return f"std::complex<double>({v.real!r}, {v.imag!r})"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return repr(float(v))
        raise ValueError(f"unsupported constant {v!r}")

    def visit_BinOp(self, node):
        l, r = self.visit(node.left), self.visit(node.right)
        op = type(node.op)
        if op is ast.Pow:
            # complex-safe power
            self.ctx["complex_maybe"] = True
            return f"cph_pow({l}, {r})"
        sym = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
               ast.Mod: "fmod", ast.LShift: "<<", ast.RShift: ">>",
               ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^"}.get(op)
        if sym is None:
            raise ValueError(f"unsupported binop {op}")
        return f"({l} {sym} {r})"

    def visit_UnaryOp(self, node):
        v = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return f"(-{v})"
        if isinstance(node.op, ast.UAdd):
            return f"(+{v})"
        if isinstance(node.op, ast.Not):
            return f"(!(bool)({v}))"
        raise ValueError("unsupported unaryop")

    def visit_Compare(self, node):
        l = self.visit(node.left)
        out = None
        for op, comp in zip(node.ops, node.comparators):
            r = self.visit(comp)
            sym = {ast.Lt: "<", ast.Gt: ">", ast.LtE: "<=", ast.GtE: ">=",
                   ast.Eq: "==", ast.NotEq: "!=", ast.In: "==",
                   ast.NotIn: "!="}.get(type(op))
            if sym is None:
                raise ValueError(f"unsupported cmp {op}")
            expr = f"({l} {sym} {r})"
            out = expr if out is None else f"(({out}) && ({expr}))"
            l = r
        return out

    def visit_BoolOp(self, node):
        sym = "&&" if isinstance(node.op, ast.And) else "||"
        parts = [f"(bool)({self.visit(v)})" for v in node.values]
        return "(" + f" {sym} ".join(parts) + ")"

    def visit_Call(self, node):
        fname = None
        if isinstance(node.func, ast.Name):
            fname = node.func.id
        elif isinstance(node.func, ast.Attribute):
            fname = node.func.attr
        if fname == "float32":  # nb.float32 / np.float32 casts
            return f"(double)({self.visit(node.args[0])})"
        if fname in self.helpers:
            args = ", ".join(self.visit(a) for a in node.args)
            return f"{fname}({args})"
        if fname in _FN_MAP:
            args = ", ".join(self.visit(a) for a in node.args)
            return f"{_FN_MAP[fname]}({args})"
        if fname == "where":
            c, a, b = (self.visit(x) for x in node.args)
            return f"(({c}) ? ({a}) : ({b}))"
        raise ValueError(f"unsupported function {fname!r}")

    def visit_Attribute(self, node):
        # math.pi / np.pi etc.
        base = node.value
        if isinstance(base, ast.Name) and base.id in ("math", "np", "numpy"):
            if node.attr == "pi":
                return "M_PI"
            if node.attr in _FN_MAP:
                return _FN_MAP[node.attr]
            raise ValueError(f"unsupported {base.id}.{node.attr}")
        v = self.visit(node.value)
        if node.attr == "real":
            self.ctx["complex"] = True
            return f"({v}).real()"
        if node.attr == "imag":
            self.ctx["complex"] = True
            return f"({v}).imag()"
        raise ValueError(f"unsupported attribute {node.attr!r}")

    def visit_IfExp(self, node):
        return (f"(({self.visit(node.test)}) ? ({self.visit(node.body)})"
                f" : ({self.visit(node.orelse)}))")

    def visit_Subscript(self, node):
        # e.g. theta[sliceT] — treat scalars as-is
        return self.visit(node.value)

    def visit_List(self, node):
        raise ValueError("lists unsupported in dfun expressions")


def cph_pow(l, r):
    return f"cph_pow({l}, {r})"


def translate(expr, names, ctx, helpers=(), allow_mode=False):
    expr = expr.replace("_{m}", "_m")  # reduced-set subscripts
    tree = ast.parse(expr.strip(), mode="eval")
    return Expr2Cpp(names, ctx, helpers, allow_mode=allow_mode)(tree.body)


# ---------------------------------------------------------------------------
# model collection
# ---------------------------------------------------------------------------

def collect_models():
    """Map model class name -> configured instance for all expression-driven
    models (those declaring state_variable_dfuns)."""
    import pkgutil
    from tvb.simulator.models.base import Model
    found = {}
    import tvb.simulator.models as pkg
    mods = []
    if hasattr(pkg, '__path__'):
        for mi in pkgutil.iter_modules(pkg.__path__):
            try:
                mods.append(importlib.import_module(
                    f"tvb.simulator.models.{mi.name}"))
            except Exception:
                pass
    else:
        mods = [getattr(pkg, m) for m in dir(pkg)
                if inspect.ismodule(getattr(pkg, m))]
    for mod in mods:
        for cn in dir(mod):
            cls = getattr(mod, cn)
            if isinstance(cls, type) and issubclass(cls, Model) and cls is not Model:
                if cn in found:
                    continue
                try:
                    o = cls()
                    o.configure()
                except Exception:
                    continue
                svd = getattr(o, "state_variable_dfuns", None)
                if isinstance(svd, dict) and svd:
                    found[cn] = o
    return found


# ---------------------------------------------------------------------------
# C++ emission
# ---------------------------------------------------------------------------

def _emit_model(mid, model, fns):
    svars = list(model.state_variables)
    cvars = list(model.coupling_terms) if getattr(model, "coupling_terms", None) else []
    globals_ = list(model.global_parameter_names)
    spatial = list(model.spatial_parameter_names)
    params = globals_ + spatial
    constants = getattr(model, "dfun_constants", None) or {}
    helpers = getattr(model, "dfun_helpers", None) or []
    inter = getattr(model, "dfun_intermediates", None) or []

    is_combined = getattr(model, "dfun_mode", None) == "combined"
    dm_names = (list(getattr(model, "derived_matrix_names", [])) if is_combined
                else [])
    dm_ops = (list(getattr(model, "derived_matrix_ops", [])) if is_combined
              else [])
    op_names = [op[0] for op in dm_ops]
    dm_data = {}
    if is_combined:
        for dn in dm_names:
            arr = getattr(model, dn, None)
            if arr is not None:
                dm_data[dn] = np.asarray(arr, dtype=np.float32).ravel()

    ctx = {}
    helper_names = {h[0] for h in helpers}
    helpers = list(helpers)
    known = (set(params) | set(constants) | set(svars) | set(cvars)
             | {"pi"} | helper_names | set(dm_names) | set(op_names))
    translated_inter = []
    for iname, iexpr in inter:
        translated_inter.append((iname, translate(iexpr, known, ctx, helper_names,
                                                  allow_mode=is_combined)))
        known.add(iname)
    translated_dx = []
    for sv in svars:
        translated_dx.append(
            (sv, translate(model.state_variable_dfuns[sv], known, ctx, helper_names,
                           allow_mode=is_combined)))
    use_complex = bool(ctx.get("complex"))
    T = "std::complex<double>" if use_complex else "double"
    wrap = (lambda e: f"cplx({e})") if use_complex else (lambda e: f"({e})")

    lines = []
    fn_name = f"dfun_gen_{mid}"
    for hname, hargs, hexpr in helpers:
        arg_names = [a.strip() for a in hargs.split(",")]
        body = translate(hexpr, set(arg_names) | set(constants), ctx, helper_names,
                         allow_mode=is_combined)
        decls = "".join(
            f"  const double {cname} = {float(cval)!r};\n"
            for cname, cval in constants.items())
        lines.append(
            f"static inline double {hname}({', '.join(f'double {a}' for a in arg_names)}) {{\n"
            f"{decls}"
            f"  return {body};\n}}")

    lines.append(
        f"static void {fn_name}(float *__restrict dxarr, "
        f"const float *__restrict xarr, "
        f"int node, int mode, int n_node, int n_modes, "
        f"const float *__restrict carr, "
        f"const float *__restrict parr, int n_parm, int Wn) {{")
    lines.append(f"  const int n_svar = {len(svars)}, n_cvar = {max(len(cvars), 1)};")
    lines.append("  (void)n_svar; (void)n_cvar; (void)n_parm;")
    if not cvars:
        lines.append("  (void)carr;")
    lines.append("  const size_t sw = (size_t)n_modes * (size_t)Wn;")
    lines.append("  const size_t nb = ((size_t)node * (size_t)n_modes + (size_t)mode) * (size_t)Wn;")
    if is_combined:
        for dn in dm_names:
            d = dm_data.get(dn)
            if d is not None:
                lines.append(
                    f"  const double {dn}_lit[{d.size}] = "
                    f"{{ {', '.join(repr(float(x)) for x in d)} }};")
    lines.append("  const float *pk = parr + (size_t)node * n_parm * Wn;")
    lines.append("  (void)pk;")
    lines.append("  for (int i = 0; i < Wn; i++) {")
    lines.append("    const double pi = M_PI; (void)pi;")
    for k, name in enumerate(params):
        if use_complex:
            lines.append(f"    const std::complex<double> {name} = cplx((double)parr[{k} * Wn + i]);")
        else:
            lines.append(f"    const double {name} = (double)parr[{k} * Wn + i];")
    for cname, cval in constants.items():
        lines.append(f"    const double {cname} = {float(cval)!r};")
    for si, sv in enumerate(svars):
        lines.append(
            f"    const {T} {sv} = {wrap(f'xarr[(size_t){si} * (size_t)n_node * sw + nb + i]')};")
    if is_combined:
        # mode-indexed derived vector constants
        for dn in dm_names:
            d = dm_data.get(dn)
            if d is not None and d.size > 1:
                # Aik/Bik/Cik are matrices (only used in ops); the per-mode
                # scalars sit at index == n_modes
                lines.append(
                    f"    const double {dn} = {dn}_lit[((size_t)mode)];")
        # coupling summed over source modes (combined-mode convention)
        for ci, cv in enumerate(cvars):
            lines.append(f"    double {cv} = 0.0;")
            lines.append(
                f"    for (int jj = 0; jj < n_modes; jj++) "
                f"{cv} += carr[((size_t){ci} * (size_t)n_node * sw + nb"
                f" - ((size_t)mode)*Wn + (size_t)jj * Wn + i)];")
        # derived op intermediates: sum over source modes using matrix[mk][mode]
        for op_name, op_mat, op_svar in dm_ops:
            lines.append(f"    double {op_name} = 0.0;")
            lines.append(
                f"    for (int jj = 0; jj < n_modes; jj++) "
                f"{op_name} += {op_mat}_lit[(size_t)jj * (size_t)n_modes + (size_t)mode] * "
                f"xarr[((size_t){svars.index(op_svar)} * (size_t)n_node * sw + nb"
                f" - ((size_t)mode) * Wn + (size_t)jj * Wn + i)];")
    else:
        for ci, cv in enumerate(cvars):
            lines.append(
                f"    const {T} {cv} = {wrap(f'carr[(size_t){ci} * (size_t)n_node * sw + nb + i]')};")
    for iname, code in translated_inter:
        lines.append(f"    const {T} {iname} = {wrap(code)};")
    for si, (sv, code) in enumerate(translated_dx):
        real = f"(({code}).real())" if use_complex else f"({code})"
        lines.append(
            f"    dxarr[(size_t){si} * (size_t)n_node * sw + nb + i] = (float){real};")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines), len(params), len(cvars), len(svars)


def generate_lib(cache_dir: Path, extra_models: dict = None):
    """Generate + compile the generic dfun shared library.

    Returns (lib_path, model_ids) where model_ids maps model class name to
    the integer id used in the dispatch table (ids >= 100).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    models = collect_models()
    if extra_models:
        models.update(extra_models)

    src = [_PRELUDE, """
extern "C" {
"""]
    model_ids = {}
    mid = 100
    meta = {}
    for name, model in sorted(models.items()):
        try:
            code, n_parm, n_cvar, n_svar = _emit_model(mid, model, None)
        except Exception:
            continue
        src.append(code)
        model_ids[name] = mid
        meta[name] = dict(mid=mid, n_parm=n_parm, n_cvar=n_cvar, n_svar=n_svar)
        mid += 1

    src.append("""
typedef void (*dfun_fn_t)(float*, const float*, int, int, int, int,
                          const float*, const float*, int, int);

static dfun_fn_t _dfuns[] = {
""")
    for name, m in meta.items():
        src.append(f"  dfun_gen_{m['mid']},")
    src.append("};")
    src.append(f"#define N_GEN_DFUNS {len(meta)}")
    src.append("""
extern "C" void cph_generic_dfun(int model_id, float *dx, const float *x,
                                 int node, int mode, int n_node, int n_modes,
                                 const float *c, const float *p, int n_parm,
                                 int Wn) {
  int idx = model_id - 100;
  if (idx < 0 || idx >= N_GEN_DFUNS) return;
  _dfuns[idx](dx, x, node, mode, n_node, n_modes, c, p, n_parm, Wn);
}

}  // extern "C"
""")
    src_path = cache_dir / "models_gen.cpp"
    src_path.write_text("\n".join(src))
    so_path = cache_dir / "models_gen.so"
    inc = np.get_include()
    cmd = (f"g++ -O3 -march=native -fopenmp-simd -std=c++17 -fPIC -shared "
           f"-I{inc} {src_path} -o {so_path}")
    subprocess.run(cmd, shell=True, check=True, capture_output=True)
    return so_path, model_ids, meta


_META_CACHE = None


def generate_meta():
    """Model metadata (ids/counts) for the generated library."""
    global _META_CACHE
    if _META_CACHE is None:
        models = collect_models()
        mid = 100
        meta = {}
        for name, model in sorted(models.items()):
            try:
                # only models that actually emit get an id (must stay in
                # lockstep with generate_lib's assignment)
                _emit_model(mid, model, None)
            except Exception:
                continue
            cvars = list(model.coupling_terms) if getattr(
                model, "coupling_terms", None) else []
            params = (list(model.global_parameter_names)
                      + list(model.spatial_parameter_names))
            meta[name] = dict(
                mid=mid, n_parm=len(params), n_cvar=max(len(cvars), 1),
                n_svar=len(model.state_variables))
            mid += 1
        _META_CACHE = meta
    return _META_CACHE
