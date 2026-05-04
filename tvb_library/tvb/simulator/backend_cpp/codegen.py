from __future__ import annotations

import ast as _ast
import dataclasses
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import shutil
import shlex
import sysconfig
import subprocess
import sys

from mako.exceptions import text_error_template
from mako.lookup import TemplateLookup
from mako.template import Template

from .spec import SimulationSpec, SubnetworkSpec


TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
RUNTIME_DIR = Path(__file__).resolve().parent / "runtime"
DEFAULT_SIM_TEMPLATE = TEMPLATES_DIR / "sim_module.cpp.mako"
DEFAULT_BINDINGS_TEMPLATE = TEMPLATES_DIR / "module_bindings.cpp.mako"
DEFAULT_CMAKE_TEMPLATE = TEMPLATES_DIR / "CMakeLists.txt.mako"

# ---------------------------------------------------------------------------
# Python → C++ expression translator
# ---------------------------------------------------------------------------

_BARE_TO_STD: dict[str, str] = {
    "exp": "std::exp",
    "tanh": "std::tanh",
    "sinh": "std::sinh",
    "cosh": "std::cosh",
    "sqrt": "std::sqrt",
    "sin": "std::sin",
    "cos": "std::cos",
    "log": "std::log",
    "log2": "std::log2",
    "log10": "std::log10",
    "abs": "std::abs",
    "fabs": "std::fabs",
    "atan": "std::atan",
    "atan2": "std::atan2",
    "floor": "std::floor",
    "ceil": "std::ceil",
    "pow": "std::pow",
}


class _CppExprGen(_ast.NodeVisitor):
    """Walk a Python expression AST and emit equivalent C++ source."""

    def __init__(
        self,
        svar_to_idx: dict[str, int],
        param_names: set[str],
        coupling_names: set[str],
        local_names: set[str],
    ) -> None:
        self._svar = svar_to_idx
        self._params = param_names
        self._coupling = coupling_names
        self._locals = local_names

    def visit_Constant(self, node: _ast.Constant) -> str:
        if isinstance(node.value, bool):
            return "true" if node.value else "false"
        if isinstance(node.value, float):
            return f"{node.value:.17g}"
        return str(node.value)

    def visit_Name(self, node: _ast.Name) -> str:
        name = node.id
        if name == "pi":
            return "M_PI"
        if name in self._svar:
            return f"state({self._svar[name]}, node, 0)"
        if name in self._params:
            return f"param_at(kParam_{name}, node)"
        if name in self._coupling or name in self._locals:
            return name
        return name  # fallback — let C++ catch unknown identifiers

    def visit_BinOp(self, node: _ast.BinOp) -> str:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, _ast.Pow):
            return f"std::pow({left}, {right})"
        op = {
            _ast.Add: "+",
            _ast.Sub: "-",
            _ast.Mult: "*",
            _ast.Div: "/",
        }.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported binary op: {type(node.op).__name__}")
        return f"({left} {op} {right})"

    def visit_UnaryOp(self, node: _ast.UnaryOp) -> str:
        operand = self.visit(node.operand)
        if isinstance(node.op, _ast.USub):
            return f"(-{operand})"
        if isinstance(node.op, _ast.UAdd):
            return operand
        raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")

    def visit_Call(self, node: _ast.Call) -> str:
        args = [self.visit(a) for a in node.args]

        if isinstance(node.func, _ast.Attribute):
            obj = node.func.value
            attr = node.func.attr
            if isinstance(obj, _ast.Name):
                if obj.id == "nb" and attr == "float32":
                    # nb.float32(x) → x (drop the cast)
                    return args[0] if args else "0.0"
                if obj.id == "math":
                    cpp_fn = _BARE_TO_STD.get(attr, f"std::{attr}")
                    return f"{cpp_fn}({', '.join(args)})"

        if isinstance(node.func, _ast.Name):
            fn = node.func.id
            if fn in _BARE_TO_STD:
                return f"{_BARE_TO_STD[fn]}({', '.join(args)})"
            # User-defined helper or unknown — pass through
            return f"{fn}({', '.join(args)})"

        raise ValueError(f"Unsupported call: {_ast.dump(node.func)}")

    def visit_IfExp(self, node: _ast.IfExp) -> str:
        # Python:  body if test else orelse
        # C++:     (test ? body : orelse)
        cond = self.visit(node.test)
        then = self.visit(node.body)
        else_ = self.visit(node.orelse)
        return f"({cond} ? {then} : {else_})"

    def visit_Compare(self, node: _ast.Compare) -> str:
        _op_map = {
            _ast.Lt: "<", _ast.LtE: "<=",
            _ast.Gt: ">", _ast.GtE: ">=",
            _ast.Eq: "==", _ast.NotEq: "!=",
        }
        parts = [self.visit(node.left)]
        for op, comp in zip(node.ops, node.comparators):
            parts.append(_op_map[type(op)])
            parts.append(self.visit(comp))
        return " ".join(parts)

    def generic_visit(self, node: _ast.AST) -> str:
        raise ValueError(f"Unsupported AST node: {type(node).__name__}: {_ast.dump(node)}")


def py_expr_to_cpp(
    expr: str,
    svar_to_idx: dict[str, int],
    param_names: set[str],
    coupling_names: set[str],
    local_names: set[str],
) -> str:
    """Translate a Python math expression string to a C++ expression string."""
    tree = _ast.parse(expr.strip(), mode="eval")
    return _CppExprGen(svar_to_idx, param_names, coupling_names, local_names).visit(
        tree.body
    )


def _build_dfun_context(subnet: SubnetworkSpec) -> dict:
    """Pre-translate all dfun expressions to C++ and return template context."""
    svar_to_idx = {sv: i for i, sv in enumerate(subnet.state_variables)}
    param_names = set(subnet.global_parameter_names)
    coupling_names = set(subnet.coupling_terms)
    intermediate_names = {name for name, _ in subnet.dfun_intermediates}
    helper_names = {name for name, _, _ in subnet.dfun_helpers}
    local_names = intermediate_names | helper_names

    def translate(expr: str, extra_locals: set[str] | None = None) -> str:
        # State vars and params are declared as C++ locals at the top of
        # compute_dfun, so pass empty maps and add them to local_names.
        # This keeps their short identifiers in translated expressions,
        # ensuring the declarations are actually referenced (no -Wunused).
        all_locals = (
            set(svar_to_idx.keys())
            | param_names
            | local_names
            | (extra_locals or set())
        )
        return py_expr_to_cpp(expr, {}, set(), coupling_names, all_locals)

    def translate_expanded(expr: str, extra_locals: set[str] | None = None) -> str:
        # For contexts without local declarations (e.g. compute_voi), use the
        # full-expansion form so state vars and params resolve to state()/param_at().
        return py_expr_to_cpp(
            expr, svar_to_idx, param_names, coupling_names,
            local_names | (extra_locals or set()),
        )

    # Helper function declarations (struct members, 2-space indent)
    helper_decls: list[str] = []
    for h_name, h_args, h_body in subnet.dfun_helpers:
        arg_names = {a.strip() for a in h_args.split(",")}
        cpp_args = ", ".join(f"double {a.strip()}" for a in h_args.split(","))
        # Translate body with NO model context — only the function's own args
        cpp_body = py_expr_to_cpp(h_body, {}, set(), set(), arg_names)
        helper_decls.append(
            f"  static inline double {h_name}({cpp_args}) {{\n"
            f"    return {cpp_body};\n"
            f"  }}"
        )

    # State variable reads inside compute_dfun
    state_reads = [
        f"const double {sv} = state({i}, node, 0);"
        for i, sv in enumerate(subnet.state_variables)
    ]

    # Parameter reads (only globals referenced in dfun expressions)
    param_reads = [
        f"const double {p} = param_at(kParam_{p}, node);"
        for p in subnet.global_parameter_names
    ]

    # Coupling reads — from the pre-accumulated coupling array.
    # Layout: coupling[cvar_slot * kNumNodes + node]
    coupling_reads = [
        f"const double {ct} = coupling[{i} * kNumNodes + node];"
        for i, ct in enumerate(subnet.coupling_terms)
    ]

    # Intermediate variable declarations
    intermediate_decls = [
        f"const double {name} = {translate(expr)};"
        for name, expr in subnet.dfun_intermediates
    ]

    # dx[i] = ... assignments
    dx_assignments = [
        f"dx[{i}] = {translate(subnet.state_variable_dfuns[sv])};"
        for i, sv in enumerate(subnet.state_variables)
    ]

    def _cpp_double(v: float) -> str:
        s = f"{v:.17g}"
        if "." not in s and "e" not in s and "E" not in s:
            s += ".0"
        return s

    # apply_state_constraints body
    constraint_stmts: list[str] = []
    for i, sv in enumerate(subnet.state_variables):
        lo = subnet.state_lower_bounds.get(sv)
        hi = subnet.state_upper_bounds.get(sv)
        if lo is not None:
            constraint_stmts.append(
                f"state({i}, node, 0) = std::max({_cpp_double(lo)}, state({i}, node, 0));"
            )
        if hi is not None:
            constraint_stmts.append(
                f"state({i}, node, 0) = std::min({_cpp_double(hi)}, state({i}, node, 0));"
            )

    # VOI computation — compute_voi has no local declarations, so use
    # the expanded translator (state()/param_at() forms) throughout.
    voi_assignments: list[str] = []
    for ivoi, voi_name in enumerate(subnet.variables_of_interest):
        if voi_name in svar_to_idx:
            voi_assignments.append(f"voi[{ivoi}] = state({svar_to_idx[voi_name]}, node, 0);")
        else:
            voi_assignments.append(f"voi[{ivoi}] = {translate_expanded(voi_name)};")

    return {
        "dfun_helper_decls": helper_decls,
        "dfun_state_reads": state_reads,
        "dfun_param_reads": param_reads,
        "dfun_coupling_reads": coupling_reads,
        "dfun_intermediate_decls": intermediate_decls,
        "dfun_dx_assignments": dx_assignments,
        "dfun_constraint_stmts": constraint_stmts,
        "dfun_voi_assignments": voi_assignments,
        "n_coupling_vars": len(subnet.coupling_terms),
    }


@dataclasses.dataclass(frozen=True)
class GeneratedSourceArtifact:
    module_name: str
    build_dir: Path
    cpp_path: Path
    bindings_cpp_path: Path
    cmake_lists_path: Path
    runtime_header_path: Path
    sim_template_path: Path
    bindings_template_path: Path
    cmake_template_path: Path
    extension_path: Path | None = None
    sim_source_text: str = ""
    bindings_source_text: str = ""
    cmake_source_text: str = ""



def _render_mako_template(template_path: Path, ctx: dict) -> str:
    lookup = TemplateLookup(directories=[str(template_path.parent)])
    tmpl = Template(
        template_path.read_text(encoding="utf-8"),
        lookup=lookup,
        strict_undefined=True,
    )
    try:
        return tmpl.render(**ctx)
    except Exception:
        print(text_error_template().render())
        raise


def _format_projection_summary(spec: SimulationSpec) -> str:
    if not spec.all_projections:
        return "// projections: none"

    lines: list[str] = ["// projections:"]
    for projection in spec.all_projections:
        lines.append(
            "//   "
            f"{projection.name}: {projection.source_subnet} -> {projection.target_subnet}, "
            f"cfun={projection.cfun_type}, "
            f"mapping={projection.cvar_mapping_mode}, "
            f"horizon={projection.horizon}, "
            f"nnz={projection.weights_data.shape[0]}"
        )
    return "\n".join(lines)


def _format_subnetwork_summary(spec: SimulationSpec) -> str:
    lines: list[str] = ["// subnetworks:"]
    for subnet in spec.subnetworks:
        lines.append(
            "//   "
            f"{subnet.name}: model={subnet.model_type}, "
            f"integrator={subnet.integrator.type_name}, "
            f"nodes={subnet.n_nodes}, "
            f"modes={subnet.n_modes}, "
            f"state_vars={subnet.n_state_vars}, "
            f"coupling_vars={subnet.n_coupling_vars}"
        )
    return "\n".join(lines)


def _validate_spec(spec: SimulationSpec) -> None:
    """Validate that all subnets are supported by the C++ backend."""
    for subnet in spec.subnetworks:
        if not subnet.state_variable_dfuns:
            raise NotImplementedError(
                f"Model '{subnet.model_type}' has no state_variable_dfuns — "
                f"C++ dfun generation requires the standard expression-based interface."
            )
        if subnet.integrator.type_name not in (
            "EulerDeterministic",
            "EulerStochastic",
            "HeunDeterministic",
            "HeunStochastic",
        ):
            raise NotImplementedError(
                f"Subnet '{subnet.name}': only Euler{{Deterministic,Stochastic}} and "
                f"Heun{{Deterministic,Stochastic}} are currently supported "
                f"(got '{subnet.integrator.type_name}')."
            )
        if subnet.n_modes != 1:
            raise NotImplementedError(
                f"Subnet '{subnet.name}': only single-mode subnetworks are currently "
                f"supported (got n_modes={subnet.n_modes})."
            )
    _SUPPORTED_MONITORS = {"TemporalAverage", "Raw", "RawVoi",
                           "AfferentCoupling", "AfferentCouplingTemporalAverage"}
    for monitor in spec.monitors:
        if monitor.type_name not in _SUPPORTED_MONITORS:
            raise NotImplementedError(
                f"Monitor '{monitor.type_name}' is not yet supported by the C++ backend. "
                f"Supported: {sorted(_SUPPORTED_MONITORS)}."
            )


def _history_horizon(spec: SimulationSpec, subnet_name: str) -> int:
    """History buffer depth for a given source subnet.

    Covers intra-projections from that subnet and inter-projections that use
    that subnet as the source (both read from this subnet's history).
    """
    base = int(spec.source_horizons.get(subnet_name, 1))
    for proj in spec.all_projections:
        if proj.source_subnet == subnet_name:
            base = max(base, int(proj.horizon))
    return max(base, 1)


def _build_subnets_ctx(spec: SimulationSpec) -> list[dict]:
    """Build per-subnet context dicts for the Mako template.

    Each dict contains:
      subnet            — SubnetworkSpec
      dfun_ctx          — result of _build_dfun_context()
      horizon           — history buffer depth for this subnet
      intra_proj_indices — indices into spec.intra_projections targeting this subnet
      inter_proj_targets — list of (proj_idx, source_subnet_idx) for inter-projections
                           that target this subnet (ordered by proj_idx)
    """
    subnet_name_to_idx = {sn.name: i for i, sn in enumerate(spec.subnetworks)}
    subnets_ctx: list[dict] = []
    for si, subnet in enumerate(spec.subnetworks):
        intra_proj_indices = [
            pi
            for pi, proj in enumerate(spec.intra_projections)
            if proj.target_subnet == subnet.name
        ]
        inter_proj_targets = [
            (pi, subnet_name_to_idx[proj.source_subnet])
            for pi, proj in enumerate(spec.inter_projections)
            if proj.target_subnet == subnet.name
        ]
        subnets_ctx.append({
            "subnet": subnet,
            "dfun_ctx": _build_dfun_context(subnet),
            "horizon": _history_horizon(spec, subnet.name),
            "intra_proj_indices": intra_proj_indices,
            "inter_proj_targets": inter_proj_targets,
            "is_stochastic": subnet.integrator.is_stochastic,
            "is_euler": subnet.integrator.type_name in ("EulerDeterministic", "EulerStochastic"),
        })
    return subnets_ctx


def render_cpp_template(
    spec: SimulationSpec,
    module_name: str,
    template_path: Path = DEFAULT_SIM_TEMPLATE,
) -> str:
    _validate_spec(spec)
    subnets_ctx = _build_subnets_ctx(spec)
    subnet_name_to_idx = {sn.name: i for i, sn in enumerate(spec.subnetworks)}
    all_inter_proj_routes = [
        (pi, subnet_name_to_idx[proj.source_subnet], subnet_name_to_idx[proj.target_subnet])
        for pi, proj in enumerate(spec.inter_projections)
    ]
    ctx = {
        "module_name": module_name,
        "cache_key": spec.cache_key(),
        "backend_version": spec.backend_version,
        "user_source_hint": spec.user_source_hint or "",
        "dt": spec.dt,
        "num_subnetworks": len(spec.subnetworks),
        "num_inter_projections": len(spec.inter_projections),
        "num_intra_projections": len(spec.intra_projections),
        "num_monitors": len(spec.monitors),
        "subnetwork_summary": _format_subnetwork_summary(spec),
        "projection_summary": _format_projection_summary(spec),
        "subnets_ctx": subnets_ctx,
        "all_inter_proj_routes": all_inter_proj_routes,
    }
    return _render_mako_template(template_path, ctx)


def render_bindings_template(
    spec: SimulationSpec,
    module_name: str,
    generated_cpp_filename: str,
    template_path: Path = DEFAULT_BINDINGS_TEMPLATE,
) -> str:
    subnets_ctx = _build_subnets_ctx(spec)
    ctx = {
        "module_name": module_name,
        "generated_cpp_filename": generated_cpp_filename,
        "num_subnetworks": len(spec.subnetworks),
        "subnets_ctx": subnets_ctx,
    }
    return _render_mako_template(template_path, ctx)


def render_cmake_template(
    module_name: str,
    bindings_cpp_filename: str,
    template_path: Path = DEFAULT_CMAKE_TEMPLATE,
) -> str:
    ctx = {
        "module_name": module_name,
        "bindings_cpp_filename": bindings_cpp_filename,
        "python_executable": sys.executable,
    }
    return _render_mako_template(template_path, ctx)


def _discover_extension_path(build_dir: Path, module_name: str) -> Path:
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = build_dir / f"{module_name}{suffix}"
        if candidate.exists():
            return candidate
    matches = sorted(build_dir.glob(f"{module_name}*.so"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Built extension for module '{module_name}' was not found in {build_dir}"
    )


def generate_cpp_source(
    spec: SimulationSpec,
    build_dir: str | Path,
    module_name: str,
    sim_template_path: Path = DEFAULT_SIM_TEMPLATE,
    bindings_template_path: Path = DEFAULT_BINDINGS_TEMPLATE,
    cmake_template_path: Path = DEFAULT_CMAKE_TEMPLATE,
    verbose: bool = False,
) -> GeneratedSourceArtifact:
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"[tvb-cpp] Generating C++ source → {build_dir}")
    cpp_path = build_dir / f"{module_name}.cpp"
    bindings_cpp_path = build_dir / f"{module_name}_bindings.cpp"
    cmake_lists_path = build_dir / "CMakeLists.txt"
    runtime_dir = build_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_header_path = runtime_dir / "runtime.hpp"

    sim_source_text = render_cpp_template(
        spec=spec,
        module_name=module_name,
        template_path=sim_template_path,
    )
    bindings_source_text = render_bindings_template(
        spec=spec,
        module_name=module_name,
        generated_cpp_filename=cpp_path.name,
        template_path=bindings_template_path,
    )
    cmake_source_text = render_cmake_template(
        module_name=module_name,
        bindings_cpp_filename=bindings_cpp_path.name,
        template_path=cmake_template_path,
    )

    cpp_path.write_text(sim_source_text, encoding="utf-8")
    bindings_cpp_path.write_text(bindings_source_text, encoding="utf-8")
    cmake_lists_path.write_text(cmake_source_text, encoding="utf-8")
    runtime_header_path.write_text(
        (RUNTIME_DIR / "runtime.hpp").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return GeneratedSourceArtifact(
        module_name=module_name,
        build_dir=build_dir,
        cpp_path=cpp_path,
        bindings_cpp_path=bindings_cpp_path,
        cmake_lists_path=cmake_lists_path,
        runtime_header_path=runtime_header_path,
        sim_template_path=sim_template_path,
        bindings_template_path=bindings_template_path,
        cmake_template_path=cmake_template_path,
        sim_source_text=sim_source_text,
        bindings_source_text=bindings_source_text,
        cmake_source_text=cmake_source_text,
    )


def build_generated_extension(
    artifact: GeneratedSourceArtifact,
    cmake_build_type: str = "Release",
    verbose: bool = False,
) -> GeneratedSourceArtifact:
    try:
        return _build_generated_extension_with_cmake(
            artifact=artifact,
            cmake_build_type=cmake_build_type,
            verbose=verbose,
        )
    except subprocess.CalledProcessError:
        if verbose:
            print("[tvb-cpp] CMake build failed — falling back to direct compiler")
        return _build_generated_extension_with_compiler(artifact, verbose=verbose)


def _build_generated_extension_with_cmake(
    artifact: GeneratedSourceArtifact,
    cmake_build_type: str,
    verbose: bool = False,
) -> GeneratedSourceArtifact:
    cmake_build_dir = artifact.build_dir / "cmake-build"
    if cmake_build_dir.exists():
        shutil.rmtree(cmake_build_dir)
    cmake_build_dir.mkdir(parents=True, exist_ok=True)

    configure_cmd = [
        "cmake",
        "-S",
        str(artifact.build_dir),
        "-B",
        str(cmake_build_dir),
        f"-DCMAKE_BUILD_TYPE={cmake_build_type}",
        f"-DPython3_EXECUTABLE={sys.executable}",
        f"-DPYTHON_EXECUTABLE={sys.executable}",
    ]
    build_cmd = [
        "cmake",
        "--build",
        str(cmake_build_dir),
        "--config",
        cmake_build_type,
    ]

    subprocess_kwargs: dict = {"check": True, "cwd": artifact.build_dir, "text": True}
    if verbose:
        print(f"[tvb-cpp] CMake configure: {' '.join(configure_cmd)}")
    else:
        subprocess_kwargs["capture_output"] = True

    subprocess.run(configure_cmd, **subprocess_kwargs)

    if verbose:
        print(f"[tvb-cpp] CMake build:     {' '.join(build_cmd)}")

    subprocess.run(build_cmd, **subprocess_kwargs)

    extension_path = _discover_extension_path(artifact.build_dir, artifact.module_name)
    if verbose:
        print(f"[tvb-cpp] Built:           {extension_path}")
    return dataclasses.replace(artifact, extension_path=extension_path)


def _build_generated_extension_with_compiler(
    artifact: GeneratedSourceArtifact,
    verbose: bool = False,
) -> GeneratedSourceArtifact:
    include_flags = _resolve_compiler_include_flags()
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not ext_suffix:
        raise RuntimeError("Unable to determine Python extension suffix.")
    ldflags = _resolve_python_ldflags()

    extension_path = artifact.build_dir / f"{artifact.module_name}{ext_suffix}"
    compile_cmd = [
        "c++",
        "-O3",
        "-Wall",
        "-shared",
        "-std=c++17",
        "-fPIC",
        *include_flags,
        str(artifact.bindings_cpp_path),
        "-o",
        str(extension_path),
        *ldflags,
    ]
    if verbose:
        print(f"[tvb-cpp] Compiler: {' '.join(compile_cmd)}")
    try:
        subprocess.run(
            compile_cmd,
            check=True,
            cwd=artifact.build_dir,
            capture_output=not verbose,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        if detail:
            raise RuntimeError(f"Direct compiler build failed:\n{detail}") from exc
        raise
    if verbose:
        print(f"[tvb-cpp] Built:    {extension_path}")
    return dataclasses.replace(artifact, extension_path=extension_path)


def _resolve_compiler_include_flags() -> list[str]:
    flags: list[str] = []

    python_include = sysconfig.get_paths().get("include")
    plat_include = sysconfig.get_paths().get("platinclude")
    for include_dir in (python_include, plat_include):
        if include_dir and Path(include_dir).exists():
            flags.append(f"-I{include_dir}")

    pybind11_include = _resolve_pybind11_include_dir()
    flags.append(f"-I{pybind11_include}")
    return flags


def _resolve_pybind11_include_dir() -> str:
    env_include = os.environ.get("PYBIND11_INCLUDE_DIR")
    if env_include and Path(env_include).exists():
        return env_include

    spec = importlib.util.find_spec("pybind11")
    if spec and spec.origin:
        package_dir = Path(spec.origin).resolve().parent
        include_dir = package_dir / "include"
        if include_dir.exists():
            return str(include_dir)

    candidates = [
        Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "pybind11" / "include",
        Path.home() / "anaconda3" / "lib" / "python3.11" / "site-packages" / "pybind11" / "include",
        Path.home() / ".local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "pybind11" / "include",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    for root in (
        Path.home() / "anaconda3",
        Path.home() / ".local",
        Path.home() / "prog" / "spack",
    ):
        if not root.exists():
            continue
        matches = list(root.glob("**/pybind11/include"))
        if matches:
            return str(matches[0])

    raise RuntimeError(
        "Unable to locate pybind11 headers. Install pybind11 in the active "
        "environment or set PYBIND11_INCLUDE_DIR."
    )


def _resolve_python_ldflags() -> list[str]:
    config_dir = sysconfig.get_config_var("LIBPL")
    if config_dir:
        python_config = Path(config_dir).parent.parent / "bin" / "python3-config"
        if python_config.exists():
            try:
                output = subprocess.run(
                    [str(python_config), "--ldflags"],
                    check=True,
                    capture_output=True,
                    text=True,
                    cwd=".",
                ).stdout.strip()
                if output:
                    return shlex.split(output)
            except subprocess.CalledProcessError:
                pass

    ldflags: list[str] = []
    libdir = sysconfig.get_config_var("LIBDIR")
    if libdir:
        ldflags.append(f"-L{libdir}")

    version = sysconfig.get_config_var("VERSION")
    abiflags = sysconfig.get_config_var("ABIFLAGS") or ""
    if version:
        ldflags.append(f"-lpython{version}{abiflags}")

    for var_name in ("LIBS", "SYSLIBS", "LINKFORSHARED"):
        value = sysconfig.get_config_var(var_name)
        if value:
            ldflags.extend(shlex.split(value))
    return ldflags
