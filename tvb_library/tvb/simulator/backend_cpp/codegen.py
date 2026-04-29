from __future__ import annotations

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

from .spec import SimulationSpec


TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
RUNTIME_DIR = Path(__file__).resolve().parent / "runtime"
DEFAULT_SIM_TEMPLATE = TEMPLATES_DIR / "sim_module.cpp.mako"
DEFAULT_BINDINGS_TEMPLATE = TEMPLATES_DIR / "module_bindings.cpp.mako"
DEFAULT_CMAKE_TEMPLATE = TEMPLATES_DIR / "CMakeLists.txt.mako"


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
    sim_source_text: str
    bindings_source_text: str
    cmake_source_text: str
    extension_path: Path | None = None


@dataclasses.dataclass(frozen=True)
class DelayedSelfFeedbackConfig:
    delay_steps: int
    gain: float
    source_state_var: str = "r"
    target_state_var: str = "V"


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


def _single_subnet(spec: SimulationSpec):
    if len(spec.subnetworks) != 1:
        raise NotImplementedError(
            "Native run_simulation is currently implemented only for single-subnetwork specs."
        )
    if spec.inter_projections or spec.intra_projections:
        raise NotImplementedError(
            "Native run_simulation is currently implemented only for specs without projections."
        )
    subnet = spec.subnetworks[0]
    if subnet.model_type != "MontbrioPazoRoxin":
        raise NotImplementedError(
            "Native run_simulation is currently implemented only for MontbrioPazoRoxin."
        )
    if subnet.integrator.type_name != "HeunDeterministic":
        raise NotImplementedError(
            "Native run_simulation is currently implemented only for HeunDeterministic."
        )
    return subnet


def render_cpp_template(
    spec: SimulationSpec,
    module_name: str,
    delayed_self_feedback: DelayedSelfFeedbackConfig | None = None,
    template_path: Path = DEFAULT_SIM_TEMPLATE,
) -> str:
    subnet = _single_subnet(spec)
    voi_index_map = {name: idx for idx, name in enumerate(subnet.state_variables)}
    voi_indices = [voi_index_map[name] for name in subnet.variables_of_interest]

    delayed_enabled = delayed_self_feedback is not None
    delayed_source_svar = 0
    delayed_target_svar = 1
    delayed_gain = 0.0
    delayed_steps = 0
    source_history_horizon = int(spec.source_horizons.get(subnet.name, 1))
    if delayed_self_feedback is not None:
        state_index_map = {name: idx for idx, name in enumerate(subnet.state_variables)}
        delayed_source_svar = state_index_map[delayed_self_feedback.source_state_var]
        delayed_target_svar = state_index_map[delayed_self_feedback.target_state_var]
        delayed_gain = float(delayed_self_feedback.gain)
        delayed_steps = int(delayed_self_feedback.delay_steps)
        source_history_horizon = max(source_history_horizon, delayed_steps + 1)

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
        "subnet": subnet,
        "voi_indices": voi_indices,
        "source_history_horizon": source_history_horizon,
        "delayed_enabled": delayed_enabled,
        "delayed_steps": delayed_steps,
        "delayed_gain": delayed_gain,
        "delayed_source_svar": delayed_source_svar,
        "delayed_target_svar": delayed_target_svar,
        "subnetwork_summary": _format_subnetwork_summary(spec),
        "projection_summary": _format_projection_summary(spec),
    }
    return _render_mako_template(template_path, ctx)


def render_bindings_template(
    module_name: str,
    generated_cpp_filename: str,
    template_path: Path = DEFAULT_BINDINGS_TEMPLATE,
) -> str:
    ctx = {
        "module_name": module_name,
        "generated_cpp_filename": generated_cpp_filename,
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
    delayed_self_feedback: DelayedSelfFeedbackConfig | None = None,
    sim_template_path: Path = DEFAULT_SIM_TEMPLATE,
    bindings_template_path: Path = DEFAULT_BINDINGS_TEMPLATE,
    cmake_template_path: Path = DEFAULT_CMAKE_TEMPLATE,
) -> GeneratedSourceArtifact:
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    cpp_path = build_dir / f"{module_name}.cpp"
    bindings_cpp_path = build_dir / f"{module_name}_bindings.cpp"
    cmake_lists_path = build_dir / "CMakeLists.txt"
    runtime_dir = build_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_header_path = runtime_dir / "runtime.hpp"

    sim_source_text = render_cpp_template(
        spec=spec,
        module_name=module_name,
        delayed_self_feedback=delayed_self_feedback,
        template_path=sim_template_path,
    )
    bindings_source_text = render_bindings_template(
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
) -> GeneratedSourceArtifact:
    try:
        return _build_generated_extension_with_cmake(
            artifact=artifact,
            cmake_build_type=cmake_build_type,
        )
    except subprocess.CalledProcessError:
        return _build_generated_extension_with_compiler(artifact)


def _build_generated_extension_with_cmake(
    artifact: GeneratedSourceArtifact,
    cmake_build_type: str,
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

    subprocess.run(
        configure_cmd,
        check=True,
        cwd=artifact.build_dir,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        build_cmd,
        check=True,
        cwd=artifact.build_dir,
        capture_output=True,
        text=True,
    )
    extension_path = _discover_extension_path(artifact.build_dir, artifact.module_name)
    return dataclasses.replace(artifact, extension_path=extension_path)


def _build_generated_extension_with_compiler(
    artifact: GeneratedSourceArtifact,
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
    subprocess.run(compile_cmd, check=True, cwd=artifact.build_dir)
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
