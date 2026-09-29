# -*- coding: utf-8 -*-
"""Permanent tests for build-time generated generic dfuns.

What the feature guarantees, and what each test below pins down:

* ``CMakeLists.txt`` runs ``dfungen --emit --strict`` and compiles the emitted
  TU into ``_cpp_hybrid`` (``CPH_HAVE_BUILTIN_GEN=1``), so the 26 stock
  generic-route models are served by kernels that ship inside the extension.
  A stock generic model therefore needs **no compiler at runtime** (a).
* The extension's ``generic_model_table()`` is the authority for those models:
  ids, buffer shapes and parameter packing order are the very table the
  kernels were emitted from, so Python and C++ cannot disagree (b).
* The runtime g++/ctypes fallback still exists for models the build did not
  know about, and it is emitted in an id range strictly **above** the built-in
  one, so a user model that sorts before every stock name can never take a
  built-in id and run the wrong kernel (c, d).
* With no built-in table at all the fallback keeps its original 100-based
  range, i.e. the pre-feature behaviour (e).
* Parameter packing is checked against the table the kernel came from and a
  mismatch fails loudly, naming the model (f).
* A fallback-compiled model still matches the numba reference (g).

Model discovery (``dfungen._collect_models``) scans ``tvb.simulator.models``
via ``pkgutil.iter_modules(pkg.__path__)``, so the "user-defined model" used
here is a real module dropped on a temporary entry of that ``__path__`` — the
same route a user's model package takes — rather than a monkeypatched dict.
"""

import importlib
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

import tvb.simulator.models as _models_pkg
from tvb.simulator.backend import cpp_hybrid as cph_pkg
from tvb.simulator.backend.cpp_hybrid import _cpp_hybrid, dfungen
from tvb.simulator.backend.cpp_hybrid import backend as cphb
from tvb.simulator.backend.cpp_hybrid.backend import CppHybridBackend
from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.integrators import HeunDeterministic

DT = 0.01
NSTEP = 20
NNODES = 3
#: first generic model id; also the id the runtime fallback falls back to when
#: the extension carries no built-in generic kernels
GENERIC_FIRST = dfungen._FIRST_MODEL_ID
#: a user model class name that sorts before every stock model name, i.e. the
#: worst case for positional id assignment over sorted class names
PROBE_NAME = "AaaProbe"
#: the models the runtime fallback library is built with in these tests: the
#: probe plus two stock generic models, so the fallback's own id assignment is
#: observable without compiling all 27 kernels (which costs ~8 s of g++)
RUNTIME_MODEL_NAMES = (PROBE_NAME, "CoombesByrne", "DumontGutkin")

_PROBE_SOURCE = '''
"""A user-defined model, for tests of the runtime dfun fallback."""

import numpy
from tvb.basic.neotraits.api import NArray, List, Range, Final
from tvb.simulator.models.base import Model


class AaaProbe(Model):
    """Two state variables, one coupling term, three global parameters.

    Expression-driven like the stock generic models: ``state_variable_dfuns``
    is what makes it a dfungen candidate.  ``dfun`` exists only because
    ``Model.dfun`` is abstract; the hybrid backends never call it.
    """

    tau = NArray(label="tau", default=numpy.array([1.0]),
                 domain=Range(lo=0.01, hi=10.0, step=0.01))
    a = NArray(label="a", default=numpy.array([2.0]),
               domain=Range(lo=-10.0, hi=10.0, step=0.01))
    b = NArray(label="b", default=numpy.array([3.0]),
               domain=Range(lo=-10.0, hi=10.0, step=0.01))

    state_variables = ["x", "y"]
    _nvar = 2
    cvar = numpy.array([0], dtype="int32")
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"x": numpy.array([-2.0, 2.0]), "y": numpy.array([-2.0, 2.0])},
    )
    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={"x": numpy.array([-numpy.inf, numpy.inf]),
                 "y": numpy.array([-numpy.inf, numpy.inf])},
    )
    coupling_terms = Final(label="Coupling terms", default=["cx"])
    state_variable_dfuns = Final(
        label="Drift functions",
        default={"x": "(-x + a * y + cx) / tau", "y": "b * x - y * y * y"},
    )
    variables_of_interest = List(
        of=str, label="VOI", choices=("x", "y"), default=("x", "y"))
    parameter_names = List(of=str, label="params", default="tau a b".split())

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        x, y = state_variables
        return numpy.array([(-x + y * self.a[0] + coupling[0]) / self.tau[0],
                            self.b[0] * x - y ** 3])
'''


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def emitted_stock():
    """``emit_sources()`` for the stock model set: (source_text, meta).

    Taken once, before any test installs the probe model on the discovery
    path, so it is the stock model set the build emitted.  The assertion
    below is what keeps a leaked probe from silently corrupting every test
    that compares against it.
    """
    src, meta = dfungen.emit_sources()
    assert PROBE_NAME not in meta, "probe model leaked into stock model discovery"
    return src, meta


@pytest.fixture(scope="module")
def probe_class(tmp_path_factory):
    """The user-defined model class, discoverable like a stock model.

    Its module lives in a temporary directory appended to
    ``tvb.simulator.models.__path__``, so ``dfungen``'s own
    ``pkgutil.iter_modules`` scan finds it — no discovery hook is faked.
    """
    pkg_dir = tmp_path_factory.mktemp("tvb_user_models")
    (pkg_dir / "aaa_probe.py").write_text(_PROBE_SOURCE, encoding="utf-8")
    _models_pkg.__path__.append(str(pkg_dir))
    try:
        module = importlib.import_module("tvb.simulator.models.aaa_probe")
        yield module.AaaProbe
    finally:
        _models_pkg.__path__.remove(str(pkg_dir))
        sys.modules.pop("tvb.simulator.models.aaa_probe", None)


@pytest.fixture
def isolated_backend(monkeypatch):
    """Per-test backend state: fresh runtime-library state and cache dir.

    The runtime dfun library and the built-in table are process-global
    (``backend._GEN_*`` / ``_BUILTIN_TABLE``), and the C++ core holds a raw
    function pointer to whichever library was last injected.  Every test that
    touches them runs here and the previous state — including that pointer —
    is put back afterwards.
    """
    import ctypes
    import tempfile
    from pathlib import Path

    cache_dir = Path(tempfile.mkdtemp(prefix="tvb_buildgen_cache_"))
    saved = (cphb._GEN_LIB, dict(cphb._GEN_IDS), dict(cphb._GEN_META),
             cphb._BUILTIN_TABLE, CppHybridBackend._CACHE_DIR,
             os.environ.get("TVB_NHYBRID_CACHE_DIR"))
    monkeypatch.setenv("TVB_NHYBRID_CACHE_DIR", str(cache_dir))
    CppHybridBackend._CACHE_DIR = None
    cphb._GEN_LIB, cphb._GEN_IDS, cphb._GEN_META = None, {}, {}
    try:
        yield cache_dir
    finally:
        cphb._GEN_LIB, cphb._GEN_IDS, cphb._GEN_META = (saved[0], saved[1],
                                                         saved[2])
        cphb._BUILTIN_TABLE = saved[3]
        CppHybridBackend._CACHE_DIR = saved[4]
        if saved[0] is not None:
            # the core still dispatches ids the built-in table does not cover
            # through this pointer; put back whatever was there before
            cph_pkg.enable_generic_dfuns(
                ctypes.cast(saved[0].cph_generic_dfun, ctypes.c_void_p).value)


@pytest.fixture
def small_runtime_model_set(monkeypatch, probe_class):
    """Limit which models the runtime fallback compiles (speed, not coverage).

    Discovery stays real: the probe class comes from the module installed by
    :func:`probe_class`, and the stock models are ordinary configured
    instances.  Only the *number* of kernels handed to g++ is reduced.
    """
    def _small(strict=False):
        out = {}
        for name in RUNTIME_MODEL_NAMES:
            inst = (probe_class() if name == PROBE_NAME else _stock_class(name)())
            inst.configure()
            out[name] = inst
        return out

    monkeypatch.setattr(dfungen, "collect_models", _small)
    return _small


_STOCK_CLASSES = None
#: bound at import so a test-local monkeypatch of ``collect_models`` (see
#: :func:`small_runtime_model_set`) can never shrink this cache
_REAL_COLLECT_MODELS = dfungen.collect_models


def _stock_classes():
    """Every stock model class, from dfungen's own discovery pass (cached).

    Cached as *classes* so each test gets a freshly configured instance and
    one test's parameter mutations cannot leak into another's run.
    """
    global _STOCK_CLASSES
    if _STOCK_CLASSES is None:
        _STOCK_CLASSES = {name: type(model)
                          for name, model in _REAL_COLLECT_MODELS().items()}
    return _STOCK_CLASSES


def _stock_class(name):
    return _stock_classes()[name]


def _stock_model(name):
    model = _stock_class(name)()
    model.configure()
    return model


def _network(models):
    subnets = []
    for i, model in enumerate(models):
        sn = Subnetwork(name=f"S{i}", model=model,
                        scheme=HeunDeterministic(dt=DT), nnodes=NNODES)
        sn.configure()
        subnets.append(sn)
    ns = NetworkSet(subnets=subnets, projections=[])
    ns.configure()
    return ns


def _run_cpp(model):
    """cpp_hybrid trajectory (nstep, n_svar, n_node, 1) for one subnet."""
    return np.asarray(CppHybridBackend().run_network(_network([model]),
                                                     NSTEP)[0][1])


# ---------------------------------------------------------------------------
# (a) stock generic models need no compiler at runtime
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name", ["CoombesByrne", "DumontGutkin"])
def test_stock_generic_model_needs_no_compiler(model_name, isolated_backend,
                                               monkeypatch):
    """A generic-route model runs with ``generate_lib`` unable to run.

    The proof is twofold: the runtime generator is never reached (it would
    raise), and the redirected cache dir ends up with no ``models_gen.*``
    file — nothing was emitted or compiled for this run.  PATH is left alone
    because the editable install rebuilds the extension on import and needs
    cmake/ninja; the built-in kernels are proven by the generator being
    unreachable, not by the compiler being absent.
    """
    calls = []

    def no_runtime_generation(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError(
            "dfungen.generate_lib must not run for a built-in generic model")

    monkeypatch.setattr(dfungen, "generate_lib", no_runtime_generation)

    builtin = cphb._builtin_model_table()
    assert model_name not in cphb._MODEL_IDS, "not a generic-route model"
    assert model_name in builtin, "extension carries no built-in kernel for it"

    out = CppHybridBackend().run_network(_network([_stock_model(model_name)]),
                                         NSTEP)
    data = np.asarray(out[0][1])
    # the monitor output spans the model's variables_of_interest, not every
    # state variable (CoombesByrne has 4 state variables, 2 of interest)
    n_voi = len(_stock_model(model_name).variables_of_interest)
    assert data.shape == (NSTEP, n_voi, NNODES, 1)
    assert int(builtin[model_name]["n_svar"]) >= n_voi
    assert np.all(np.isfinite(data))
    assert np.any(data != 0.0), "model did not simulate"

    assert calls == [], f"runtime generation was requested: {calls}"
    assert cphb._GEN_LIB is None, "runtime dfun library was loaded"
    generated = sorted(p.name for p in isolated_backend.glob("models_gen.*"))
    assert generated == [], f"runtime dfun artifacts in cache dir: {generated}"


# ---------------------------------------------------------------------------
# (b) the extension's table is the emitted metadata
# ---------------------------------------------------------------------------

def test_builtin_table_matches_emitted_metadata(emitted_stock):
    """``generic_model_table()`` == ``emit_sources()[1]``, ids 100.. contiguous.

    The table the extension exposes is read out of the compiled metadata
    table, which ``emit_sources`` builds from the same dict it returns to
    Python.  If either side drifts — an id reassigned, a parameter renamed, a
    count off by one — the backend would pack parameters for one kernel and
    dispatch another.
    """
    src, meta = emitted_stock
    table = _cpp_hybrid.generic_model_table()

    assert table, "extension carries no built-in generic model table"
    assert len(table) >= 26, (f"built-in coverage shrank: {len(table)} models "
                              f"(the stock set has 26)")
    assert set(table) == set(meta)
    assert len(table) == len(meta), (sorted(table), sorted(meta))

    ids = sorted(int(m["mid"]) for m in table.values())
    assert ids == list(range(GENERIC_FIRST, GENERIC_FIRST + len(ids))), ids

    for name, expected in sorted(meta.items()):
        got = table[name]
        for key in ("mid", "n_parm", "n_cvar", "n_svar"):
            assert int(got[key]) == int(expected[key]), (name, key, got[key],
                                                         expected[key])
        assert [str(p) for p in got["parm_names"]] == [
            str(p) for p in expected["parm_names"]], (name, got["parm_names"],
                                                      expected["parm_names"])
        # the kernel the id names is the kernel the metadata describes
        assert f"dfun_gen_{expected['mid']}" in src


def test_emitted_entry_symbol_matches_the_core_declaration():
    """The name the build emits under is the name ``_core.cpp`` declares.

    CMake runs ``dfungen --entry-name cph_builtin_dfun`` and ``_core.cpp``
    redeclares that symbol under ``CPH_HAVE_BUILTIN_GEN``.  The two are kept
    in step only by convention, so a rename on either side would link an
    extension whose generated kernels are never dispatched — a silent return
    to the runtime-compiler path.  This checks both sides of the seam.
    """
    build_src, _ = dfungen.emit_sources(models={},
                                        entry_name="cph_builtin_dfun")
    assert "int cph_builtin_dfun(int model_id," in build_src

    core_src = (Path(cphb.__file__).parent / "_core.cpp").read_text()
    assert "#ifdef CPH_HAVE_BUILTIN_GEN" in core_src
    assert "int cph_builtin_dfun(int model_id," in core_src
    assert "int cph_builtin_count(void);" in core_src
    assert "const cph_gen_entry *cph_builtin_entry(int idx);" in core_src

    with pytest.raises(ValueError, match="entry symbol"):
        dfungen.emit_sources(models={}, entry_name="not an identifier")


def test_backend_uses_the_builtin_table_as_authority(emitted_stock):
    """``_model_key`` returns the built-in id/shape for a stock generic model.

    Without this the backend would go through the runtime library for models
    the extension already carries — i.e. it would need a compiler again.
    """
    _, meta = emitted_stock
    table = _cpp_hybrid.generic_model_table()
    backend = CppHybridBackend()
    # only models without a hand-written kernel: those take precedence over
    # the built-in generic table (see _model_key's resolution order)
    generic_only = sorted(name for name in meta if name not in cphb._MODEL_IDS)
    assert len(generic_only) >= 4, generic_only
    for name in generic_only:
        model_id, n_parm, n_cvar, n_svar, parm = backend._model_key(
            _stock_model(name))
        assert model_id == int(table[name]["mid"])
        assert model_id == int(meta[name]["mid"])
        assert n_parm == len(meta[name]["parm_names"])
        assert n_cvar == int(meta[name]["n_cvar"])
        assert n_svar == int(meta[name]["n_svar"])
        assert list(parm) == list(meta[name]["parm_names"])
        assert model_id >= GENERIC_FIRST


def test_hand_written_kernels_take_precedence_over_the_builtin_table():
    """A model with both a hand-written kernel and a generated one keeps id 0..12.

    The generated table covers every model declaring ``state_variable_dfuns``,
    which includes several that also have hand-written kernels (JansenRit,
    WilsonCowan, Epileptor, ...).  Parity for those was established against
    the hand-written kernels, so they must keep resolving to ids 0..12 even
    though a generated kernel for them is compiled into the extension too.
    """
    table = _cpp_hybrid.generic_model_table()
    backend = CppHybridBackend()
    overlap = sorted(set(cphb._MODEL_IDS) & set(table))
    assert overlap, "expected generated kernels alongside the hand-written ones"
    for name in overlap:
        model_id = backend._model_key(_stock_model(name))[0]
        assert model_id == cphb._MODEL_IDS[name] < GENERIC_FIRST
        assert int(table[name]["mid"]) >= GENERIC_FIRST


# ---------------------------------------------------------------------------
# (c) user models get ids strictly above the built-in range
# ---------------------------------------------------------------------------

def test_user_model_ids_are_disjoint_from_builtin_range(probe_class,
                                                        isolated_backend,
                                                        small_runtime_model_set):
    """``AaaProbe`` sorts first yet cannot take a built-in id.

    Ids are positional over sorted class names, so a user model sorting
    before the stock ones is exactly the case that would shift every stock id
    down and dispatch the wrong kernel.  The fallback library is emitted at
    ``max(built-in) + 1``, so:
      * the probe lands above every built-in id,
      * the fallback's own copies of stock models also land above it,
      * stock models keep resolving to their built-in id, and their
        trajectory is bitwise unchanged by the user model being present.
    """
    builtin = cphb._builtin_model_table()
    max_builtin = max(int(m["mid"]) for m in builtin.values())
    assert max_builtin == GENERIC_FIRST + len(builtin) - 1

    # stock trajectory with no runtime library in the process at all
    before = _run_cpp(_stock_model("CoombesByrne"))
    assert cphb._GEN_LIB is None

    # running the user model is what triggers the runtime library
    probe_out = np.asarray(
        CppHybridBackend().run_network(_network([probe_class()]), NSTEP)[0][1])
    assert probe_out.shape == (NSTEP, 2, NNODES, 1)
    assert np.all(np.isfinite(probe_out))

    runtime_ids = dict(cphb._GEN_IDS)
    assert PROBE_NAME in runtime_ids
    # positional ids over sorted names would have given the probe the first
    # id of the range; the disjoint range is what prevents that
    assert sorted(runtime_ids)[0] == PROBE_NAME
    assert runtime_ids[PROBE_NAME] > max_builtin
    assert min(runtime_ids.values()) > max_builtin, runtime_ids
    for name in RUNTIME_MODEL_NAMES[1:]:
        # the fallback's copy of a stock model never overlaps its built-in id
        assert runtime_ids[name] > max_builtin
        assert cphb._GEN_META[name]["mid"] == runtime_ids[name]

    # stock models still resolve to the built-in table, not the fallback
    model_id, n_parm, _cvar, _svar, parm = CppHybridBackend()._model_key(
        _stock_model("CoombesByrne"))
    assert model_id == int(builtin["CoombesByrne"]["mid"])
    assert model_id < runtime_ids["CoombesByrne"]
    assert list(parm) == list(builtin["CoombesByrne"]["parm_names"])
    assert n_parm == len(parm)

    after = _run_cpp(_stock_model("CoombesByrne"))
    np.testing.assert_array_equal(after, before)


# ---------------------------------------------------------------------------
# (d) emit_sources(start_id=...) gap padding
# ---------------------------------------------------------------------------

def test_emit_sources_rejects_ids_below_the_generic_range():
    """``start_id < 100`` is refused, not silently emitted into dead code.

    Ids below 100 are the hand-written kernels' range in _core.cpp: a
    generated kernel there would never be dispatched, so the emission fails
    instead of producing a table nobody reads.
    """
    for bad in (0, 99, GENERIC_FIRST - 1):
        with pytest.raises(ValueError, match="start_id"):
            dfungen.emit_sources(models={}, start_id=bad)
    # the boundary itself is fine
    _src, meta = dfungen.emit_sources(models={}, start_id=GENERIC_FIRST)
    assert meta == {}


@pytest.mark.parametrize("start_id", [GENERIC_FIRST, 126, 130])
def test_emit_sources_gap_pads_the_dispatch_table(start_id,
                                                  small_runtime_model_set):
    """Gap slots pad the dispatch table; the metadata table stays compact.

    Asserted on the emitted text only — no compiler needed.  ``_dfuns`` is
    indexed by ``id - 100`` so it must cover the ids this TU does *not* own
    with ``nullptr`` (the entry returns 0 for them, letting the built-in
    table serve those ids).  ``_cph_gen`` holds one row per emitted model, so
    ``N_GEN_MODELS`` (metadata rows) and ``N_GEN_DFUNS`` (slots) differ
    exactly by the gap.
    """
    models = small_runtime_model_set()
    src, meta = dfungen.emit_sources(models, start_id=start_id)

    n_models = int(re.search(r"#define N_GEN_MODELS (\d+)", src).group(1))
    n_slots = int(re.search(r"#define N_GEN_DFUNS (\d+)", src).group(1))
    assert n_models == len(meta) == len(models)
    gap = start_id - GENERIC_FIRST
    assert n_slots == gap + n_models
    if gap:
        assert n_slots != n_models
    else:
        assert n_slots == n_models

    block = re.search(r"static dfun_fn_t _dfuns\[\] = \{\n(.*?)\n\};", src,
                      re.S).group(1)
    rows = [ln.strip().rstrip(",") for ln in block.splitlines() if ln.strip()]
    assert len(rows) == n_slots
    assert rows[:gap] == ["nullptr"] * gap
    assert rows[gap:] == [f"dfun_gen_{m['mid']}" for m in meta.values()]
    assert rows.count("nullptr") == gap

    # ids in the gap fall through instead of calling a null pointer
    assert "if (idx < 0 || idx >= N_GEN_DFUNS) return 0;" in src
    assert "if (fn == nullptr) return 0;" in src

    # metadata rows carry the ids the kernels were emitted under
    for name, m in meta.items():
        assert m["mid"] == start_id + sorted(models).index(name)
        row = ('{"%s", %d, %d, %d, %d, _cph_parm_names_%d},' % (
            name, m["mid"], m["n_parm"], m["n_cvar"], m["n_svar"], m["mid"]))
        assert row in src

    # the runtime entry symbol is the one the core injects a pointer for
    assert "int cph_generic_dfun(int model_id," in src


# ---------------------------------------------------------------------------
# (e) no built-in table keeps the original 100-based range
# ---------------------------------------------------------------------------

def test_runtime_range_starts_at_100_without_builtin_kernels(isolated_backend,
                                                            small_runtime_model_set,
                                                            monkeypatch):
    """An extension without built-in generic kernels keeps the old behaviour.

    ``TVB_CPP_GENERATE_MODELS=OFF`` (or an extension predating the table)
    leaves ``generic_model_table()`` empty; the fallback must then take ids
    from 100, exactly as it did before the feature.  Only the id range is
    asserted here: with kernels *actually* compiled into the extension the
    built-in table would serve ids 100.. first, so running a model allocated
    into that range by an emptied Python-side table would dispatch the stock
    kernel — the two views have to agree, and they do whenever the table is
    genuinely absent.
    """
    monkeypatch.setattr(cphb, "_BUILTIN_TABLE", {})
    assert cphb._builtin_model_table() == {}

    CppHybridBackend()._ensure_generic_dfuns()

    runtime_ids = dict(cphb._GEN_IDS)
    assert runtime_ids, "no runtime model ids were allocated"
    assert min(runtime_ids.values()) == GENERIC_FIRST
    assert sorted(runtime_ids.values()) == list(
        GENERIC_FIRST + np.arange(len(runtime_ids)))
    assert runtime_ids[PROBE_NAME] == GENERIC_FIRST

    # the library really was emitted + compiled into this run's cache dir,
    # with no gap padding because the range starts at the base id
    src = (isolated_backend / "models_gen.cpp").read_text()
    so = isolated_backend / "models_gen.so"
    assert so.exists()
    n_models = int(re.search(r"#define N_GEN_MODELS (\d+)", src).group(1))
    n_slots = int(re.search(r"#define N_GEN_DFUNS (\d+)", src).group(1))
    assert n_slots == n_models == len(runtime_ids)
    block = re.search(r"static dfun_fn_t _dfuns\[\] = \{\n(.*?)\n\};", src,
                      re.S).group(1)
    assert "nullptr" not in block


# ---------------------------------------------------------------------------
# (f) parameter packing mismatches name the model
# ---------------------------------------------------------------------------

def _coombesbyrne_with_params(names):
    from tvb.simulator.models.infinite_theta import CoombesByrne
    model = CoombesByrne()
    model.configure()
    extra = np.array([1.5], dtype=np.float64)
    for name in names:
        if not hasattr(model, name):
            setattr(model, name, extra)
    model.parameter_names = list(names)
    return model


@pytest.mark.parametrize("mutate, reason", [
    pytest.param(lambda p: list(p) + ["bogus"], "parameter(s)",
                 id="extra-parameter"),
    pytest.param(lambda p: ["etax" if n == "eta" else n for n in p],
                 "packs", id="renamed-parameter"),
])
def test_parameter_packing_mismatch_names_the_model(mutate, reason):
    """A model whose parameters disagree with its kernel fails, naming both.

    ``parr`` is indexed positionally, so a count mismatch reads past the end
    of the packed array and a name mismatch lands values in the wrong slots —
    both produce wrong dynamics rather than an error, unless the table is
    consulted.  The failure must name the model and show both lists.
    """
    table = cphb._builtin_model_table()["CoombesByrne"]
    expected = [str(p) for p in table["parm_names"]]
    model = _coombesbyrne_with_params(mutate(expected))

    with pytest.raises(ValueError) as excinfo:
        CppHybridBackend()._model_key(model)

    message = str(excinfo.value)
    assert "CoombesByrne" in message
    assert reason in message
    assert "bogus" in message or "etax" in message
    for name in expected:
        assert name in message


def test_matching_parameters_are_accepted():
    """The same check passes for an unmodified stock model."""
    table = cphb._builtin_model_table()["CoombesByrne"]
    model = _coombesbyrne_with_params([str(p) for p in table["parm_names"]])
    model_id, n_parm, _cvar, _svar, parm = CppHybridBackend()._model_key(model)
    assert model_id == int(table["mid"])
    assert list(parm) == [str(p) for p in table["parm_names"]]
    assert n_parm == len(parm)


# ---------------------------------------------------------------------------
# (g) the runtime fallback still matches the numba reference
# ---------------------------------------------------------------------------

def test_runtime_fallback_model_matches_numba(probe_class, isolated_backend,
                                              small_runtime_model_set,
                                              monkeypatch):
    """A fallback-compiled model matches NbHybridBackend at float32 tolerance.

    The fallback library is compiled without ``-ffp-contract=off`` (the
    runtime g++ command does not pass it), so this is a tolerance comparison,
    not a bitwise one.  nb_hybrid keeps a fixed supported-model list, so the
    probe is added to that list for the duration of the test.
    """
    import tvb.simulator.backend.nb_hybrid as nbh
    nbh._get_supported_models_classes()
    monkeypatch.setattr(nbh, "_SUPPORTED_MODELS_CACHE",
                        tuple(nbh._SUPPORTED_MODELS_CACHE) + (probe_class,))

    model = probe_class()
    model.configure()
    cpp_data = _run_cpp(model)
    nb_data = np.asarray(
        NbHybridBackend().compile(_network([model]), eager=True).run(NSTEP)[0][1])

    assert cpp_data.shape == nb_data.shape
    assert cphb._GEN_LIB is not None, "the runtime library was not compiled"
    assert PROBE_NAME in cphb._GEN_IDS
    np.testing.assert_allclose(cpp_data, nb_data, rtol=1e-3, atol=1e-4,
                               err_msg="runtime-compiled dfun diverged from "
                                       "the numba reference")
