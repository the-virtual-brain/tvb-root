# -*- coding: utf-8 -*-
"""Permanent tests for build-time generated generic dfuns.

What the feature guarantees, and what each test below pins down:

* ``CMakeLists.txt`` runs ``dfungen --emit --strict --only-supported`` and
  compiles the emitted TU into ``_cpp_hybrid`` (``CPH_HAVE_BUILTIN_GEN=1``), so
  the 25 supported stock generic-route models are served by kernels that ship
  inside the extension.  A stock generic model therefore needs **no compiler
  at runtime** (a).
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
The build-time path (``only_supported=True`` / ``--only-supported``, handoff
item 6 option 2) additionally scopes that scan to the supported model set
(``dfungen._SUPPORTED_MODEL_CLASSES``/``_SUPPORTED_MODEL_MODULES``, mirroring
``nb_hybrid._get_supported_models_classes``): unsupported modules are not
imported, unsupported dfun classes (``DecoBalancedExcInh``) are not emitted,
and strict failure expectations apply to the supported set only.  The runtime
fallback keeps the full scan, which is why the probe module stays
discoverable.
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
    path, so it is the stock model set the build emitted.  Scoped to the
    supported model set (``emit_sources(only_supported=True)``), exactly like
    the CMake build invokes ``--only-supported``.  The assertion below is what
    keeps a leaked probe from silently corrupting every test that compares
    against it.
    """
    src, meta = dfungen.emit_sources(only_supported=True)
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
    assert len(table) >= 25, (f"built-in coverage shrank: {len(table)} models "
                              f"(the supported stock set has 25)")
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
        # the dfun fingerprint travels through the same table
        assert str(got["signature"]) == str(expected["signature"]), name
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

    # the struct the build emits and the one _core.cpp redeclares must carry
    # the same fields (they are the same struct across the TU boundary); the
    # signature field is where the dfun fingerprint rides through to
    # generic_model_table()
    assert "const char *signature;" in build_src
    assert "const char *signature;" in core_src

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
    ``max(built-in) + 1`` and *excludes* the built-in table's names, so:
      * the probe lands strictly above every built-in id,
      * no stock model is re-emitted into the runtime library at all,
      * stock models keep resolving to their built-in id, and their
        trajectory is bitwise unchanged by the user model being present.
    """
    builtin = cphb._builtin_model_table()
    max_builtin = max(int(m["mid"]) for m in builtin.values())
    assert max_builtin == GENERIC_FIRST + len(builtin) - 1

    # stock trajectory with no runtime library in the process at all
    before = _run_cpp(_stock_model("CoombesByrne"))
    assert cphb._GEN_LIB is None

    # running the user model is what triggers the runtime library; it is the
    # only model the library carries (stock names are excluded)
    probe_out = np.asarray(
        CppHybridBackend().run_network(_network([probe_class()]), NSTEP)[0][1])
    assert probe_out.shape == (NSTEP, 2, NNODES, 1)
    assert np.all(np.isfinite(probe_out))

    runtime_ids = dict(cphb._GEN_IDS)
    assert set(runtime_ids) == {PROBE_NAME}, runtime_ids
    assert runtime_ids[PROBE_NAME] > max_builtin
    assert min(runtime_ids.values()) > max_builtin, runtime_ids

    # stock models still resolve to the built-in table, not the fallback
    model_id, n_parm, _cvar, _svar, parm = CppHybridBackend()._model_key(
        _stock_model("CoombesByrne"))
    assert model_id == int(builtin["CoombesByrne"]["mid"])
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
        assert re.fullmatch(r"[0-9a-f]{64}", str(m["signature"]))
        row = ('{"%s", %d, %d, %d, %d, _cph_parm_names_%d, "%s"},' % (
            name, m["mid"], m["n_parm"], m["n_cvar"], m["n_svar"],
            m["mid"], m["signature"]))
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


# ---------------------------------------------------------------------------
# (h) dfun fingerprint guard (class-name shadowing hole)
# ---------------------------------------------------------------------------
# Model lookup is by class ``__name__``; parameter packing is checked against
# the kernel's table, but only for *shapes*.  A class named ``CoombesByrne``
# with the same parameters and different equations therefore used to resolve
# to the stock built-in id and silently run the stock kernel.  The fingerprint
# closes that hole: ``model_dfun_signature`` is computed identically at
# emission (stored in meta, carried in the emitted table row, exposed by
# ``generic_model_table()``) and at validation (``backend._model_key``).

_SIG_RE = re.compile(r"[0-9a-f]{64}")

#: a user class that shadows a stock class name with *different* equations
#: (same state variables, coupling term, intermediates and parameters)
_SHADOW_SOURCE = '''
"""A class shadowing the stock CoombesByrne name with altered equations."""

from tvb.basic.neotraits.api import Final
from tvb.simulator.models.infinite_theta import CoombesByrne as _Real


class CoombesByrne(_Real):
    """Same parameter shape as the stock model, different dynamics."""

    state_variable_dfuns = Final(
        label="Drift functions (shadowed)",
        default={
            "r": "Delta / math.pi + 2 * V * r - g * r + 0.5 * eta",
            "V": "V**2 - math.pi**2 * r**2 + eta + (v_syn - V) * g "
                  "+ Coupling_Term_r",
            "g": "alpha * q",
            "q": "alpha * (k * math.pi * r - g - 2 * q)",
        },
    )
'''

#: the same class name with byte-identical equations: must be accepted
_IDENTICAL_SOURCE = '''
"""A class shadowing the stock CoombesByrne name with identical equations."""

from tvb.basic.neotraits.api import Final
from tvb.simulator.models.infinite_theta import CoombesByrne as _Real


class CoombesByrne(_Real):
    """Bit-for-bit the stock expressions, re-declared under the same name."""

    state_variable_dfuns = Final(
        label="Drift functions (redeclared)",
        default={
            "r": "Delta / math.pi + 2 * V * r - g * r",
            "V": "V**2 - math.pi**2 * r**2 + eta + (v_syn - V) * g "
                  "+ Coupling_Term_r",
            "g": "alpha * q",
            "q": "alpha * (k * math.pi * r - g - 2 * q)",
        },
    )
'''


@pytest.fixture(scope="module")
def shadow_pkg(tmp_path_factory):
    """Modules whose ``CoombesByrne`` shadows the stock name.

    Two variants: ``shadow_coombes`` (altered equations) and
    ``identical_coombes`` (the stock equations re-declared).  Discovery never
    picks either up for emission - ``_collect_models`` keeps the first class
    it saw for a name, and the stock ``infinite_theta`` module is scanned
    before this temp dir - so the stock table stays authoritative; only
    ``_model_key`` resolves them, by name, through the built-in table.
    """
    pkg_dir = tmp_path_factory.mktemp("tvb_shadow_models")
    (pkg_dir / "shadow_coombes.py").write_text(_SHADOW_SOURCE, encoding="utf-8")
    (pkg_dir / "identical_coombes.py").write_text(
        _IDENTICAL_SOURCE, encoding="utf-8")
    _models_pkg.__path__.append(str(pkg_dir))
    try:
        module = importlib.import_module("tvb.simulator.models.shadow_coombes")
        identical = importlib.import_module(
            "tvb.simulator.models.identical_coombes")
        yield type("ShadowPkg", (), {
            "altered": module.CoombesByrne,
            "identical": identical.CoombesByrne,
        })()
    finally:
        _models_pkg.__path__.remove(str(pkg_dir))
        sys.modules.pop("tvb.simulator.models.shadow_coombes", None)
        sys.modules.pop("tvb.simulator.models.identical_coombes", None)


def test_builtin_table_signatures_are_64_hex(emitted_stock):
    """Every ``generic_model_table()`` entry carries a 64-hex signature.

    The signature is stored in meta at emission, emitted into the table row
    (the struct field ``_core.cpp`` redeclares) and read back out of the
    extension.  A missing field would mean the emitted TU and the extension
    disagree about the struct layout.
    """
    src, meta = emitted_stock
    table = _cpp_hybrid.generic_model_table()
    assert table, "extension carries no built-in generic model table"
    for name, rec in table.items():
        sig = str(rec.get("signature") or "")
        assert _SIG_RE.fullmatch(sig), (name, sig)
        assert sig == str(meta[name]["signature"]), name
    # the emitted TU really carries the field in its metadata rows
    assert "const char *signature;" in src


def test_builtin_table_signatures_match_model_dfun_signature(emitted_stock):
    """Emission- and validation-side fingerprints agree for every built-in.

    ``model_dfun_signature`` is the single canonical function; the meta dict
    and the extension table were both produced by it at emission time, so
    recomputing it on a fresh instance of the same class must reproduce the
    stored value for all 25 supported stock generic models (ids 100..124) -
    otherwise the guard would reject stock models.
    """
    _, meta = emitted_stock
    table = _cpp_hybrid.generic_model_table()
    ids = sorted(int(m["mid"]) for m in meta.values())
    assert ids == list(range(GENERIC_FIRST, GENERIC_FIRST + len(ids))), ids
    for name, m in sorted(meta.items()):
        got = dfungen.model_dfun_signature(_stock_model(name))
        assert _SIG_RE.fullmatch(got), name
        assert got == str(m["signature"]) == str(table[name]["signature"]), name


def test_model_dfun_signature_is_stable_and_sensitive(shadow_pkg):
    """Recomputation is stable; one changed expression flips the fingerprint.

    Same class instantiated twice -> same signature; a class re-declaring the
    stock equations under the stock name -> same as the real model; the
    altered variant -> different, even though its parameters, state variables
    and coupling terms are identical.
    """
    real = _stock_model("CoombesByrne")
    s_real = dfungen.model_dfun_signature(real)
    assert dfungen.model_dfun_signature(real) == s_real

    altered = shadow_pkg.altered()
    altered.configure()
    identical = shadow_pkg.identical()
    identical.configure()

    assert type(altered).__name__ == "CoombesByrne"
    assert type(identical).__name__ == "CoombesByrne"
    # same parameter shape as the stock kernel table
    table = cphb._builtin_model_table()["CoombesByrne"]
    parm = CppHybridBackend()._packing_parm_names(altered, table)
    assert CppHybridBackend()._packing_parm_names(
        identical, table) == parm == [str(p) for p in table["parm_names"]]

    s_altered = dfungen.model_dfun_signature(altered)
    s_identical = dfungen.model_dfun_signature(identical)
    assert len(s_real) == len(s_altered) == 64
    assert s_identical == s_real
    assert s_altered != s_real


def test_shadowing_class_with_different_equations_raises(shadow_pkg):
    """A same-named class with different equations fails, naming the model.

    This is the hole the fingerprint guard closes: before it, the altered
    class resolved to the stock built-in id because its parameter shape
    matched and it silently ran the stock kernel.
    """
    model = shadow_pkg.altered()
    model.configure()

    with pytest.raises(ValueError) as excinfo:
        CppHybridBackend()._model_key(model)

    message = str(excinfo.value)
    assert "CoombesByrne" in message
    assert "different equations" in message
    assert re.search(r"[0-9a-f]{64} != [0-9a-f]{64}", message)


def test_identical_class_is_accepted(shadow_pkg):
    """The same class name with the stock equations still resolves normally.

    A faithful re-declaration (a subclass overriding ``state_variable_dfuns``
    with byte-identical values) has the same fingerprint as the kernel it
    resolves to, so the guard must let it through to the built-in id and
    packing order.
    """
    model = shadow_pkg.identical()
    model.configure()

    table = cphb._builtin_model_table()["CoombesByrne"]
    model_id, n_parm, n_cvar, n_svar, parm = CppHybridBackend()._model_key(
        model)
    assert model_id == int(table["mid"]) >= GENERIC_FIRST
    assert n_parm == len(table["parm_names"])
    assert n_cvar == int(table["n_cvar"])
    assert n_svar == int(table["n_svar"])
    assert list(parm) == [str(p) for p in table["parm_names"]]


# ---------------------------------------------------------------------------
# (i) the runtime library excludes built-in models (no dead stock kernels)
# ---------------------------------------------------------------------------

def test_runtime_library_has_no_builtin_kernels(probe_class, isolated_backend,
                                                small_runtime_model_set):
    """A user model triggers the runtime library; stock models stay out.

    Before the exclusion filter, the first user model re-emitted every stock
    generic model into the runtime library too (dead kernels above the
    built-in ids, ~1 s of extra g++ per compile).  Now
    ``_ensure_generic_dfuns`` passes the built-in table's names as
    ``exclude``, so the emitted source carries exactly the user model's
    kernel and gap-pads the built-in range with ``nullptr`` — the built-in
    ids fall through to the prebuilt extension's table.
    """
    builtin = cphb._builtin_model_table()
    assert len(builtin) >= 25
    builtin_ids = sorted(int(m["mid"]) for m in builtin.values())
    assert builtin_ids == list(range(GENERIC_FIRST, GENERIC_FIRST +
                                     len(builtin_ids)))

    CppHybridBackend()._ensure_generic_dfuns()

    runtime_ids = dict(cphb._GEN_IDS)
    assert set(runtime_ids) == {PROBE_NAME}, runtime_ids

    src = (isolated_backend / "models_gen.cpp").read_text()
    so = isolated_backend / "models_gen.so"
    assert so.exists(), "runtime library was not compiled"

    # no dead stock kernels: no dfun_gen_<built-in mid> function at all
    for mid in builtin_ids:
        assert f"dfun_gen_{mid}" not in src, \
            f"built-in kernel {mid} re-emitted into the runtime library"
    # the only kernel in the library is the user model's, at max(builtin)+1
    probe_id = runtime_ids[PROBE_NAME]
    assert probe_id == max(builtin_ids) + 1
    assert f"dfun_gen_{probe_id}" in src
    assert src.count("static void dfun_gen_") == 1

    # N_GEN_MODELS counts kernels, N_GEN_DFUNS counts slots (built-in gap
    # padding included); the two must not drift apart
    n_models = int(re.search(r"#define N_GEN_MODELS (\d+)", src).group(1))
    n_slots = int(re.search(r"#define N_GEN_DFUNS (\d+)", src).group(1))
    gap = probe_id - GENERIC_FIRST
    assert n_models == 1
    assert n_slots == gap + n_models
    block = re.search(r"static dfun_fn_t _dfuns\[\] = \{\n(.*?)\n\};", src,
                      re.S).group(1)
    rows = [ln.strip().rstrip(",") for ln in block.splitlines() if ln.strip()]
    assert len(rows) == n_slots
    assert rows[:gap] == ["nullptr"] * gap
    assert rows[gap:] == [f"dfun_gen_{probe_id}"]

    # stock models still dispatch through the built-in table
    model_id, n_parm, _cvar, _svar, parm = CppHybridBackend()._model_key(
        _stock_model("CoombesByrne"))
    assert model_id == int(builtin["CoombesByrne"]["mid"])
    assert list(parm) == [str(p) for p in
                          builtin["CoombesByrne"]["parm_names"]]


def test_generate_lib_exclude_filters_discovered_models(isolated_backend,
                                                        small_runtime_model_set):
    """``generate_lib(exclude=...)`` drops the named models before emission.

    Discovery still returns the full small set (probe + two stock models);
    the exclusion filter removes the named ones, leaving only the probe.
    Ids are contiguous from ``start_id`` over the survivors, the dispatch
    table gap-pads the front range with ``nullptr``, and neither the kernel
    nor the metadata of an excluded model appears anywhere in the emitted
    source — N_GEN_MODELS (kernels) vs N_GEN_DFUNS (slots) stay correct
    across the gap.
    """
    models = small_runtime_model_set()
    assert set(models) == set(RUNTIME_MODEL_NAMES)

    start_id = GENERIC_FIRST + 10
    so_path, ids, meta = dfungen.generate_lib(
        isolated_backend, start_id=start_id,
        exclude=set(RUNTIME_MODEL_NAMES[1:]))

    assert set(ids) == {PROBE_NAME}, ids
    assert so_path.exists() and so_path.name == "models_gen.so"
    assert ids[PROBE_NAME] == start_id
    assert meta[PROBE_NAME]["mid"] == start_id

    src = (isolated_backend / "models_gen.cpp").read_text()
    for name in RUNTIME_MODEL_NAMES[1:]:
        assert name not in ids
        assert f'"{name}"' not in src, f"excluded model {name} still emitted"
        assert dfungen.model_dfun_signature(models[name]) not in src
    assert f"dfun_gen_{start_id}" in src

    n_models = int(re.search(r"#define N_GEN_MODELS (\d+)", src).group(1))
    n_slots = int(re.search(r"#define N_GEN_DFUNS (\d+)", src).group(1))
    assert n_models == 1
    assert n_slots == (start_id - GENERIC_FIRST) + n_models
    block = re.search(r"static dfun_fn_t _dfuns\[\] = \{\n(.*?)\n\};", src,
                      re.S).group(1)
    rows = [ln.strip().rstrip(",") for ln in block.splitlines() if ln.strip()]
    assert len(rows) == n_slots
    assert rows[:start_id - GENERIC_FIRST] == ["nullptr"] * (
        start_id - GENERIC_FIRST)
    assert rows[start_id - GENERIC_FIRST:] == [f"dfun_gen_{start_id}"]


# ---------------------------------------------------------------------------
# (j) supported-set scoping (handoff item 6, option 2)
# ---------------------------------------------------------------------------
# The build emits only the models the hybrid backends actually support (the
# 27 classes of nb_hybrid._get_supported_models_classes) -- CMake passes
# ``--only-supported`` -- so the built-in table is exactly the documented
# support list and unsupported model code in the tree cannot break or slow
# the build.  The runtime fallback keeps full discovery, so a dfun model
# outside the supported set still simulates (compiled on first use).

def test_supported_only_scopes_emission_to_the_supported_set():
    """``only_supported=True`` emits exactly the supported dfun models.

    Full discovery (the runtime path) still finds every in-tree dfun model,
    including ones outside the supported set (``DecoBalancedExcInh``); scoped
    discovery (the build path) emits only the supported expression-driven
    classes.  The two numba-only supported models (CerebellarMF,
    ZerlautAdaptationFirstOrder) declare no ``state_variable_dfuns`` and are
    not generic candidates, so they are absent from both.
    """
    _full_src, full = dfungen.emit_sources()
    _scoped_src, scoped = dfungen.emit_sources(only_supported=True)

    assert "DecoBalancedExcInh" in full, full.keys()
    assert "DecoBalancedExcInh" not in scoped
    # everything scoped emission keeps is part of the supported set
    assert set(scoped) <= dfungen._SUPPORTED_MODEL_CLASSES
    # everything full discovery finds beyond the scoped set is unsupported
    assert (set(full) - set(scoped)).isdisjoint(
        dfungen._SUPPORTED_MODEL_CLASSES)

    from tvb.simulator.backend.nb_hybrid import _get_supported_models_classes
    supported = _get_supported_models_classes()
    expected = {c.__name__ for c in supported
                if dfungen._declares_dfuns(c)}
    assert set(scoped) == expected


def test_supported_set_mirrors_nb_hybrid():
    """dfungen's scoping constants equal nb_hybrid's support list.

    The build-time constants must not drift from the runtime support list: one
    extra class would silently widen the built-in table, one missing class
    would silently drop a kernel the backend documents as supported.
    """
    from tvb.simulator.backend.nb_hybrid import _get_supported_models_classes
    classes = _get_supported_models_classes()
    assert dfungen._SUPPORTED_MODEL_CLASSES == {c.__name__ for c in classes}
    assert dfungen._SUPPORTED_MODEL_MODULES == {
        c.__module__.split(".")[-1] for c in classes}


def test_supported_only_strict_mode_stays_loud_but_scoped(monkeypatch):
    """Strictness is scoped to the supported set, and stays loud within it.

    A module outside the supported set must be irrelevant to a strict
    supported-only emission: exclude it from the scoped module set, break its
    import, and the build still succeeds (that model is simply not emitted).
    The same breakage in a supported module must fail the build loudly,
    naming the module -- the guard rail of the handoff item.
    """
    import sys

    scoped_modules = frozenset(dfungen._SUPPORTED_MODEL_MODULES)
    poison = "tvb.simulator.models.linear"
    monkeypatch.setitem(sys.modules, poison, None)  # import now raises

    # 'linear' declared out of scope: the poison is invisible to the build
    monkeypatch.setattr(dfungen, "_SUPPORTED_MODEL_MODULES",
                        scoped_modules - {"linear"})
    _src, meta = dfungen.emit_sources(only_supported=True, strict=True)
    assert "Linear" not in meta

    # 'linear' back in scope: the same poison is a loud build failure
    monkeypatch.setattr(dfungen, "_SUPPORTED_MODEL_MODULES", scoped_modules)
    with pytest.raises(dfungen.DfunGenerationError) as excinfo:
        dfungen.emit_sources(only_supported=True, strict=True)
    assert poison in str(excinfo.value)


def test_unsupported_stock_model_still_runs_via_runtime_fallback(
        isolated_backend):
    """A dfun model outside the supported set still simulates.

    Option-2 scoping removes non-supported models from the built-in table;
    support itself is unchanged because the runtime fallback keeps full
    discovery.  DecoBalancedExcInh -- an in-tree dfun model that is not one
    of the supported 27 -- must run through the runtime-compiled library.
    """
    table = cphb._builtin_model_table()
    assert "DecoBalancedExcInh" not in table
    assert "DecoBalancedExcInh" not in cphb._MODEL_IDS

    model = _stock_model("DecoBalancedExcInh")
    n_voi = len(model.variables_of_interest)
    out = CppHybridBackend().run_network(_network([model]), NSTEP)
    data = np.asarray(out[0][1])
    assert data.shape == (NSTEP, n_voi, NNODES, 1)
    assert np.all(np.isfinite(data))
    assert np.any(data != 0.0), "model did not simulate"
    assert "DecoBalancedExcInh" in cphb._GEN_IDS
    assert (isolated_backend / "models_gen.so").exists(), \
        "runtime library was not compiled"
