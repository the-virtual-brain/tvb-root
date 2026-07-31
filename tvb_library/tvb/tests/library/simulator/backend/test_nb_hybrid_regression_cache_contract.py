# -*- coding: utf-8 -*-
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package.
#
# (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

"""Regression contracts for the hybrid Numba generated-module cache."""

import sys

import pytest


_CACHE_ENV = "TVB_NHYBRID_CACHE_DIR"


@pytest.fixture
def isolated_generated_modules():
    from tvb.simulator.backend import nb_hybrid

    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name.startswith("nbhybrid_")
    }
    original_cache = dict(nb_hybrid._COMPILED_MOD_CACHE)
    for name in original_modules:
        del sys.modules[name]
    nb_hybrid._COMPILED_MOD_CACHE.clear()
    try:
        yield nb_hybrid
    finally:
        for name in tuple(sys.modules):
            if name.startswith("nbhybrid_"):
                del sys.modules[name]
        sys.modules.update(original_modules)
        nb_hybrid._COMPILED_MOD_CACHE.clear()
        nb_hybrid._COMPILED_MOD_CACHE.update(original_cache)


def test_cache_directory_honors_environment_override(
    tmp_path, monkeypatch, isolated_generated_modules
):
    nb_hybrid = isolated_generated_modules
    configured_cache = tmp_path / "configured-hybrid-cache"
    monkeypatch.setenv(_CACHE_ENV, str(configured_cache))
    cache_key = "1" * 64

    run_network = nb_hybrid._build_as_module(
        "def run_network():\n    return 'configured'\n", cache_key
    )

    assert nb_hybrid.NbHybridBackend.get_cache_dir() == configured_cache
    assert run_network() == "configured"
    assert (configured_cache / f"nbhybrid_{cache_key}.py").is_file()


def test_unusable_configured_cache_has_clear_deterministic_error(
    tmp_path, monkeypatch, isolated_generated_modules
):
    nb_hybrid = isolated_generated_modules
    cache_blocker = tmp_path / "not-a-directory"
    cache_blocker.write_text("blocks directory creation", encoding="utf-8")
    monkeypatch.setenv(_CACHE_ENV, str(cache_blocker))

    with pytest.raises((OSError, RuntimeError)) as raised:
        nb_hybrid._build_as_module(
            "def run_network():\n    return None\n", "2" * 64
        )

    message = str(raised.value)
    assert _CACHE_ENV in message
    assert str(cache_blocker) in message
    assert "cache" in message.lower()
    assert any(word in message.lower() for word in ("writable", "create", "directory"))


def test_full_sha256_keys_create_separate_generated_modules(
    tmp_path, monkeypatch, isolated_generated_modules
):
    nb_hybrid = isolated_generated_modules
    monkeypatch.setenv(_CACHE_ENV, str(tmp_path))
    first_key = "a" * 64
    second_key = "a" * 16 + "b" * 48

    first = nb_hybrid._build_as_module(
        "def run_network():\n    return 'first'\n", first_key
    )
    second = nb_hybrid._build_as_module(
        "def run_network():\n    return 'second'\n", second_key
    )

    first_module = sys.modules[first.__module__]
    second_module = sys.modules[second.__module__]
    assert first.__module__ == f"nbhybrid_{first_key}"
    assert second.__module__ == f"nbhybrid_{second_key}"
    assert first_module is not second_module
    assert first() == "first"
    assert second() == "second"
    assert (tmp_path / f"nbhybrid_{first_key}.py").is_file()
    assert (tmp_path / f"nbhybrid_{second_key}.py").is_file()
