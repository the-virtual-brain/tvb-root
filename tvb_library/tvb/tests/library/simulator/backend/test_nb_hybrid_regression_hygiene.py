# -*- coding: utf-8 -*-
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package.
#
# (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
"""Static and behavioral hygiene regressions for the hybrid Numba backend."""

import re
from pathlib import Path

import numpy as np

import tvb.simulator.backend.nb_hybrid as nb_hybrid
from tvb.simulator.hybrid.coupling import Linear, Sigmoidal


BACKEND_DIR = Path(nb_hybrid.__file__).resolve().parent
TEMPLATE_DIR = BACKEND_DIR / "templates"
STATUS_RE = re.compile(
    r"^##\s*Status:\s*(active|deprecated)\s*$", re.IGNORECASE | re.MULTILINE
)
INCLUDE_RE = re.compile(r'<%include\s+file=["\'](nb-hybrid-[^"\']+\.mako)["\']')
GPL_MARKERS = (
    "TheVirtualBrain-Scientific Package",
    "This program is free software",
    "GNU General Public License",
)


def test_nb_hybrid_templates_have_an_accurate_lifecycle_status():
    templates = sorted(TEMPLATE_DIR.glob("nb-hybrid-*.mako"))
    assert templates, "no nb-hybrid Mako templates were found"

    statuses = {}
    unmarked = []
    for template in templates:
        match = STATUS_RE.search(template.read_text(encoding="utf-8"))
        if match is None:
            unmarked.append(template.name)
        else:
            statuses[template.name] = match.group(1).lower()

    assert not unmarked, (
        "nb-hybrid templates must contain '## Status: active' or "
        f"'## Status: deprecated': {unmarked}"
    )

    references = set()
    for source in [*BACKEND_DIR.glob("*.py"), *TEMPLATE_DIR.glob("*.mako")]:
        references.update(INCLUDE_RE.findall(source.read_text(encoding="utf-8")))

    wrongly_active = sorted(
        name for name, status in statuses.items()
        if status == "active" and name not in references
    )
    wrongly_deprecated = sorted(
        name for name, status in statuses.items()
        if status == "deprecated" and name in references
    )
    assert not wrongly_active and not wrongly_deprecated, (
        "template lifecycle status disagrees with include usage; "
        f"unreferenced active={wrongly_active}, referenced deprecated={wrongly_deprecated}"
    )


def test_sweep_sources_have_repository_gpl_header():
    sources = sorted(BACKEND_DIR.glob("*sweep*.py"))
    sources += sorted(TEMPLATE_DIR.glob("*sweep*.mako"))
    assert sources, "no sweep source files were found"

    missing = {}
    for source in sources:
        text = source.read_text(encoding="utf-8")
        absent_markers = [marker for marker in GPL_MARKERS if marker not in text]
        if absent_markers:
            missing[source.relative_to(BACKEND_DIR).as_posix()] = absent_markers

    assert not missing, f"sweep sources missing repository GPL header markers: {missing}"


def _assignment_spy(base, attributes):
    class Spy(base):
        def __init__(self):
            self.trait_assignments = []
            super().__init__()

        def __setattr__(self, name, value):
            if name in attributes:
                self.trait_assignments.append(name)
            super().__setattr__(name, value)

    return Spy


def test_cfun_set_param_assigns_trait_once():
    cases = [
        (_assignment_spy(Linear, {"a", "b"}), 0, "a"),
        (_assignment_spy(Sigmoidal, {"a", "sigma", "midpoint", "cmin", "cmax"}), 0, "a"),
    ]

    for spy_type, param_idx, attribute in cases:
        cfun = spy_type()
        cfun.trait_assignments.clear()

        nb_hybrid.NbHybridBackend._cfun_set_param(cfun, param_idx, 2.5)

        assert cfun.trait_assignments == [attribute]
        np.testing.assert_array_equal(getattr(cfun, attribute), np.array([2.5]))
