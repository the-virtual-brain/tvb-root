# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

"""
Functions responsible for collecting and rendering the descriptions and the documentations
of the dynamic models in Simulator/Phase Plane.

.. moduleauthor:: David Bacter <david.bacter@codemart.ro>
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
.. moduleauthor:: Robert Vincze <robert.vincze@codemart.ro>
"""

from tvb.adapters.forms.model_forms import ModelsEnum
from tvb.basic.neotraits.info import convert_rst_to_html


def configure_matjax_doc():
    """
    Builds a list containing the model names, inline descriptions and descriptions,
    which will be displayed in the Simulation/Phase Plane section.
    """
    models_docs = []

    for member in list(ModelsEnum):
        clz_name = str(member)
        clz = member.value
        models_docs.append({
            'name': clz_name.replace(' ', '_'),
            'inline_description': _dfun_math_directives_to_matjax(clz),
            'description': convert_rst_to_html(clz.__doc__)
        })

    return models_docs


def _dfun_math_directives_to_matjax(model):
    """
    Looks for sphinx math directives if the docstring of the dfun function of a model.
    It converts them in html text that will be interpreted by mathjax
    The parsing is simplistic, not a full rst parser.
    """
    try:
        doc = model.dfun.__doc__
    except AttributeError:
        doc = None

    if doc is not None:
        return convert_rst_to_html(doc)

    # try the parent __doc__
    try:
        doc = model.__mro__[1].dfun.__doc__
    except (AttributeError, IndexError):
        doc = None

    if doc is not None:
        return convert_rst_to_html('Documentation is missing. Copy-ed from parent\n' + doc)

    return 'Documentation is missing. '
