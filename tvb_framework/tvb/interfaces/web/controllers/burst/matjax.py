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
from docutils.core import publish_parts


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
            # 'description': _format_doc(clz.__doc__).replace('\n', '<br/>')
            # I let this here since the html parse has a small flaw regarding some overlapping text
            # and we might consider switching back to plain text
            'description': publish_parts(clz.__doc__, writer_name='html')['html_body']
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
        return _format_doc(doc)

    # try the parent __doc__
    try:
        doc = model.__mro__[1].dfun.__doc__
    except (AttributeError, IndexError):
        doc = None

    if doc is not None:
        return _format_doc('Documentation is missing. Copy-ed from parent\n' + doc)

    return 'Documentation is missing. '


def _format_doc(doc):
    return _multiline_math_directives_to_matjax(doc).replace('&', '&amp;').replace('.. math::', '')


def _multiline_math_directives_to_matjax(doc):
    """
    Looks for multi-line sphinx math directives in the given rst string
    It converts them in html text that will be interpreted by mathjax
    The parsing is simplistic, not a rst parser.
    Wraps .. math :: body in \[\begin{split}\end{split}\]
    """

    # doc = text | math
    BEGIN = r'\[\begin{split}'
    END = r'\end{split}\]'

    in_math = False  # 2 state parser
    out_lines = []
    indent = ''

    for line in doc.splitlines():
        if not in_math:
            # math = indent directive math_body
            indent, sep, _ = line.partition('.. math::')
            if sep:
                out_lines.append(BEGIN)
                in_math = True
            else:
                out_lines.append(line)
        else:
            # math body is at least 1 space more indented than the directive, but we tolerate empty lines
            if line.startswith(indent + ' ') or line.strip() == '':
                out_lines.append(line)
            else:
                # this line is not properly indented, math block is over
                out_lines.append(END)
                out_lines.append(line)
                in_math = False

    if in_math:
        # close math tag
        out_lines.append(END)

    return '\n'.join(out_lines)
