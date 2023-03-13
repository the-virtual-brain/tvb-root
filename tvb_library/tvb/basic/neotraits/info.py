# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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
Functions that inform a user about the state of a traited class or object.

Some of these functions are here so that they won't clutter the core trait implementation.
"""

import uuid
import numpy
import typing

try:
    from docutils.core import publish_parts
except ImportError:
    def publish_parts(param1, _param2):
        return param1


def auto_docstring(cls):
    """ generate a docstring for the new class in which the Attrs are documented """
    header = 'Traited class [{}.{}]'.format(cls.__module__, cls.__name__)

    doc = [
        header,
        len(header) * '^',
        '',
    ]
    if cls.__doc__ is not None:
        doc.extend([cls.__doc__, ''])

    doc.extend([
        'Attributes declared',
        '"""""""""""""""""""',
        ''
    ])
    # a rst definition list for all attributes
    for attr_name in cls.declarative_attrs:
        attr = getattr(cls, attr_name)
        # the standard repr of the attribute
        doc.append('{} : {}'.format(attr_name, str(attr)))
        # and now the doc property
        for line in attr.doc.splitlines():
            doc.append('    ' + line.lstrip())
        doc.append('')

    if cls.declarative_props:
        doc.extend([
            '',
            'Properties declared',
            '"""""""""""""""""""',
            ''
        ])

    for prop_name in cls.declarative_props:
        prop = getattr(cls, prop_name)
        # the standard repr
        doc.append('  {} : {}'.format(prop_name, str(prop)))
        # now fish the docstrings
        for line in prop.attr.doc.splitlines():
            doc.append('    ' + line.lstrip())
        if prop.fget.__doc__ is not None:
            for line in prop.fget.__doc__.splitlines():
                doc.append('    ' + line.lstrip())

    doc = '\n'.join(doc)

    return doc


def narray_summary_info(ar, ar_name='', condensed=False):
    # type: (numpy.ndarray, str, bool) -> typing.Dict[str, str]
    """
    A 2 column table represented as a dict of str->str
    """

    key_none = 'is None'
    key_empty = 'is empty'
    key_min_max = '[min, median, max]'
    key_nan = 'has NaN'
    key_shape = 'shape'
    key_type = 'dtype'

    if ar is None:
        return {f'{ar_name} {key_none}': 'True'} if ar_name else {key_none: 'True'}

    if ar.size == 0:
        return {f'{ar_name} {key_empty}': 'True'} if ar_name else {key_empty: 'True'}

    ret = {key_shape: str(ar.shape),
           key_type: str(ar.dtype)}

    if ar.dtype.kind in 'iufc':
        has_nan = numpy.isnan(ar).any()
        if has_nan:
            ret[key_nan] = 'True'
        ret[key_min_max] = '[{:g}, {:g}, {:g}]'.format(ar.min(), numpy.median(ar), ar.max())

    if condensed:
        condensed_desc = ""
        if ar.shape == (1,):
            condensed_desc += str(ar.item())
        else:
            for key in [key_min_max, key_type, key_shape, key_empty, key_nan]:
                if key in ret:
                    condensed_desc += f' {key} = {ret[key]}'

        return {ar_name: condensed_desc} if ar_name else {'array': condensed_desc}

    if ar_name:
        return {ar_name + ' ' + k: v for k, v in ret.items()}
    else:
        return ret


def narray_describe(ar):
    # type: (numpy.ndarray) -> str
    summary = narray_summary_info(ar)
    ret = []
    for k in sorted(summary):
        ret.append('{:<12}{}'.format(k, summary[k]))
    return '\n'.join(ret)


# these are here and not on HasTraits just so that that class is not
# complicated by irrelevant string formatting


def trait_object_str(self):
    cls = type(self)
    summary = self.summary_info()
    result = ['{} ('.format(cls.__name__)]
    maxlenk = max(len(k) for k in summary)

    for k in sorted(summary):
        result.append('  {:.<{}} {}'.format(k + ' ', maxlenk, summary[k]))
    result.append(')')
    return '\n'.join(result)


def trait_object_repr_html(self):
    cls = type(self)

    subtitle = None
    if hasattr(self, "dfun"):
        subtitle = self.dfun.__doc__
    elif hasattr(cls, "__doc_old__"):
        subtitle = cls.__doc_old__

    result = [
        '<table>',
        '<thead><h3>{}</h3></thead>'.format(cls.__name__),
        '<tbody>',
        '<tr><td colspan="2"><p>{}</p></td></tr>'.format(prepare_html(subtitle)) if subtitle is not None else "",
        '<tr><th></th><th style="text-align:left;width:80%">value</th></tr>',
    ]

    summary = self.summary_info()

    for k in sorted(summary):
        row_fmt = '<tr><td>{}</td><td style="text-align:left;"><pre>{}</pre></td></tr>'
        result.append(row_fmt.format(k, summary[k]))

    result += ['</tbody></table>']

    return '\n'.join(result)


def convert_rst_to_html(doc):
    """
    Convert from rst to html that can be rendered by Mathjax
    """
    kwargs = {
        'writer_name': 'html',
        'settings_overrides': {
            '_disable_config': True,
            'report_level': 5,
            'math_output': "MathJax /dummy.js",
        },
    }

    return publish_parts(doc, **kwargs)['html_body']


def prepare_html(doc):
    # type: (str) -> str
    """
    Create html (that can be further enhanced by MathJax) from the description received as parameter
    """
    try:
        html_id = uuid.uuid1()

        html = convert_rst_to_html(doc)

        html = html.replace('div class="document"', f'div class="document" id="{html_id}"', 1)
        html += fr'<script>MathJax.Hub.Queue(["Typeset", MathJax.Hub, "{html_id}"]);</script>'

    except Exception:
        html = str(doc)
    return html
