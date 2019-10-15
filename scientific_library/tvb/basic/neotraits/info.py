"""
Functions that inform a user about the state of a traited class or object.

Some of these functions are here so that they won't clutter the core trait implementation.
"""

import numpy


def auto_docstring(cls):
    # type: (HasTraits) -> str
    """ generate a docstring for the new class in which the Attrs are documented """
    doc = [
        'Traited class [{}.{}]'.format(cls.__module__, cls.__name__),
        '',
        '  Attributes declared',
        '  -------------------',
        ''
    ]

    for attr_name in cls.declarative_attrs:
        attr = getattr(cls, attr_name)
        # the standard repr of the attribute
        doc.append('  {} : {}'.format(attr_name, str(attr)))
        # and now the doc property
        for line in attr.doc.splitlines():
            doc.append('    ' + line.lstrip())

    doc.extend([
        '',
        '  Properties declared',
        '  -------------------',
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

    if cls.__doc__ is not None:
        return cls.__doc__ + doc
    else:
        return doc


def narray_describe(ar):
    if ar is None:
        return 'None'
    ret = [
        'shape    {}'.format(ar.shape),
        'dtype    {}'.format(ar.dtype),
    ]

    if ar.size == 0:
        ret.append('is empty')
        return '\n'.join(ret)

    if ar.dtype.kind in 'iufc':
        ret += [
            'has NaN  {}'.format(numpy.isnan(ar).any()),
            'maximum  {}'.format(ar.max()),
            'minimum  {}'.format(ar.min())
        ]
    return '\n'.join(ret)


def trait_object_str(self):
    cls = type(self)
    result = ['{} ('.format(self.__class__.__name__)]
    for aname in cls.declarative_attrs:
        attr_field = getattr(self, aname)
        if isinstance(attr_field, numpy.ndarray):
            attr_repr = ['array'] + narray_describe(attr_field).splitlines()
        else:
            # str would be pretty. but recursive types will stack overflow then
            attr_repr = repr(attr_field).splitlines()
        attr_repr = attr_repr[:1] + ['      ' + s for s in attr_repr[1:]]
        attr_repr = '\n'.join(attr_repr)
        result.append('  {} = {},'.format(aname, attr_repr))
    result.append(')')
    return '\n'.join(result)


def trait_object_repr_html(self):
    cls = type(self)
    result = [
        '<table>',
        '<h3>{}</h3>'.format(self.__class__.__name__),
        '<thead><tr><th>attribute</th><th style="text-align:left;width:40%">value</th><th>about</th></tr></thead>',
        '<tbody>']

    for aname in cls.declarative_attrs:
        attr_field = getattr(self, aname)
        attr_doc = getattr(cls, aname).doc
        row_fmt = '<tr><td>{}</td><td style="text-align:left;"><pre>{}</pre></td><td style="text-align:left;">{}</td>'
        if isinstance(attr_field, numpy.ndarray):
            attr_repr = narray_describe(attr_field).splitlines()
        else:
            attr_repr = repr(attr_field).splitlines()
        attr_repr = '\n'.join(attr_repr)
        result.append(row_fmt.format(aname, attr_repr, attr_doc))

    result += ['</tbody></table>']

    return '\n'.join(result)


