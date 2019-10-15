"""
Functions that inform a user about the state of a traited class or object.

Some of these functions are here so that they won't clutter the core trait implementation.
"""

import numpy


def auto_docstring(cls):
    """ generate a docstring for the new class in which the Attrs are documented """
    doc = [
        'Traited class [{}.{}]'.format(cls.__module__, cls.__name__),
        '  + Attributes declared by the type:',
        '  {'
    ]

    for attr_name in cls.declarative_attrs:
        attr_repr = str(getattr(cls, attr_name))
        attr_repr = '\n      '.join(attr_repr.split(','))
        doc.append('    {} = {}'.format(attr_name, attr_repr))
    doc.append('  }')
    doc = '\n'.join(doc)

    if cls.__doc__ is not None:
        return cls.__doc__ + doc
    else:
        return doc


def narray_describe(ar):
    if ar is None:
        return str(ar)
    ret = [
        'shape    {}'.format(ar.shape),
        'dtype    {}'.format(ar.dtype),
    ]
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
        '<caption>{}</caption>'.format(self.__class__.__name__),
        '<thead><tr><th>attribute</th><th style="text-align:left;width:40%">value</th><th>about</th></tr></thead>',
        '<tbody>']

    for aname in cls.declarative_attrs:
        attr_field = getattr(self, aname)
        attr_doc = getattr(cls, aname).doc
        row_fmt = '<tr><td>{}</td><td><pre>{}</pre></td><td>{}</td>'
        if isinstance(attr_field, numpy.ndarray):
            attr_repr = narray_describe(attr_field).splitlines()
        else:
            attr_repr = repr(attr_field).splitlines()
        attr_repr = '\n'.join(attr_repr)
        result.append(row_fmt.format(aname, attr_repr, attr_doc))

    result += ['</tbody></table>']

    return '\n'.join(result)


