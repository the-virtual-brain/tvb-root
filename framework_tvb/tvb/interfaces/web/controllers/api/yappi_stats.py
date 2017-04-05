# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
.. moduleauthor:: Marmaduke Woodman <mw@eml.cc>
"""

try:
    import yappi

except ImportError:
    raise Exception("Yappi not available!!")


def impl():
    out = '''
    <p> Profile is{0}running. 
    <a href='start'>Start</a> <a href='stop'>Stop</a> <a href='clear_stats'>Clear stats</a>
    '''.format(
        ' ' if yappi.is_running() else ' not '
    )
    fkeys = 'index module lineno name ncall nactualcall builtin ttot tsub tavg'.split()
    out += '<table><tr>{0}</tr>'.format(''.join(map('<th>{0}</th>'.format, fkeys)))
    fmt = '<tr>{0}</tr>'.format(''.join('<td>{0.%s}</td>' % (k, ) for k in fkeys))
    out += ''.join(map(fmt.format, yappi.get_func_stats()))
    out += '</table>'
    return out
