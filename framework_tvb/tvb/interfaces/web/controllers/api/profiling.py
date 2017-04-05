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

Facilitate profiling.

"""

import cherrypy

try:
    import yappi
    import yappi_stats

    class Yappi(object):
        exposed = True


        @cherrypy.expose
        def start(self):
            yappi.start()
            return '<a href="stats">stats</a>'


        @cherrypy.expose
        def stop(self):
            yappi.stop()
            return '<a href="stats">stats</a>'


        @cherrypy.expose
        def clear_stats(self):
            yappi.clear_stats()
            return '<a href="stats">stats</a>'


        @cherrypy.expose
        def stats(self):
            reload(yappi_stats)
            return yappi_stats.impl()

except ImportError:
    class Yappi(object):
        exposed = True


        @cherrypy.expose
        def start(self):
            return 'Yappi not available'


