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
Providing the Burst services as an HTTP/JSON API.

"""

import json
import cherrypy
import tvb.interfaces.web.controllers.base_controller as base
import tvb.interfaces.web.controllers.api.burst_impl as impl
from tvb.core.services.burst_service import BurstService
from tvb.datatypes import connectivity, equations, surfaces, patterns
from tvb.simulator import noise, integrators, models, coupling, monitors



class BurstAPIController(base.BaseController):
    """
    Provides an HTTP/JSON API to the simulator/burst mechanics
    reusing a BurstController where necessary.
    """

    exposed = True

    def __init__(self):
        super(BurstAPIController, self).__init__()
        self.burst_service = BurstService()


    @cherrypy.expose
    def index(self):

        reload(impl)
        return impl.index(self)


    @cherrypy.expose
    def read(self, pid):
        """
        Get information on existing burst operations
        """

        reload(impl)
        return impl.read(self, pid)


    @cherrypy.expose
    def dir(self):
        """
        Query existing classes in TVB
        """

        info = {}

        for m in [models, coupling, integrators, noise, monitors, connectivity, equations, surfaces, patterns]:
            minfo = {}
            for k in dir(m):
                v = getattr(m, k)
                if isinstance(v, type):
                    minfo[k] = k + '\n\n' + getattr(v, '__doc__', k)
            info[m.__name__.split('.')[-1]] = minfo

        return json.dumps(info)


    @cherrypy.expose
    def create(self, opt):
        """
        Create a new simulation/burst
        """

        reload(impl)
        return impl.create(self, opt)


