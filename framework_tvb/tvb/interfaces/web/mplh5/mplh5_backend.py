# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
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
Html5 Backend for Matplotlib.

.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import thread
import mplh5canvas.simple_server as simple_server
from tvb.basic.config.settings import TVBSettings as cfg
from mplh5canvas.backend_h5canvas import web_socket_transfer_data
from mplh5canvas.backend_h5canvas import FigureManagerH5Canvas

# new_figure_manager and draw_if_interactive are required for a valid backend
from mplh5canvas.backend_h5canvas import new_figure_manager
from mplh5canvas.backend_h5canvas import draw_if_interactive

try:
    FIGURES_SERVER = simple_server.WebSocketServer(('', cfg.MPLH5_SERVER_PORT),
                                                   web_socket_transfer_data,
                                                   simple_server.WebSocketRequestHandler)
    THREAD = thread.start_new_thread(FIGURES_SERVER.serve_forever, ())
    cfg.MPLH5_Server_Thread = FIGURES_SERVER
except Exception, excep:
    print "Error WebSocketServer %i(%s)" % (cfg.MPLH5_SERVER_PORT, str(excep))

FigureManager = FigureManagerH5Canvas


