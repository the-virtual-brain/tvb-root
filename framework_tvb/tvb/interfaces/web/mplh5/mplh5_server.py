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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import threading
import matplotlib
# Import this, to make sure the build process marks this code as reference.
import mplh5canvas.simple_server

SYNC_EVENT = threading.Event()



class ServerStarter(threading.Thread):
    """
    Handler for starting in a different thread the MPLH5 server.
    Synchronization event. We want to start MPLH5 server in a new thread, 
    but the main thread should wait for it, otherwise wrong import 
    of pylb might be used.
    """
    logger = None

    def run(self):
        """
        Start MPLH5 server. 
        This method needs to be executed as soon as possible, before any import of pylab.
        Otherwise the proper mplh5canvas back-end will not be used correctly.
        """
        try:
            matplotlib.use('module://tvb.interfaces.web.mplh5.mplh5_backend')
            self.logger.info("MPLH5 back-end server started.")
        except Exception, excep:
            self.logger.error("Could not start MatplotLib server side!!!")
            self.logger.exception(excep)
        SYNC_EVENT.set()



def start_server(logger):
    """Start MPLH5 server in a new thread, to avoid crashes."""
    thread = ServerStarter()
    thread.logger = logger
    thread.start()
    SYNC_EVENT.wait()



