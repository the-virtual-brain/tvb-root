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
.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
"""

import os
import logging
from logging.handlers import MemoryHandler
from tvb.basic.config.settings import TVBSettings
from tvb.basic.logger.simple_handler import SimpleTimedRotatingFileHandler


class ClusterTimedRotatingFileHandler(MemoryHandler):
    """
    This is a custom rotating file handler which computes the name of the file depending on the 
    execution environment (web node or cluster node)
    """

    # Name of the log file where code from Web application will be stored
    WEB_LOG_FILE = "web_application.log"

    # Name of the file where to write logs from the code executed on cluster nodes
    CLUSTER_NODES_LOG_FILE = "operations_executions.log"

    # Size of the buffer which store log entries in memory
    # in number of lines
    BUFFER_CAPACITY = 20


    def __init__(self, when='h', interval=1, backupCount=0):
        """
        Constructor for logging formatter.
        """
        # Formatting string
        format_str = '%(asctime)s - %(levelname)s'
        if TVBSettings.OPERATION_EXECUTION_PROCESS:
            log_file = os.path.join(TVBSettings.TVB_LOG_FOLDER, self.CLUSTER_NODES_LOG_FILE)
            if TVBSettings.RUNNING_ON_CLUSTER_NODE:
                node_name = TVBSettings.CLUSTER_NODE_NAME
                if node_name is not None:
                    format_str += ' [node:' + str(node_name) + '] '
            else:
                format_str += ' [proc:' + str(os.getpid()) + '] '
        else:
            log_file = os.path.join(TVBSettings.TVB_LOG_FOLDER, self.WEB_LOG_FILE)

        format_str += ' - %(name)s - %(message)s'

        rotating_file_handler = SimpleTimedRotatingFileHandler(log_file, when, interval, backupCount)
        rotating_file_handler.setFormatter(logging.Formatter(format_str))

        MemoryHandler.__init__(self, capacity=self.BUFFER_CAPACITY, target=rotating_file_handler)
        
        
        
        