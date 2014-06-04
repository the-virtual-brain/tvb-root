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

from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.abcadapter import ABCAsynchronous, ABCGroupAdapter


class PythonAdapter(ABCAsynchronous, ABCGroupAdapter):
    """
    Interface between some simple Python analyzers and TVB Framework.
    """
    def __init__(self, xml_file_path):
        ABCAsynchronous.__init__(self)
        ABCGroupAdapter.__init__(self, xml_file_path)
        self.log = get_logger(self.__class__.__module__)
    
    def get_required_memory_size(self, **kwargs):
        """
        Return the required memory to run this algorithm.
        """
        # Don't know how much memory is needed.
        return -1
    
    def get_required_disk_size(self, **kwargs):
        """
        Returns the required disk size to be able to run the adapter (in kB).
        """
        return 0
    
    def launch(self, **kwargs):
        """
        Pick the correct algorithm to use, and launch the PYTHON call. 
        After computation, make sure the correct results are returned.
        """
        algorithm, kwargs = self.get_algorithm_and_attributes(**kwargs)
        
        # Execute PYTHON code
        algorithm_module = __import__(self.get_import_code(algorithm),
                                      globals(), locals(), ["__init__"])
        python_code = self.get_call_code(algorithm)
        self.log.info("Starting execution of PYTHON code:" + python_code)
        algo_inputs = self.xml_reader.get_inputs(algorithm)
        result = eval('algorithm_module.' + python_code, globals(), locals())
        self.log.debug("Finished PYTHON execution:" + str(result))
        
        #Now build PYTHON result objects
        return self.build_result(algorithm, [result], kwargs)

