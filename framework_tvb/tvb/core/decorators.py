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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import os


def synchronized(lock):
    """ 
    Synchronization annotation. 
    We try to mimic the same behavior as Java has with keyword synchronized, for methods.
    """


    def wrap(func):
        """Wrap current function with a lock mechanism"""


        def new_function(*args, **kw):
            """ New function will actually write the Lock."""
            lock.acquire()
            try:
                return func(*args, **kw)
            finally:
                lock.release()


        return new_function


    return wrap


def user_environment_execution(func):
    """
    Decorator that makes sure a function is executed in a 'user' environment,
    removing any TVB specific configurations that alter either LD_LIBRARY_PATH
    or LD_RUN_PATH.
    """
    
    def new_function(*args, **kwargs):
        # Wrapper function
        ORIGINAL_LD_LIBRARY_PATH = os.environ.get('LD_LIBRARY_PATH', None)
        ORIGINAL_LD_RUN_PATH = os.environ.get('LD_RUN_PATH', None)
        if ORIGINAL_LD_LIBRARY_PATH:
            del os.environ['LD_LIBRARY_PATH']
        if ORIGINAL_LD_RUN_PATH:
            del os.environ['LD_RUN_PATH']
        func(*args, **kwargs)
        ## Restore environment settings after function executed.
        if ORIGINAL_LD_LIBRARY_PATH:
            os.environ['LD_LIBRARY_PATH'] = ORIGINAL_LD_LIBRARY_PATH
        if ORIGINAL_LD_RUN_PATH:
            os.environ['LD_RUN_PATH'] = ORIGINAL_LD_RUN_PATH
            
    return new_function

