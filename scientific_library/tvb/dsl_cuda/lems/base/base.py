"""
PyLEMS base class.

@author: Gautham Ganapathy
@organization: LEMS (http://neuroml.org/lems/, https://github.com/organizations/LEMS)
@contact: gautham@lisphacker.org
"""

import copy

class LEMSBase(object):
    """
    Base object for PyLEMS.
    """

    def copy(self):
        return copy.deepcopy(self)

    def toxml(self):
        return ''
