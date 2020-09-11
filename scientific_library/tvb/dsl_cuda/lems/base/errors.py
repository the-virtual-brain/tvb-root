"""
Error classes.

@author: Gautham Ganapathy
@organization: LEMS (http://neuroml.org/lems/, https://github.com/organizations/LEMS)
@contact: gautham@lisphacker.org
"""

class LEMSError(Exception):
    """
    Base exception class.
    """

    def __init__(self, message, *params, **key_params):
        """
        Constructor

        @param message: Error message.
        @type message: string

        @param params: Optional arguments for formatting.
        @type params: list

        @param key_params: Named arguments for formatting.
        @type key_params: dict
        """
        
        self.message = None
        """ Error message
        @type: string """

        if params:
            if key_params:
                self.message = message.format(*params, **key_params)
            else:
                self.message = message.format(*params)
        else:
            if key_params:
                self.message = message(**key_params)
            else:
                self.message = message    

    def __str__(self):
        """
        Returns the error message string.

        @return: The error message
        @rtype: string
        """
        
        return self.message

class StackError(LEMSError):
    """
    Exception class to signal errors in the Stack class.
    """

    pass

class ParseError(LEMSError):
    """
    Exception class to signal errors found during parsing.
    """

    pass

class ModelError(LEMSError):
    """
    Exception class to signal errors in creating the model.
    """

    pass

class SimBuildError(LEMSError):
    """
    Exception class to signal errors in building the simulation.
    """

    pass

class SimError(LEMSError):
    """
    Exception class to signal errors in simulation.
    """

    pass
