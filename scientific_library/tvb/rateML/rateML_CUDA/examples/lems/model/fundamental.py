"""
Dimension and Unit definitions in terms of the fundamental SI units.

@author: Gautham Ganapathy
@organization: LEMS (http://neuroml.org/lems/, https://github.com/organizations/LEMS)
@contact: gautham@lisphacker.org
"""

from lems.base.base import LEMSBase

class Include(LEMSBase):
    """
    Include another LEMS file.
    """
    
    def __init__(self, filename):
        """
        Constructor.

        @param filename: Name of the file.
        @type name: str

        """
        
        self.file = filename
        """ Name of the file.
        @type: str """
        

    def toxml(self):
        """
        Exports this object into a LEMS XML object
        """

        return '<Include file="%s"/>'%self.file

class Dimension(LEMSBase):
    """
    Stores a dimension in terms of the seven fundamental SI units.
    """
    
    def __init__(self, name, description = '', **params):
        """
        Constructor.

        @param name: Name of the dimension.
        @type name: str

        @param params: Key arguments specifying powers for each of the 
        seven fundamental SI dimensions.
        @type params: dict()
        """
        
        self.name = name
        """ Name of the dimension.
        @type: str """

        self.m = params['m'] if 'm' in params else 0
        """ Power for the mass dimension.
        @type: int """
        
        self.l = params['l'] if 'l' in params else 0
        """ Power for the length dimension.
        @type: int """
        
        self.t = params['t'] if 't' in params else 0
        """ Power for the time dimension.
        @type: int """
        
        self.i = params['i'] if 'i' in params else 0
        """ Power for the electic current dimension.
        @type: int """
        
        self.k = params['k'] if 'k' in params else 0
        """ Power for the temperature dimension.
        @type: int """
        
        self.n = params['n'] if 'n' in params else 0
        """ Power for the quantity dimension.
        @type: int """
        
        self.j = params['j'] if 'j' in params else 0
        """ Power for the luminous intensity dimension.
        @type: int """

        self.description = description
        """ Description of this dimension.
        @type: str """
        

    def toxml(self):
        """
        Exports this object into a LEMS XML object
        """

        return '<Dimension name="{0}"'.format(self.name) +\
          (' m = "{0}"'.format(self.m) if self.m != 0 else '') +\
          (' l = "{0}"'.format(self.l) if self.l != 0 else '') +\
          (' t = "{0}"'.format(self.t) if self.t != 0 else '') +\
          (' i = "{0}"'.format(self.i) if self.i != 0 else '') +\
          (' k = "{0}"'.format(self.k) if self.k != 0 else '') +\
          (' n = "{0}"'.format(self.n) if self.n != 0 else '') +\
          (' j = "{0}"'.format(self.j) if self.j != 0 else '') +\
          '/>'
          
class Unit(LEMSBase):
    """
    Stores a unit definition.
    """
    
    def __init__(self, name, symbol, dimension, power = 0, scale = 1.0, offset = 0.0, description = ''):
        """
        Constructor.

        See instance variable documentation for more details on parameters.
        """
        
        self.name = name
        """ Name of the unit.
        @type: str """
        
        self.symbol = symbol
        """ Symbol for the unit.
        @type: str """
        
        self.dimension = dimension
        """ Dimension for the unit.
        @type: str """
        
        self.power = power
        """ Scaling by power of 10.
        @type: int """
        
        self.scale = scale
        """ Scaling.
        @type: float """
        
        self.offset = offset
        """ Offset for non-zero units.
        @type: float """

        self.description = description
        """ Description of this unit.
        @type: str """
        
    def toxml(self):
        """
        Exports this object into a LEMS XML object
        """

        # Probably name should be removed altogether until its usage is decided, see
        # https://github.com/LEMS/LEMS/issues/4
        #  '''(' name = "{0}"'.format(self.name) if self.name else '') +\'''

        return '<Unit' +\
          (' symbol = "{0}"'.format(self.symbol) if self.symbol else '') +\
          (' dimension = "{0}"'.format(self.dimension) if self.dimension else '') +\
          (' power = "{0}"'.format(self.power) if self.power else '') +\
          (' scale = "{0}"'.format(self.scale) if self.scale else '') +\
          (' offset = "{0}"'.format(self.offset) if self.offset else '') +\
          (' description = "{0}"'.format(self.description) if self.description else '') +\
          '/>'
