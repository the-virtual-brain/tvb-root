"""
Structural properties of component types.

@author: Gautham Ganapathy
@organization: LEMS (http://neuroml.org/lems/, https://github.com/organizations/LEMS)
@contact: gautham@lisphacker.org
"""

from ..base.base import LEMSBase
from ..base.map import Map
from ..base.errors import ModelError

class With(LEMSBase):
    """
    Stores a with-as statement.
    """

    def __init__(self, instance, as_, list=None, index=None):
        """
        Constructor referencing single identified instance, or list/index.
        
        """

        self.instance = instance
        """ Instance to be referenced.
        @type: str """

        self.as_ = as_
        """ Alternative name.
        @type: str """

        self.list = list
        """ list of components, e.g. population
        @type: str """

        self.index = index
        """ index in list to be referenced.
        @type: str """
        

    def toxml(self):
        """
        Exports this object into a LEMS XML object
        """

        return '<With ' + \
          (' instance="{0}"'.format(self.instance) if self.instance else '') +\
          (' list="{0}"'.format(self.list) if self.list else '') + \
          (' index="{0}"'.format(self.index) if self.index else '') + \
          ' as="{1}"/>'.format(self.instance, self.as_)
          
          
class Tunnel(LEMSBase):
    """
    Stores a Tunnel.
    """

    def __init__(self, name, end_a, end_b, component_a, component_b):
        """
        Constructor.
        
        """

        self.name = name
        """ name of Tunnel.
        @type: str """

        self.end_a = end_a
        """ 'A' end of Tunnel.
        @type: str """

        self.end_b = end_b
        """ 'B' end of Tunnel.
        @type: str """

        self.component_a = component_a
        """ Component to create at A.
        @type: str """
        
        self.component_b = component_b
        """ Component to create at B.
        @type: str """

        

    def toxml(self):
        """
        Exports this object into a LEMS XML object
        """

        return '<Tunnel name="{0}"'.format(self.name) + \
               ' endA="{0}"'.format(self.end_a) + \
               ' endB="{0}"'.format(self.end_b) + \
               ' componentA="{0}"'.format(self.component_a) + \
               ' componentB="{0}"'.format(self.component_b) + '/>'

class EventConnection(LEMSBase):
    """
    Stores an event connection specification.
    """

    def __init__(self, from_, to, 
                 source_port, target_port,
                 receiver, receiver_container):
        """
        Constructor.
        
        See instance variable documentation for more details on parameters.
        """

        self.from_ = from_
        """ Name of the source component for event.
        @type: str """

        self.to = to
        """ Name of the target component for the event.
        @type: str """

        self.source_port = source_port
        """ Source port name.
        @type: str """

        self.target_port = target_port
        """ Target port name.
        @type: str """

        self.receiver = receiver
        """ Name of the proxy receiver component attached to the target component that actually receiving the event.
        @type: str """

        self.receiver_container = receiver_container
        """ Name of the child component grouping to add the receiver to.
        @type: str """

    def __eq__(self, o):
        return (self.from_ == o.from_ and self.to == o.to and
                self.source_port == o.source_port and self.target_port == o.target_port)


    def toxml(self):
        """
        Exports this object into a LEMS XML object
        """

        return '<EventConnection' +\
          (' from="{0}"'.format(self.from_) if self.from_ else '') +\
          (' to="{0}"'.format(self.to) if self.to else '') +\
          (' sourcePort="{0}"'.format(self.source_port) if self.source_port else '') +\
          (' targetPort="{0}"'.format(self.target_port) if self.target_port else '') +\
          (' receiver="{0}"'.format(self.receiver) if self.receiver else '') +\
          (' receiverContainer="{0}"'.format(self.receiver_container) if self.receiver_container else '') +\
          '/>'

class ChildInstance(LEMSBase):
    """
    Stores a child instantiation specification.
    """

    def __init__(self, component, referenced_component = None):
        """
        Constructor.
        
        See instance variable documentation for more details on parameters.
        """

        self.component = component
        """ Name of the component reference to be used for instantiation.
        @type: str """

        self.referenced_component = referenced_component
        """ Target component being referenced after resolution.
        @type: lems.model.component.FatComponent """
        
    def __eq__(self, o):
        return self.component == o.component

    def toxml(self):
        """
        Exports this object into a LEMS XML object
        """

        return '<ChildInstance component="{0}"/>'.format(self.component)

class Assign(LEMSBase):
    """
    Stores a child assign specification.
    """

    def __init__(self, property, value):
        """
        Constructor.
        
        See instance variable documentation for more details on parameters.
        """

        self.property_ = property
        """ Name of the property reference to be used for instantiation.
        @type: str """

        self.value = value
        """ Value of the property.
        @type: str"""

    def __eq__(self, o):
        return self.property_ == o.property_ and self.value == o.value

    def toxml(self):
        """
        Exports this object into a LEMS XML object
        """
        return '<Assign property="{0}" value="{1}"/>'.format(self.property_, self.value)


class MultiInstantiate(LEMSBase):
    """
    Stores a child multi-instantiation specification.
    """

    def __init__(self, component=None, number=None, component_type=None):
        """
        Constructor.
        
        See instance variable documentation for more details on parameters.
        """
        if component and component_type:
            raise AttributeError("MultiInstantiate should contain either"
                                 " an attribute component or an attribute"
                                 " component_type, not both.")

        self.component = component
        """ Name of the component reference to be used for instantiation.
        @type: str """

        self.component_type = component_type
        """ Name of the component type reference to be used for instantiation.
        @type: str """

        self.number = number
        """ Name of the paramter specifying the number of times the component 
        reference is to be instantiated.
        @type: str"""

        self.assignments = []
        """ List of assignments included in MultiInstantiate.
        @type: list(Assign) """
        
    def __eq__(self, o):
        if self.component:
            flag = self.component == o.component and self.number == o.number
        else:
            flag = self.component_type == o.component_type and self.number == o.number
        return flag

    def add_assign(self, assign):
        """
        Adds an Assign to the structure.

        @param assign: Assign structure.
        @type assign: lems.model.structure.Assign
        """
        self.assignments.append(assign)

    def add(self, child):
        """
        Adds a typed child object to the structure object.

        @param child: Child object to be added.
        """

        if isinstance(child, Assign):
            self.add_assign(child)
        else:
            raise ModelError('Unsupported child element')

    def toxml(self):
        """
        Exports this object into a LEMS XML object
        """
        argstr = ''
        if self.component:
            argstr += 'component="{0}" '.format(self.component)
        if self.component_type:
            argstr += 'componentType="{0}" '.format(self.component_type)
        if self.number:
            argstr += 'number="{0}" '.format(self.number)
        if self.assignments:
            chxmlstr = ''
            for assign in self.assignments:
                chxmlstr += assign.toxml()
            return '<MultiInstantiate {0}>{1}</MultiInstantiate>'.format(argstr, chxmlstr)
        else:
            return '<MultiInstantiate {0}/>'.format(argstr)

class ForEach(LEMSBase):
    """
    ForEach specification.
    """
    def __init__(self, instances, as_):

        self.instances = instances
        
        self.as_ = as_
        
        self.event_connections = list()
        """ List of event connections.
        @type: list(lems.model.structure.EventConnection) """
        
        self.for_eachs = list()
        """ List of for each specs.
        @type: list(lems.model.structure.ForEach) """
        
        
    def add_for_each(self, fe):
        """
        Adds a for-each specification.

        @param fe: For-each specification.
        @type fe: lems.model.structure.ForEach
        """

        self.for_eachs.append(fe)
        
        
    def add_event_connection(self, ec):
        """
        Adds an event conenction to the structure.

        @param ec: Event connection.
        @type ec: lems.model.structure.EventConnection
        """

        self.event_connections.append(ec)
        
    def toxml(self):
        """
        Exports this object into a LEMS XML object
        """
        chxmlstr = ''

        for event_connection in self.event_connections:
            chxmlstr += event_connection.toxml()

        for for_each in self.for_eachs:
            chxmlstr += for_each.toxml()


        return '<ForEach instances="{0}" as="{1}">{2}</ForEach>'.format(self.instances, self.as_, chxmlstr)
        
        
class Structure(LEMSBase):
    """
    Stores structural properties of a component type.
    """

    def __init__(self):
        """
        Constructor.
        """

        self.withs = Map()
        """ Map of With statements.
        @type: Map(str -> lems.model.structure.With) """
        
        self.tunnels = Map()
        """ Map of tunnel statements.
        @type: Map(str -> lems.model.structure.Tunnel) """

        self.event_connections = list()
        """ List of event connections.
        @type: list(lems.model.structure.EventConnection) """

        self.child_instances = list()
        """ List of child instantations.
        @type: list(lems.model.structure.ChildInstance) """

        self.multi_instantiates = list()
        """ List of child multi-instantiations.
        @type: list(lems.model.structure.MultiInstantiate) """

        self.for_eachs = list()
        """ List of for each specs.
        @type: list(lems.model.structure.ForEach) """
        
    def has_content(self):
        if len(self.withs)==0 and \
           len(self.event_connections)==0 and \
           len(self.child_instances)==0 and \
           len(self.multi_instantiates)==0 and \
           len(self.for_eachs)==0:
               return False
        else:
            return True

    def add_with(self, with_):
        """
        Adds a with-as specification to the structure.

        @param with_: With-as specification.
        @type with_: lems.model.structure.With
        """

        self.withs[with_.as_] = with_

    def add_tunnel(self, tunnel):
        """
        Adds a tunnel specification to the structure.

        @param tunnel: tunnel specification.
        @type tunnel: lems.model.structure.Tunnel
        """

        self.tunnels[tunnel.name] = tunnel
        
    def add_event_connection(self, ec):
        """
        Adds an event conenction to the structure.

        @param ec: Event connection.
        @type ec: lems.model.structure.EventConnection
        """

        self.event_connections.append(ec)

    def add_child_instance(self, ci):
        """
        Adds a child instantiation specification.

        @param ci: Child instantiation specification.
        @type ci: lems.model.structure.ChildInstance
        """

        self.child_instances.append(ci)

    def add_multi_instantiate(self, mi):
        """
        Adds a child multi-instantiation specification.

        @param mi: Child multi-instantiation specification.
        @type mi: lems.model.structure.MultiInstantiate
        """

        self.multi_instantiates.append(mi)

    def add_for_each(self, fe):
        """
        Adds a for-each specification.

        @param fe: For-each specification.
        @type fe: lems.model.structure.ForEach
        """

        self.for_eachs.append(fe)

    def add(self, child):
        """
        Adds a typed child object to the structure object.

        @param child: Child object to be added.
        """

        if isinstance(child, With):
            self.add_with(child)
        elif isinstance(child, EventConnection):
            self.add_event_connection(child)
        elif isinstance(child, ChildInstance):
            self.add_child_instance(child)
        elif isinstance(child, MultiInstantiate):
            self.add_multi_instantiate(child)
        elif isinstance(child, ForEach):
            self.add_for_each(child)
        else:
            raise ModelError('Unsupported child element')

    def toxml(self):
        """
        Exports this object into a LEMS XML object
        """
        chxmlstr = ''

        for with_ in self.withs:
            chxmlstr += with_.toxml()

        for event_connection in self.event_connections:
            chxmlstr += event_connection.toxml()

        for child_instance in self.child_instances:
            chxmlstr += child_instance.toxml()

        for multi_instantiate in self.multi_instantiates:
            chxmlstr += multi_instantiate.toxml()

        for for_each in self.for_eachs:
            chxmlstr += for_each.toxml()

        if chxmlstr:
            xmlstr = '<Structure>' + chxmlstr + '</Structure>'
        else:
            xmlstr = ''

        return xmlstr
