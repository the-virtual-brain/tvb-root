"""
Simulation specification classes.

@author: Gautham Ganapathy
@organization: LEMS (http://neuroml.org/lems/, https://github.com/organizations/LEMS)
@contact: gautham@lisphacker.org
"""

from lems.base.base import LEMSBase
from lems.base.errors import ModelError
from lems.base.map import Map

class Run(LEMSBase):
    """
    Stores the description of an object to be run according to an independent
    variable (usually time).
    """

    def __init__(self, component, variable, increment, total):
        """
        Constructor.

        See instance variable documentation for information on parameters.
        """

        self.component = component
        """ Name of the target component to be run according to the
        specification given for an independent state variable.
        @type: str """

        self.variable = variable
        """ The name of an independent state variable according to which the
        target component will be run.
        @type: str """

        self.increment = increment
        """ Increment of the state variable on each step.
        @type: str """

        self.total = total
        """ Final value of the state variable.
        @type: str """

    def toxml(self):
        """
        Exports this object into a LEMS XML object
        """

        return '<Run component="{0}" variable="{1}" increment="{2}" total="{3}"/>'.format(self.component,
                                                                                          self.variable,
                                                                                          self.increment,
                                                                                          self.total)

class Record(LEMSBase):
    """
    Stores the parameters of a <Record> statement.
    """

    def __init__(self, quantity, scale = None, color = None, id = None):
        """
        Constructor.

        See instance variable documentation for information on parameters.
        """

        self.id = ''
        """ Id of the quantity
        @type: str """
        
        self.quantity = quantity
        """ Path to the quantity to be recorded.
        @type: str """

        self.scale = scale
        """ Text parameter to be used for scaling the quantity before display.
        @type: str """

        self.color = color
        """ Text parameter to be used to specify the color for display.
        @type: str """

        self.id = id
        """ Text parameter to be used to specify an id for the record
        @type: str """

    def toxml(self):
        """
        Exports this object into a LEMS XML object
        """

        return '<Record quantity="{0}" scale="{1}" color="{2}" id="{3}"/>'.format(self.quantity,
                                                                         self.scale,
                                                                         self.color,
                                                                         self.id)
class EventRecord(LEMSBase):
    """
    Stores the parameters of an <EventRecord> statement.
    """

    def __init__(self, quantity, eventPort):
        """
        Constructor.

        See instance variable documentation for information on parameters.
        """

        self.id = ''
        """ Id of the quantity
        @type: str """
        
        self.quantity = quantity
        """ Path to the quantity to be recorded.
        @type: str """

        self.eventPort = eventPort
        """ eventPort to be used for the event record
        @type: str """

    def toxml(self):
        """
        Exports this object into a LEMS XML object
        """

        return '<EventRecord quantity="{0}" eventPort="{1}"/>'.format(self.quantity,
                                                                         self.eventPort)

class DataOutput(LEMSBase):
    """
    Generic data output specification class.
    """

    def __init__(self):
        """
        Constuctor.
        """

        pass

class DataDisplay(DataOutput):
    """
    Stores specification for a data display.
    """

    def __init__(self, title, data_region):
        """
        Constuctor.

        See instance variable documentation for information on parameters.
        """

        DataOutput.__init__(self)

        self.title = title
        """ Title for the display.
        @type: string """

        self.data_region = data_region
        """ Display position
        @type: string """

        self.time_scale = 1
        """ Time scale
        @type: Number """
        
    def toxml(self):
        """
        Exports this object into a LEMS XML object
        """

        return '<DataDisplay title="{0}" dataRegion="{1}"/>'.format(self.title,
                                                                    self.data_region)

class DataWriter(DataOutput):
    """
    Stores specification for a data writer.
    """

    def __init__(self, path, file_name):
        """
        Constuctor.

        See instance variable documentation for information on parameters.
        """

        DataOutput.__init__(self)
        
        self.path = path
        """ Path to the quantity to be saved to file.
        @type: string """

        self.file_name = file_name
        """ Text parameter to be used for the file name
        @type: string """


    def toxml(self):
        """
        Exports this object into a LEMS XML object
        """

        return '<DataWriter path="{0}" fileName="{1}"/>'.format(self.path,
                                                                self.file_name)
                                                                
    def __str__(self):
        return 'DataWriter, path: {0}, fileName: {1}'.format(self.path, self.file_name)
    
    
class EventWriter(DataOutput):
    """
    Stores specification for an event writer.
    """

    def __init__(self, path, file_name, format):
        """
        Constuctor.

        See instance variable documentation for information on parameters.
        """

        DataOutput.__init__(self)
        
        self.path = path
        """ Path to the quantity to be saved to file.
        @type: string """

        self.file_name = file_name
        """ Text parameter to be used for the file name
        @type: string """

        self.format = format
        """ Text parameter to be used for the format
        @type: string """


    def toxml(self):
        """
        Exports this object into a LEMS XML object
        """

        return '<EventWriter path="{0}" fileName="{1}" format="{2}"/>'.format(self.path,
                                                                self.file_name, self.format)
                                                                
    def __str__(self):
        return 'EventWriter, path: {0}, fileName: {1}, format: {2}'.format(self.path, self.file_name, self.format)



class Simulation(LEMSBase):
    """
    Stores the simulation-related attributes of a component-type.
    """

    def __init__(self):
        """
        Constructor.
        """

        self.runs = Map()
        """ Map of runs in this dynamics regime.
        @type: Map(string -> lems.model.simulation.Run) """

        self.records = Map()
        """ Map of recorded variables in this dynamics regime.
        @type: Map(string -> lems.model.simulation.Record """

        self.event_records = Map()
        """ Map of recorded events in this dynamics regime.
        @type: Map(string -> lems.model.simulation.EventRecord """

        self.data_displays = Map()
        """ Map of data displays mapping titles to regions.
        @type: Map(string -> string) """

        self.data_writers = Map()
        """ Map of recorded variables to data writers.
        @type: Map(string -> lems.model.simulation.DataWriter """

        self.event_writers = Map()
        """ Map of recorded variables to event writers.
        @type: Map(string -> lems.model.simulation.EventWriter """

    def add_run(self, run):
        """
        Adds a runnable target component definition to the list of runnable
        components stored in this context.

        @param run: Run specification
        @type run: lems.model.simulation.Run
        """

        self.runs[run.component] = run

    def add_record(self, record):
        """
        Adds a record object to the list of record objects in this dynamics
        regime.

        @param record: Record object to be added.
        @type record: lems.model.simulation.Record
        """

        self.records[record.quantity] = record

    def add_event_record(self, event_record):
        """
        Adds an eventrecord object to the list of event_record objects in this dynamics
        regime.

        @param event_record: EventRecord object to be added.
        @type event_record: lems.model.simulation.EventRecord
        """

        self.event_records[event_record.quantity] = event_record

    def add_data_display(self, data_display):
        """
        Adds a data display to this simulation section.

        @param data_display: Data display to be added.
        @type data_display: lems.model.simulation.DataDisplay
        """

        self.data_displays[data_display.title] = data_display

    def add_data_writer(self, data_writer):
        """
        Adds a data writer to this simulation section.

        @param data_writer: Data writer to be added.
        @type data_writer: lems.model.simulation.DataWriter
        """

        self.data_writers[data_writer.path] = data_writer

    def add_event_writer(self, event_writer):
        """
        Adds an event writer to this simulation section.

        @param event_writer: event writer to be added.
        @type event_writer: lems.model.simulation.EventWriter
        """

        self.event_writers[event_writer.path] = event_writer

    def add(self, child):
        """
        Adds a typed child object to the simulation spec.

        @param child: Child object to be added.
        """

        if isinstance(child, Run):
            self.add_run(child)
        elif isinstance(child, Record):
            self.add_record(child)
        elif isinstance(child, EventRecord):
            self.add_event_record(child)
        elif isinstance(child, DataDisplay):
            self.add_data_display(child)
        elif isinstance(child, DataWriter):
            self.add_data_writer(child)
        elif isinstance(child, EventWriter):
            self.add_event_writer(child)
        else:
            raise ModelError('Unsupported child element')
        
    def toxml(self):
        """
        Exports this object into a LEMS XML object
        """

        chxmlstr = ''

        for run in self.runs:
            chxmlstr += run.toxml()

        for record in self.records:
            chxmlstr += record.toxml()

        for event_record in self.event_records:
            chxmlstr += event_record.toxml()

        for data_display in self.data_displays:
            chxmlstr += data_display.toxml()

        for data_writer in self.data_writers:
            chxmlstr += data_writer.toxml()

        for event_writer in self.event_writers:
            chxmlstr += event_writer.toxml()

        if chxmlstr:
            xmlstr = '<Simulation>' + chxmlstr + '</Simulation>'
        else:
            xmlstr = ''

        return xmlstr
