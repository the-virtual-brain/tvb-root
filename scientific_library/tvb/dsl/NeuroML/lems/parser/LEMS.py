"""
LEMS XML file format parser.

@author: Gautham Ganapathy
@organization: LEMS (http://neuroml.org/lems/, https://github.com/organizations/LEMS)
@contact: gautham@lisphacker.org

MAvdVlag: altered attributes for constants, state_variables, derived_variables, time_derivatives,
conditional_derived_variable and exposures
"""

import xml.etree.ElementTree as xe

from ..base.base import LEMSBase
# from lems.model.fundamental import *
from ..model.component import *
# from lems.model.dynamics import *
# from lems.model.structure import *
# from lems.model.simulation import *

# from lems.base.util import make_id

from ..model.dynamics import *

from collections import OrderedDict


def get_nons_tag_from_node(node):
    tag = node.tag
    bits = tag.split('}')
    if len(bits) == 1:
        return tag
    elif '/lems/' in bits[0]:
        return bits[1]
    elif 'neuroml2' in bits[0]:
        return bits[1]
    elif 'rdf' in bits[0]:
        return "rdf_"+bits[1]
    elif 'model-qualifiers' in bits[0]:
        return "bqmodel_"+bits[1]
    elif 'biology-qualifiers' in bits[0]:
        return "bqbiol_"+bits[1]
    else:
        return "%s:%s"%(bits[0],bits[1])

class LEMSXMLNode:
    def __init__(self, pyxmlnode):
        self.tag = get_nons_tag_from_node(pyxmlnode)
        self.ltag = self.tag.lower()

        self.attrib = dict()
        self.lattrib = dict()

        for k in pyxmlnode.attrib:
            self.attrib[k] = pyxmlnode.attrib[k]
            self.lattrib[k.lower()] = pyxmlnode.attrib[k]

        self.children = list()
        for pyxmlchild in pyxmlnode:
            self.children.append(LEMSXMLNode(pyxmlchild))
            
    def __str__(self):
        return 'LEMSXMLNode <{0} {1}>'.format(self.tag, self.attrib)
        
class LEMSFileParser(LEMSBase):
    """
    LEMS XML file format parser class.
    """
    
    def __init__(self, model, include_dirs = [], include_includes=True):
        """
        Constructor.

        See instance variable documentation for more details on parameters.
        """

        self.model = model
        """ Model instance to be populated from the parsed file.
        @type: lems.model.model.Model """

        self.include_dirs = include_dirs
        """ List of directories to search for included files.
        @type: list(str) """

        self.tag_parse_table = OrderedDict()
        """ Dictionary of xml tags to parse methods
        @type: dict(string -> function) """

        self.valid_children = None
        """ Dictionary mapping each tag to it's list of valid child tags.
        @type: dict(string -> string) """

        self.id_counter = None
        """ Counter generator for generating unique ids.
        @type: generator(int) """
        
        self.include_includes = include_includes
        """ Whether to include LEMS definitions in <Include> elements
        @type: boolean """
        
        self.init_parser()

    def init_parser(self):
        """
        Initializes the parser
        """

        #self.token_list = None
        #self.prev_token_lists = None

        self.valid_children = dict()
        self.valid_children['lems'] = ['component', 'componenttype',
                                       'target', 'include',
                                       'dimension', 'unit', 'assertion']
                                       
        #TODO: make this generic for any domain specific language based on LEMS
        self.valid_children['neuroml'] = ['include', 'componenttype']
                                       
        self.valid_children['componenttype'] = ['dynamics',
                                                'child', 'children',
                                                'componentreference',
                                                'exposure', 'eventport',
                                                'fixed', 'link', 'parameter',
                                                'property',
                                                'indexparameter',
                                                'path', 'requirement',
                                                'componentrequirement',
                                                'instancerequirement',
                                                'simulation', 'structure',
                                                'text', 'attachments',
                                                'constant', 'derivedparameter',
                                                'function']
                                                
        self.valid_children['dynamics'] = ['derivedvariable',
                                           'conditionalderivedvariable',
                                           'oncondition',
                                           'onevent', 'onstart',
                                           'statevariable', 'timederivative',
                                           'kineticscheme', 'regime']
                                           
        self.valid_children['component'] = ['component']
                                           
        self.valid_children['conditionalderivedvariable'] = ['case']
        
        self.valid_children['regime'] = ['oncondition', 'onentry', 'timederivative']
        self.valid_children['oncondition'] = ['eventout', 'stateassignment', 'transition']
        self.valid_children['onentry'] = ['eventout', 'stateassignment', 'transition']
        self.valid_children['onevent'] = ['eventout', 'stateassignment', 'transition']
        self.valid_children['onstart'] = ['eventout', 'stateassignment', 'transition']
        self.valid_children['structure'] = ['childinstance',
                                            'eventconnection',
                                            'foreach',
                                            'multiinstantiate',
                                            'with',
                                            'tunnel']
                                       
        self.valid_children['foreach'] = ['foreach', 'eventconnection']     
                                            
        self.valid_children['simulation'] = ['record', 'eventrecord', 'run',
                                             'datadisplay', 'datawriter', 'eventwriter']

        self.tag_parse_table = dict()
        #self.tag_parse_table['assertion'] = self.parse_assertion
        self.tag_parse_table['attachments'] = self.parse_attachments
        self.tag_parse_table['child'] = self.parse_child
        self.tag_parse_table['childinstance'] = self.parse_child_instance
        self.tag_parse_table['children'] = self.parse_children
        self.tag_parse_table['component'] = self.parse_component
        self.tag_parse_table['componentreference'] = self.parse_component_reference   
        self.tag_parse_table['componentrequirement'] = self.parse_component_requirement
        self.tag_parse_table['componenttype'] = self.parse_component_type
        self.tag_parse_table['constant'] = self.parse_constant
        self.tag_parse_table['function'] = self.parse_function
        self.tag_parse_table['datadisplay'] = self.parse_data_display
        self.tag_parse_table['datawriter'] = self.parse_data_writer
        self.tag_parse_table['eventwriter'] = self.parse_event_writer
        self.tag_parse_table['derivedparameter'] = self.parse_derived_parameter
        self.tag_parse_table['derivedvariable'] = self.parse_derived_variable
        self.tag_parse_table['conditionalderivedvariable'] = self.parse_conditional_derived_variable
        self.tag_parse_table['case'] = self.parse_case
        self.tag_parse_table['dimension'] = self.parse_dimension
        self.tag_parse_table['dynamics'] = self.parse_dynamics
        self.tag_parse_table['eventconnection'] = self.parse_event_connection
        self.tag_parse_table['eventout'] = self.parse_event_out
        self.tag_parse_table['eventport'] = self.parse_event_port
        self.tag_parse_table['exposure'] = self.parse_exposure
        self.tag_parse_table['fixed'] = self.parse_fixed
        self.tag_parse_table['foreach'] = self.parse_for_each
        self.tag_parse_table['include'] = self.parse_include
        self.tag_parse_table['indexparameter'] = self.parse_index_parameter
        self.tag_parse_table['kineticscheme'] = self.parse_kinetic_scheme
        self.tag_parse_table['link'] = self.parse_link
        self.tag_parse_table['multiinstantiate'] = self.parse_multi_instantiate
        self.tag_parse_table['oncondition'] = self.parse_on_condition
        self.tag_parse_table['onentry'] = self.parse_on_entry
        self.tag_parse_table['onevent'] = self.parse_on_event
        self.tag_parse_table['onstart'] = self.parse_on_start
        self.tag_parse_table['parameter'] = self.parse_parameter
        self.tag_parse_table['property'] = self.parse_property
        self.tag_parse_table['path'] = self.parse_path
        self.tag_parse_table['record'] = self.parse_record
        self.tag_parse_table['eventrecord'] = self.parse_event_record
        self.tag_parse_table['regime'] = self.parse_regime
        self.tag_parse_table['requirement'] = self.parse_requirement
        self.tag_parse_table['instancerequirement'] = self.parse_instance_requirement     
        self.tag_parse_table['run'] = self.parse_run
        #self.tag_parse_table['show'] = self.parse_show
        self.tag_parse_table['simulation'] = self.parse_simulation
        self.tag_parse_table['stateassignment'] = self.parse_state_assignment
        self.tag_parse_table['statevariable'] = self.parse_state_variable
        self.tag_parse_table['structure'] = self.parse_structure
        self.tag_parse_table['target'] = self.parse_target
        self.tag_parse_table['text'] = self.parse_text
        self.tag_parse_table['timederivative'] = self.parse_time_derivative
        self.tag_parse_table['transition'] = self.parse_transition
        self.tag_parse_table['tunnel'] = self.parse_tunnel
        self.tag_parse_table['unit'] = self.parse_unit
        self.tag_parse_table['with'] = self.parse_with

        self.xml_node_stack = []

        self.current_component_type = None
        self.current_dynamics = None
        self.current_regime = None
        self.current_event_handler = None
        self.current_structure = None
        self.current_simulation = None
        self.current_component = None
        
        def counter():
            count = 1
            while True:
                yield count
                count = count + 1

        self.id_counter = counter()

        
    def process_nested_tags(self, node, tag = ''):
        """
        Process child tags.

        @param node: Current node being parsed.
        @type node: xml.etree.Element

        @raise ParseError: Raised when an unexpected nested tag is found.
        """
        ##print("---------Processing: %s, %s"%(node.tag,tag))

        if tag == '':
            t = node.ltag
        else:
            t = tag.lower()
        
        for child in node.children:
            self.xml_node_stack = [child] + self.xml_node_stack

            ctagl = child.ltag

            if ctagl in self.tag_parse_table and ctagl in self.valid_children[t]:
                #print("Processing known type: %s"%ctagl)
                self.tag_parse_table[ctagl](child)
            else:
                #print("Processing unknown type: %s"%ctagl)
                self.parse_component_by_typename(child, child.tag)

            self.xml_node_stack = self.xml_node_stack[1:]

    def parse(self, xmltext):
        """
        Parse a string containing LEMS XML text.

        @param xmltext: String containing LEMS XML formatted text.
        @type xmltext: str
        """
        
        xml = LEMSXMLNode(xe.XML(xmltext))

        if xml.ltag != 'lems' and xml.ltag != 'neuroml':
            raise ParseError('<Lems> expected as root element (or even <neuroml>), found: {0}'.format(xml.ltag))
        '''
        if xml.ltag == 'lems':
            if 'description' in xml.lattrib:
                self.description = xml.lattrib['description']
        '''

        self.process_nested_tags(xml)


    def raise_error(self, message, *params, **key_params):
        """
        Raise a parse error.
        """
        
        s = 'Parser error in '

        self.xml_node_stack.reverse()
        if len(self.xml_node_stack) > 1:
            node = self.xml_node_stack[0]
            s += '<{0}'.format(node.tag)
            if 'name' in node.lattrib:
                s += ' name=\"{0}\"'.format(node.lattrib['name'])
            if 'id' in node.lattrib:
                s += ' id=\"{0}\"'.format(node.lattrib['id'])
            s += '>'

        for node in self.xml_node_stack[1:]:
            s += '.<{0}'.format(node.tag)
            if 'name' in node.lattrib:
                s += ' name=\"{0}\"'.format(node.lattrib['name'])
            if 'id' in node.lattrib:
                s += ' id=\"{0}\"'.format(node.lattrib['id'])
            s += '>'

        s += ':\n  ' + message

        raise ParseError(s, *params, **key_params)

        self.xml_node_stack.reverse()


#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################





    def parse_assertion(self, node):
        """
        Parses <Assertion>

        @param node: Node containing the <Assertion> element
        @type node: xml.etree.Element
        """

        print('TODO - <Assertion>')


    def parse_attachments(self, node):
        """
        Parses <Attachments>

        @param node: Node containing the <Attachments> element
        @type node: xml.etree.Element
        """

        if 'name' in node.lattrib:
            name = node.lattrib['name']
        else:
            self.raise_error('<Attachments> must specify a name.')

        if 'type' in node.lattrib:
            type_ = node.lattrib['type']
        else:
            self.raise_error("Attachment '{0}' must specify a type.",
                             name)

        description = node.lattrib.get('description', '')
        self.current_component_type.add_attachments(Attachments(name, type_, description))

    def parse_child(self, node):
        """
        Parses <Child>

        @param node: Node containing the <Child> element
        @type node: xml.etree.Element
        """

        if 'name' in node.lattrib:
            name = node.lattrib['name']
        else:
            self.raise_error('<Child> must specify a name.')

        if 'type' in node.lattrib:
            type_ = node.lattrib['type']
        else:
            self.raise_error("Child '{0}' must specify a type.", name)

        self.current_component_type.add_children(Children(name, type_, False))

    def parse_child_instance(self, node):
        """
        Parses <ChildInstance>

        @param node: Node containing the <ChildInstance> element
        @type node: xml.etree.Element
        """

        if 'component' in node.lattrib:
            component = node.lattrib['component']
        else:
            self.raise_error('<ChildInstance> must specify a component reference')

        self.current_structure.add_child_instance(ChildInstance(component))

    def parse_children(self, node):
        """
        Parses <Children>

        @param node: Node containing the <Children> element
        @type node: xml.etree.Element
        """

        if 'name' in node.lattrib:
            name = node.lattrib['name']
        else:
            self.raise_error('<Children> must specify a name.')

        if 'type' in node.lattrib:
            type_ = node.lattrib['type']
        else:
            self.raise_error("Children '{0}' must specify a type.", name)

        self.current_component_type.add_children(Children(name, type_, True))

    def parse_component_by_typename(self, node, type_):
        """
        Parses components defined directly by component name.

        @param node: Node containing the <Component> element
        @type node: xml.etree.Element

        @param type_: Type of this component.
        @type type_: string

        @raise ParseError: Raised when the component does not have an id.
        """
        #print('Parsing component {0} by typename {1}'.format(node, type_))
        if 'id' in node.lattrib:
            id_ = node.lattrib['id']
        else:
            #self.raise_error('Component must have an id')
            id_ = node.tag #make_id()

        if 'type' in node.lattrib:
            type_ = node.lattrib['type']
        else:
            type_ = node.tag

        component = Component(id_, type_)

        if self.current_component:
            component.set_parent_id(self.current_component.id)
            self.current_component.add_child(component)
            
        else:
            self.model.add_component(component)

        for key in node.attrib:
            if key.lower() not in ['id', 'type']:
                component.set_parameter(key, node.attrib[key])

        old_component = self.current_component
        self.current_component = component
        self.process_nested_tags(node, 'component')
        self.current_component = old_component

    def parse_component(self, node):
        """
        Parses <Component>

        @param node: Node containing the <Component> element
        @type node: xml.etree.Element
        """

        if 'id' in node.lattrib:
            id_ = node.lattrib['id']
        else:
            #self.raise_error('Component must have an id')
            id_ = make_id()

        if 'type' in node.lattrib:
            type_ = node.lattrib['type']
        else:
                self.raise_error("Component {0} must have a type.",
                                 id_)

        component = Component(id_, type_)

        if self.current_component:
            component.set_parent_id(self.current_component.id)
            self.current_component.add_child(component)
        else:
            self.model.add_component(component)
                
        for key in node.attrib:
            if key.lower() not in ['id', 'type']:
                component.set_parameter(key, node.attrib[key])

        old_component = self.current_component
        self.current_component = component
        self.process_nested_tags(node)
        self.current_component = old_component

    def parse_component_reference(self, node):
        """
        Parses <ComponentReference>

        @param node: Node containing the <ComponentTypeRef> element
        @type node: xml.etree.Element
        """

        if 'name' in node.lattrib:
            name = node.lattrib['name']
        else:
            self.raise_error('<ComponentReference> must specify a name for the ' +
                             'reference.')

        if 'type' in node.lattrib:
            type_ = node.lattrib['type']
        else:
            self.raise_error('<ComponentReference> must specify a type for the ' +
                             'reference.')
                             
        if 'local' in node.lattrib:
            local = node.lattrib['local']
        else:
            local = None

        self.current_component_type.add_component_reference(ComponentReference(name, type_, local))

    def parse_component_type(self, node):
        """
        Parses <ComponentType>

        @param node: Node containing the <ComponentType> element
        @type node: xml.etree.Element

        @raise ParseError: Raised when the component type does not have a
        name.
        """

        try:
            name = node.lattrib['name']
        except:
            self.raise_error('<ComponentType> must specify a name')

        if 'extends' in node.lattrib:
            extends = node.lattrib['extends']
        else:
            extends = None

        if 'description' in node.lattrib:
            description = node.lattrib['description']
        else:
            description = ''

        component_type = ComponentType(name, description, extends)
        self.model.add_component_type(component_type)

        self.current_component_type = component_type
        self.process_nested_tags(node)
        self.current_component_type = None

    def parse_constant(self, node):
        """
        Parses <Constant>

        @param node: Node containing the <Constant> element
        @type node: xml.etree.Element

        MV: fixed the symbol part. It was not there for constant parsing
        """

        try:
            name = node.lattrib['name']
        except:
            self.raise_error('<Constant> must specify a name.')

        domain = node.lattrib.get('domain', None)
        symbol = node.lattrib.get('symbol', None)

        try:
            default = node.lattrib['default']
        except:
            self.raise_error("Constant '{0}' must have a value.", name)

        description = node.lattrib.get('description', '')

        constant = Constant(name, default, domain, symbol, description)

        if self.current_component_type:
            self.current_component_type.add_constant(constant)
        else:
            self.model.add_constant(constant)

    def parse_function(self, node):
        """
        Parses <Function>

        @param node: Node containing the <Function> element
        @type node: xml.etree.Element

        MV: added function for pre and post coupling behavior
        """
        try:
            name = node.lattrib['name']
        except:
            self.raise_error('<Function> must specify a name.')

        try:
            value = node.lattrib['value']
        except:
            self.raise_error("Function '{0}' must have a value.", name)

        dimension = None
        symbol = None
        description = None

        function = Function(name, value, dimension, symbol, description)

        if self.current_component_type:
            self.current_component_type.add_function(function)
        else:
            self.model.add_function(function)


    def parse_data_display(self, node):
        """
        Parses <DataDisplay>

        @param node: Node containing the <DataDisplay> element
        @type node: xml.etree.Element
        """

        if 'title' in node.lattrib:
            title = node.lattrib['title']
        else:
            self.raise_error('<DataDisplay> must have a title.')

        if 'dataregion' in node.lattrib:
            data_region = node.lattrib['dataregion']
        else:
            data_region = None

        self.current_simulation.add_data_display(DataDisplay(title, data_region))

    def parse_data_writer(self, node):
        """
        Parses <DataWriter>

        @param node: Node containing the <DataWriter> element
        @type node: xml.etree.Element
        """

        if 'path' in node.lattrib:
            path = node.lattrib['path']
        else:
            self.raise_error('<DataWriter> must specify a path.')

        if 'filename' in node.lattrib:
            file_path = node.lattrib['filename']
        else:
            self.raise_error("Data writer for '{0}' must specify a filename.",
                             path)

        self.current_simulation.add_data_writer(DataWriter(path, file_path))

    def parse_event_writer(self, node):
        """
        Parses <EventWriter>

        @param node: Node containing the <EventWriter> element
        @type node: xml.etree.Element
        """

        if 'path' in node.lattrib:
            path = node.lattrib['path']
        else:
            self.raise_error('<EventWriter> must specify a path.')

        if 'filename' in node.lattrib:
            file_path = node.lattrib['filename']
        else:
            self.raise_error("Event writer for '{0}' must specify a filename.",
                             path)
                             
        if 'format' in node.lattrib:
            format = node.lattrib['format']
        else:
            self.raise_error("Event writer for '{0}' must specify a format.",
                             path)

        self.current_simulation.add_event_writer(EventWriter(path, file_path, format))

    def parse_derived_parameter(self, node):
        """
        Parses <DerivedParameter>

        @param node: Node containing the <DerivedParameter> element
        @type node: xml.etree.Element
        """

        #if self.current_context.context_type != Context.COMPONENT_TYPE:
        #    self.raise_error('Dynamics must be defined inside a ' +
        #                     'component type')

        if 'name' in node.lattrib:
            name = node.lattrib['name']
        else:
            self.raise_error('A derived parameter must have a name')

        if 'dimension' in node.lattrib:
            dimension = node.lattrib['dimension']
        else:
            dimension = None

        if 'value' in node.lattrib:
            value = node.lattrib['value']
        else:
            value = None

        if 'select' in node.lattrib:
            select = node.lattrib['select']
        else:
            select = None

        self.current_component_type.add_derived_parameter(DerivedParameter(name, value,
                                                                    dimension, select))

    def parse_derived_variable(self, node):
        """
        Parses <DerivedVariable>

        @param node: Node containing the <DerivedVariable> element
        @type node: xml.etree.Element

        @raise ParseError: Raised when no name of specified for the derived variable.
        """

        if 'name' in node.lattrib:
            name = node.lattrib['name']
        elif 'exposure' in node.lattrib:
            name = node.lattrib['exposure']
        else:
            self.raise_error('<DerivedVariable> must specify a name')

        params = dict()
        for attr_name in ['dimension', 'exposure', 'select', 'expression', 'reduce', 'required']:
            if attr_name in node.lattrib:
                params[attr_name] = node.lattrib[attr_name]

        self.current_regime.add_derived_variable(DerivedVariable(name, **params))
        
        
    def parse_conditional_derived_variable(self, node):
        """
        Parses <ConditionalDerivedVariable>

        @param node: Node containing the <ConditionalDerivedVariable> element
        @type node: xml.etree.Element

        @raise ParseError: Raised when no name or value is specified for the conditional derived variable.
        """

        if 'name' in node.lattrib:
            name = node.lattrib['name']
        elif 'exposure' in node.lattrib:
            name = node.lattrib['exposure']
        else:
            self.raise_error('<ConditionalDerivedVariable> must specify a name')
            
        if 'exposure' in node.lattrib:
            exposure = node.lattrib['exposure']
        else:
            exposure = None
            
        if 'condition' in node.lattrib:
            condition = node.lattrib['condition']
        else:
            condition = None

        if 'cases' in node.lattrib:
            cases = node.lattrib['cases']
        else:
            cases = None

        conditional_derived_variable = ConditionalDerivedVariable(name, condition, exposure, cases)
        
        self.current_regime.add_conditional_derived_variable(conditional_derived_variable)
        
        self.current_conditional_derived_variable = conditional_derived_variable
        
        self.process_nested_tags(node)
        
        
    def parse_case(self, node):
        """
        Parses <Case>

        @param node: Node containing the <Case> element
        @type node: xml.etree.Element

        @raise ParseError: When no condition or value is specified
        """
        
        try:
            condition = node.lattrib['condition']
        except:
            condition = None
            
        try:
            value = node.lattrib['value']
        except:
            self.raise_error('<Case> must specify a value')

        self.current_conditional_derived_variable.add_case(Case(condition, value))

    def parse_dimension(self, node):
        """
        Parses <Dimension>

        @param node: Node containing the <Dimension> element
        @type node: xml.etree.Element

        @raise ParseError: When the name is not a string or if the
        dimension is not a signed integer.
        """

        try:
            name = node.lattrib['name']
        except:
            self.raise_error('<Dimension> must specify a name')

        description = node.lattrib.get('description', '')

        dim = dict()
        for d in ['l', 'm', 't', 'i', 'k', 'c', 'n']:
            dim[d] = int(node.lattrib.get(d, 0))

        self.model.add_dimension(Dimension(name, description, **dim))

    def parse_dynamics(self, node):
        """
        Parses <Dynamics>

        @param node: Node containing the <Behaviour> element
        @type node: xml.etree.Element
        """

        self.current_dynamics = self.current_component_type.dynamics
        self.current_regime = self.current_dynamics
        self.process_nested_tags(node)
        self.current_regime = None
        self.current_dynamics = None

    def parse_event_connection(self, node):
        """
        Parses <EventConnection>

        @param node: Node containing the <EventConnection> element
        @type node: xml.etree.Element
        """

        if 'from' in node.lattrib:
            from_ = node.lattrib['from']
        else:
            self.raise_error('<EventConnection> must provide a source (from) component reference.')

        if 'to' in node.lattrib:
            to = node.lattrib['to']
        else:
            self.raise_error('<EventConnection> must provide a target (to) component reference.')

        source_port = node.lattrib.get('sourceport', '')
        target_port = node.lattrib.get('targetport', '')
        receiver = node.lattrib.get('receiver', '')
        receiver_container = node.lattrib.get('receivercontainer', '')

        ec = EventConnection(from_, to, source_port, target_port, receiver, receiver_container)
        self.current_structure.add_event_connection(ec)

    def parse_event_out(self, node):
        """
        Parses <EventOut>

        @param node: Node containing the <EventOut> element
        @type node: xml.etree.Element
        """

        try:
            port = node.lattrib['port']
        except:
            self.raise_error('<EventOut> must be specify a port.')

        action = EventOut(port)

        self.current_event_handler.add_action(action)

    def parse_event_port(self, node):
        """
        Parses <EventPort>

        @param node: Node containing the <EventPort> element
        @type node: xml.etree.Element
        """

        if 'name' in node.lattrib:
            name = node.lattrib['name']
        else:
            self.raise_error(('<EventPort> must specify a name.'))

        if 'direction' in node.lattrib:
            direction = node.lattrib['direction']
        else:
            self.raise_error("Event port '{0}' must specify a direction.")

        direction = direction.lower()
        if direction != 'in' and direction != 'out':
            self.raise_error(('Event port direction must be \'in\' '
                              'or \'out\''))

        description = node.lattrib.get('description', '')
        
        self.current_component_type.add_event_port(EventPort(name, direction, description))

    def parse_exposure(self, node):
        """
        Parses <Exposure>

        @param node: Node containing the <Exposure> element
        @type node: xml.etree.Element

        @raise ParseError: Raised when the exposure name is not
        being defined in the context of a component type.
        """

        if self.current_component_type == None:
            self.raise_error('Exposures must be defined in a component type')

        try:
            name = node.lattrib['name']
        except:
            self.raise_error('<Exposure> must specify a name')

        try:
            choices = node.lattrib['choices']
        except:
            self.raise_error("Exposure '{0}' must specify choices",
                             name)

        try:
            default = node.lattrib['default']
        except:
            self.raise_error("Exposure '{0}' must specify default",
                             name)

        description = node.lattrib.get('description', '')

        self.current_component_type.add_exposure(Exposure(name, choices, default, description))

    def parse_fixed(self, node):
        """
        Parses <Fixed>

        @param node: Node containing the <Fixed> element
        @type node: xml.etree.Element
        """

        try:
            parameter = node.lattrib['parameter']
        except:
            self.raise_error('<Fixed> must specify a parameter to be fixed.')

        try:
            value = node.lattrib['value']
        except:
            self.raise_error("Fixed parameter '{0}'must specify a value.", parameter)

        description = node.lattrib.get('description', '')
        
        self.current_component_type.add_parameter(Fixed(parameter, value, description))

    def parse_for_each(self, node):
        """
        Parses <ForEach>

        @param node: Node containing the <ForEach> element
        @type node: xml.etree.Element
        """
        
        if self.current_structure == None:
            self.raise_error('<ForEach> can only be made within ' +
                             'a structure definition')

        if 'instances' in node.lattrib:
            instances = node.lattrib['instances']
        else:
            self.raise_error('<ForEach> must specify a reference to target'
                             'instances')

        if 'as' in node.lattrib:
            as_ = node.lattrib['as']
        else:
            self.raise_error('<ForEach> must specify a name for the '
                             'enumerated target instances')

        old_structure = self.current_structure
        fe = ForEach(instances, as_)
        self.current_structure.add_for_each(fe)
        self.current_structure = fe
        
        self.process_nested_tags(node)

        self.current_structure = old_structure

    def parse_include(self, node):
        """
        Parses <Include>

        @param node: Node containing the <Include> element
        @type node: xml.etree.Element

        @raise ParseError: Raised when the file to be included is not specified. 
        """
        if not self.include_includes:
            if self.model.debug: print("Ignoring included LEMS file: %s"%node.lattrib['file'])
        else:

            #TODO: remove this hard coding for reading NeuroML includes...
            if 'file' not in node.lattrib:
                if 'href' in node.lattrib:
                    self.model.include_file(node.lattrib['href'], self.include_dirs)
                    return
                else:
                    self.raise_error('<Include> must specify the file to be included.')

            self.model.include_file(node.lattrib['file'], self.include_dirs)

    def parse_kinetic_scheme(self, node):
        """
        Parses <KineticScheme>

        @param node: Node containing the <KineticScheme> element
        @type node: xml.etree.Element
        """

        if 'name' in node.lattrib:
            name = node.lattrib['name']
        else:
            self.raise_error('<KineticScheme> must specify a name.')

        if 'nodes' in node.lattrib:
            nodes = node.lattrib['nodes']
        else:
            self.raise_error("Kinetic scheme '{0}' must specify nodes.", name)

        if 'statevariable' in node.lattrib:
            state_variable = node.lattrib['statevariable']
        else:
            self.raise_error("Kinetic scheme '{0}' must specify a state variable.", name)

        if 'edges' in node.lattrib:
            edges = node.lattrib['edges']
        else:
            self.raise_error("Kinetic scheme '{0}' must specify edges.", name)

        if 'edgesource' in node.lattrib:
            edge_source = node.lattrib['edgesource']
        else:
            self.raise_error("Kinetic scheme '{0}' must specify the edge source attribute.", name)

        if 'edgetarget' in node.lattrib:
            edge_target = node.lattrib['edgetarget']
        else:
            self.raise_error("Kinetic scheme '{0}' must specify the edge target attribute.", name)

        if 'forwardrate' in node.lattrib:
            forward_rate = node.lattrib['forwardrate']
        else:
            self.raise_error("Kinetic scheme '{0}' must specify the forward rate attribute.", name)

        if 'reverserate' in node.lattrib:
            reverse_rate = node.lattrib['reverserate']
        else:
            self.raise_error("Kinetic scheme '{0}' must specify the reverse rate attribute", name)

        self.current_regime.add_kinetic_scheme(KineticScheme(name, nodes, state_variable,
                                                             edges, edge_source, edge_target,
                                                             forward_rate, reverse_rate))

    def parse_link(self, node):
        """
        Parses <Link>

        @param node: Node containing the <Link> element
        @type node: xml.etree.Element
        """

        if 'name' in node.lattrib:
            name = node.lattrib['name']
        else:
            self.raise_error('<Link> must specify a name')

        if 'type' in node.lattrib:
            type_ = node.lattrib['type']
        else:
            self.raise_error("Link '{0}' must specify a type", name)

        description = node.lattrib.get('description', '')

        self.current_component_type.add_link(Link(name, type_, description))

    def parse_multi_instantiate(self, node):
        """
        Parses <MultiInstantiate>

        @param node: Node containing the <MultiInstantiate> element
        @type node: xml.etree.Element
        """

        if 'component' in node.lattrib:
            component = node.lattrib['component']
        else:
            self.raise_error('<MultiInstantiate> must specify a component reference.')

        if 'number' in node.lattrib:
            number = node.lattrib['number']
        else:
            self.raise_error("Multi instantiation of '{0}' must specify a parameter specifying the number.",
                             component)

        self.current_structure.add_multi_instantiate(MultiInstantiate(component, number))

    def parse_on_condition(self, node):
        """
        Parses <OnCondition>

        @param node: Node containing the <OnCondition> element
        @type node: xml.etree.Element
        """

        try:
            test = node.lattrib['test']
        except:
            self.raise_error('<OnCondition> must specify a test.')
            
        event_handler = OnCondition(test)

        self.current_regime.add_event_handler(event_handler)

        self.current_event_handler = event_handler
        self.process_nested_tags(node)
        self.current_event_handler = None
        
    def parse_on_entry(self, node):
        """
        Parses <OnEntry>

        @param node: Node containing the <OnEntry> element
        @type node: xml.etree.Element
        """

        event_handler = OnEntry()

        self.current_event_handler = event_handler
        self.current_regime.add_event_handler(event_handler)

        self.process_nested_tags(node)

        self.current_event_handler = None

    def parse_on_event(self, node):
        """
        Parses <OnEvent>

        @param node: Node containing the <OnEvent> element
        @type node: xml.etree.Element
        """

        try:
            port = node.lattrib['port']
        except:
            self.raise_error('<OnEvent> must specify a port.')
            
        event_handler = OnEvent(port)

        self.current_regime.add_event_handler(event_handler)

        self.current_event_handler = event_handler
        self.process_nested_tags(node)
        self.current_event_handler = None

    def parse_on_start(self, node):
        """
        Parses <OnStart>

        @param node: Node containing the <OnStart> element
        @type node: xml.etree.Element
        """

        event_handler = OnStart()

        self.current_regime.add_event_handler(event_handler)

        self.current_event_handler = event_handler
        self.process_nested_tags(node)
        self.current_event_handler = None

    def parse_parameter(self, node):
        """
        Parses <Parameter>

        @param node: Node containing the <Parameter> element
        @type node: xml.etree.Element

        @raise ParseError: Raised when the parameter does not have a name.
        @raise ParseError: Raised when the parameter does not have a
        dimension.
        """

        if self.current_component_type == None:
            self.raise_error('Parameters can only be defined in ' +
                             'a component type')

        try:
            name = node.lattrib['name']
        except:
            self.raise_error('<Parameter> must specify a name')

        try:
            dimension = node.lattrib['dimension']
        except:
            self.raise_error("Parameter '{0}' has no dimension",
                             name)

        parameter = Parameter(name, dimension)

        self.current_component_type.add_parameter(parameter)

    def parse_property(self, node):
        """
        Parses <Property>

        @param node: Node containing the <Property> element
        @type node: xml.etree.Element

        @raise ParseError: Raised when the property does not have a name.
        @raise ParseError: Raised when the property does not have a
        dimension.
        """

        if self.current_component_type == None:
            self.raise_error('Property can only be defined in ' +
                             'a component type')

        try:
            name = node.lattrib['name']
        except:
            self.raise_error('<Property> must specify a name')

        try:
            dimension = node.lattrib['dimension']
        except:
            self.raise_error("Property '{0}' has no dimension",
                             name)
                             
        default_value = node.lattrib.get('defaultvalue', None)
        
        property = Property(name, dimension, default_value=default_value)

        self.current_component_type.add_property(property)
        
        
    def parse_index_parameter(self, node):
        """
        Parses <IndexParameter>

        @param node: Node containing the <IndexParameter> element
        @type node: xml.etree.Element

        @raise ParseError: Raised when the IndexParameter does not have a name.
        """

        if self.current_component_type == None:
            self.raise_error('IndexParameters can only be defined in ' +
                             'a component type')

        try:
            name = node.lattrib['name']
        except:
            self.raise_error('<IndexParameter> must specify a name')


        index_parameter = IndexParameter(name)

        self.current_component_type.add_index_parameter(index_parameter)
        
        
    def parse_tunnel(self, node):
        """
        Parses <Tunnel>

        @param node: Node containing the <Tunnel> element
        @type node: xml.etree.Element

        @raise ParseError: Raised when the Tunnel does not have a name.
        """

        try:
            name = node.lattrib['name']
        except:
            self.raise_error('<Tunnel> must specify a name')
        try:
            end_a = node.lattrib['enda']
        except:
            self.raise_error('<Tunnel> must specify: endA')
        try:
            end_b = node.lattrib['enda']
        except:
            self.raise_error('<Tunnel> must specify: endB')
        try:
            component_a = node.lattrib['componenta']
        except:
            self.raise_error('<Tunnel> must specify: componentA')
        try:
            component_b = node.lattrib['componentb']
        except:
            self.raise_error('<Tunnel> must specify: componentB')


        tunnel = Tunnel(name, end_a, end_b, component_a, component_b)

        self.current_structure.add_tunnel(tunnel)
        

    def parse_path(self, node):
        """
        Parses <Path>

        @param node: Node containing the <Path> element
        @type node: xml.etree.Element
        """

        if 'name' in node.lattrib:
            name = node.lattrib['name']
        else:
            self.raise_error('<Path> must specify a name.')

        description = node.lattrib.get('description', '')

        self.current_component_type.add_path(Path(name, description))

    def parse_record(self, node):
        """
        Parses <Record>

        @param node: Node containing the <Record> element
        @type node: xml.etree.Element
        """

        if self.current_simulation == None:
            self.raise_error('<Record> must be only be used inside a ' +
                             'simulation specification')

        if 'quantity' in node.lattrib:
            quantity = node.lattrib['quantity']
        else:
            self.raise_error('<Record> must specify a quantity.')

        scale = node.lattrib.get('scale', None)
        color  = node.lattrib.get('color', None)
        id  = node.lattrib.get('id', None)

        self.current_simulation.add_record(Record(quantity, scale, color, id))

    def parse_event_record(self, node):
        """
        Parses <EventRecord>

        @param node: Node containing the <EventRecord> element
        @type node: xml.etree.Element
        """

        if self.current_simulation == None:
            self.raise_error('<EventRecord> must be only be used inside a ' +
                             'simulation specification')

        if 'quantity' in node.lattrib:
            quantity = node.lattrib['quantity']
        else:
            self.raise_error('<EventRecord> must specify a quantity.')

        if 'eventport' in node.lattrib:
            eventPort = node.lattrib['eventport']
        else:
            self.raise_error('<EventRecord> must specify an eventPort.')


        self.current_simulation.add_event_record(EventRecord(quantity, eventPort))

    def parse_regime(self, node):
        """
        Parses <Regime>

        @param node: Node containing the <Behaviour> element
        @type node: xml.etree.Element
        """

        if 'name' in node.lattrib:
            name = node.lattrib['name']
        else:
            name = ''

        if 'initial' in node.lattrib:
            initial = (node.lattrib['initial'].strip().lower() == 'true')
        else:
            initial = False

        regime = Regime(name, self.current_dynamics, initial)
        old_regime = self.current_regime
        self.current_dynamics.add_regime(regime)
        self.current_regime = regime

        self.process_nested_tags(node)

        self.current_regime = old_regime

    def parse_requirement(self, node):
        """
        Parses <Requirement>

        @param node: Node containing the <Requirement> element
        @type node: xml.etree.Element
        """

        if 'name' in node.lattrib:
            name = node.lattrib['name']
        else:
            self.raise_error('<Requirement> must specify a name')

        if 'dimension' in node.lattrib:
            dimension = node.lattrib['dimension']
        else:
            self.raise_error("Requirement \{0}' must specify a dimension.", name)

        self.current_component_type.add_requirement(Requirement(name, dimension))
    
    def parse_component_requirement(self, node):
        """
        Parses <ComponentRequirement>

        @param node: Node containing the <ComponentRequirement> element
        @type node: xml.etree.Element
        """

        if 'name' in node.lattrib:
            name = node.lattrib['name']
        else:
            self.raise_error('<ComponentRequirement> must specify a name')

        self.current_component_type.add_component_requirement(ComponentRequirement(name))
    
    def parse_instance_requirement(self, node):
        """
        Parses <InstanceRequirement>

        @param node: Node containing the <InstanceRequirement> element
        @type node: xml.etree.Element
        """

        if 'name' in node.lattrib:
            name = node.lattrib['name']
        else:
            self.raise_error('<InstanceRequirement> must specify a name')

        if 'type' in node.lattrib:
            type = node.lattrib['type']
        else:
            self.raise_error("InstanceRequirement \{0}' must specify a type.", name)

        self.current_component_type.add_instance_requirement(InstanceRequirement(name, type))
    
    def parse_run(self, node):
        """
        Parses <Run>

        @param node: Node containing the <Run> element
        @type node: xml.etree.Element
        """

        if 'component' in node.lattrib:
            component = node.lattrib['component']
        else:
            self.raise_error('<Run> must specify a target component')

        if 'variable' in node.lattrib:
            variable = node.lattrib['variable']
        else:
            self.raise_error('<Run> must specify a state variable')

        if 'increment' in node.lattrib:
            increment = node.lattrib['increment']
        else:
            self.raise_error('<Run> must specify an increment for the ' +
                             'state variable')

        if 'total' in node.lattrib:
            total = node.lattrib['total']
        else:
            self.raise_error('<Run> must specify a final value for the ' +
                             'state variable')

        self.current_simulation.add_run(Run(component, variable, increment, total))

    def parse_show(self, node):
        """
        Parses <Show>

        @param node: Node containing the <Show> element
        @type node: xml.etree.Element
        """

        pass

    def parse_simulation(self, node):
        """
        Parses <Simulation>

        @param node: Node containing the <Simulation> element
        @type node: xml.etree.Element
        """

        self.current_simulation = self.current_component_type.simulation

        self.process_nested_tags(node)

        self.current_simulation = None

    def parse_state_assignment(self, node):
        """
        Parses <StateAssignment>

        @param node: Node containing the <StateAssignment> element
        @type node: xml.etree.Element
        """

        if 'variable' in node.lattrib:
            variable = node.lattrib['variable']
        else:
            self.raise_error('<StateAssignment> must specify a variable name')

        if 'value' in node.lattrib:
            value = node.lattrib['value']
        else:
            self.raise_error("State assignment for '{0}' must specify a value.",
                             variable)

        action = StateAssignment(variable, value)

        self.current_event_handler.add_action(action)


    def parse_state_variable(self, node):
        """
        Parses <StateVariable>

        @param node: Node containing the <StateVariable> element
        @type node: xml.etree.Element

        @raise ParseError: Raised when the state variable is not
        being defined in the context of a component type.
        """

        if 'name' in node.lattrib:
            name = node.lattrib['name']
        else:
            self.raise_error('<StateVariable> must specify a name')

        if 'default' in node.lattrib:
            default = node.lattrib['default']
        else:
            self.raise_error("State variable '{0}' must specify a dimension", name)

        if 'boundaries' in node.lattrib:
            boundaries = node.lattrib['boundaries']
        else:
            boundaries = None

        self.current_regime.add_state_variable(StateVariable(name, default, boundaries))

    def parse_structure(self, node):
        """
        Parses <Structure>

        @param node: Node containing the <Structure> element
        @type node: xml.etree.Element
        """

        self.current_structure = self.current_component_type.structure
        self.process_nested_tags(node)
        self.current_structure = None

    def parse_target(self, node):
        """
        Parses <Target>

        @param node: Node containing the <Target> element
        @type node: xml.etree.Element
        """

        self.model.add_target(node.lattrib['component'])

    def parse_text(self, node):
        """
        Parses <Text>

        @param node: Node containing the <Text> element
        @type node: xml.etree.Element
        """

        if 'name' in node.lattrib:
            name = node.lattrib['name']
        else:
            self.raise_error('<Text> must specify a name.')

        description = node.lattrib.get('description', '')

        self.current_component_type.add_text(Text(name, description))

    def parse_time_derivative(self, node):
        """
        Parses <TimeDerivative>

        @param node: Node containing the <TimeDerivative> element
        @type node: xml.etree.Element

        @raise ParseError: Raised when the time derivative does not hava a variable
        name of a value.
        """

        if 'name' in node.lattrib:
            name = node.lattrib['name']
        else:
            self.raise_error('<TimeDerivative> must specify a name.')

        if 'expression' in node.lattrib:
            expression = node.lattrib['expression']
        else:
            self.raise_error("Time derivative for '{0}' must specify an expression.",
                             variable)

        self.current_regime.add_time_derivative(TimeDerivative(name, expression))

    def parse_transition(self, node):
        """
        Parses <Transition>

        @param node: Node containing the <Transition> element
        @type node: xml.etree.Element
        """

        if 'regime' in node.lattrib:
            regime = node.lattrib['regime']
        else:
            self.raise_error('<Transition> mut specify a regime.')

        action = Transition(regime)

        self.current_event_handler.add_action(action)

    def parse_unit(self, node):
        """
        Parses <Unit>

        @param node: Node containing the <Unit> element
        @type node: xml.etree.Element

        @raise ParseError: When the name is not a string or the unit
        specfications are incorrect.

        @raise ModelError: When the unit refers to an undefined dimension.
        """

        try:
            symbol = node.lattrib['symbol']
            dimension = node.lattrib['dimension']
        except:
            self.raise_error('Unit must have a symbol and dimension.')

        if 'power' in node.lattrib:
            power = int(node.lattrib['power'])
        else:
            power = 0

        if 'name' in node.lattrib:
            name = node.lattrib['name']
        else:
            name = ''
            
        if 'scale' in node.lattrib:
            scale = float(node.lattrib['scale'])
        else:
            scale = 1.0
            
        if 'offset' in node.lattrib:
            offset = float(node.lattrib['offset'])
        else:
            offset = 0.0

        self.model.add_unit(Unit(name, symbol, dimension, power, scale, offset))

    def parse_with(self, node):
        """
        Parses <With>

        @param node: Node containing the <With> element
        @type node: xml.etree.Element
        """

        if 'instance' in node.lattrib:
            instance = node.lattrib['instance']
            list = None
            index = None
        elif 'list' in node.lattrib and 'index' in node.lattrib:
            instance = None
            list = node.lattrib['list']
            index = node.lattrib['index']
        else:
            self.raise_error('<With> must specify EITHER instance OR list & index')

        if 'as' in node.lattrib:
            as_ = node.lattrib['as']
        else:
            self.raise_error('<With> must specify a name for the '
                             'target instance')

        self.current_structure.add_with(With(instance, as_, list, index))

