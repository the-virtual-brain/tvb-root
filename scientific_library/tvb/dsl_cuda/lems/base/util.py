"""
PyLEMS utility classes / functions

@author: Gautham Ganapathy
@organization: LEMS (http://neuroml.org/lems/, https://github.com/organizations/LEMS)
@contact: gautham@lisphacker.org
"""

id_counter = 0

def make_id():
    global id_counter
    id_counter = id_counter + 1
    return '__id_{0}__'.format(id_counter)

def merge_maps(m, base):
    """
    Merge in undefined map entries from given map.
    
    @param m: Map to be merged into.
    @type m: lems.util.Map
    
    @param base: Map to be merged into.
    @type base: lems.util.Map
    """

    for k in base.keys():
        if k not in m:
            m[k] = base[k]

def merge_lists(l, base):
    """
    Merge in undefined list entries from given list.
    
    @param l: List to be merged into.
    @type l: list
    
    @param base: List to be merged into.
    @type base: list
    """
    
    for i in base:
        if i not in l:
            l.append(i)
            
        
def validate_lems(file_name):

    from lxml import etree
    try:
        from urllib2 import urlopen  # Python 2
    except:
        from urllib.request import urlopen # Python 3
        
    schema_file = urlopen("https://raw.githubusercontent.com/LEMS/LEMS/development/Schemas/LEMS/LEMS_v0.7.3.xsd")
    xmlschema = etree.XMLSchema(etree.parse(schema_file))
    print("Validating {0} against {1}".format(file_name, schema_file.geturl()))
    xmlschema.assertValid(etree.parse(file_name))
    print("It's valid!")

