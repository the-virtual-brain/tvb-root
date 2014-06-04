# Gifti IO
# Stephan Gerhard, May 2010

import os
from tvb.core.utils import store_list_data
from util import *
from xml.dom.minidom import Document



class GiftiMetaData(object):
    
    def __init__(self):
        # create a list of GiftiNVPairs objects
        # containing the Name and the Value
        self.data = []
        
    def get_data_as_dict(self):
        
        self.data_as_dict = {}
        for ele in self.data:
            self.data_as_dict[ele.name] = ele.value
        return self.data_as_dict
    
    def print_summary(self):
        print self.get_data_as_dict()

   
class GiftiNVPairs(object):
    
    name = str
    value = str

class GiftiLabelTable(object):
    
    def __init__(self):
        
        self.labels = []
        
    def get_labels_as_dict(self):
        
        self.labels_as_dict = {}
        
        for ele in self.labels:
            self.labels_as_dict[ele.index] = ele.label
            
        return self.labels_as_dict

    def print_summary(self):
        print self.get_labels_as_dict()


class GiftiLabel(object):
    
    index = int
    label = str

class GiftiCoordSystem(object):
    
    dataspace = str
    xformspace = str
    xform = None # will be numpy array

    def print_summary(self):
        
        print 'Dataspace: ', self.dataspace
        print 'XFormSpace: ', self.xformspace
        print 'Affine Transformation Matrix: \n', self.xform

class GiftiDataArray(object):
    
    intent = int
    datatype = int
    ind_ord = int
    num_dim = int
    dims = []
    encoding = int
    endian = int
    ext_fname = str
    ext_offset = None
    
    data = None
    coordsys = None # GiftiCoordSystem()
        
    def __init__(self):
        
        self.dims = []
        self.meta = GiftiMetaData()
        
    def print_summary(self):
        
        print 'Intent: ', GiftiIntentCode.intents_inv[self.intent]
        print 'DataType: ', GiftiDataType.datatypes_inv[self.datatype]
        print 'ArrayIndexingOrder: ', GiftiArrayIndexOrder.ordering_inv[self.ind_ord]
        print 'Dimensionality: ', self.num_dim
        print 'Dimensions: ', self.dims
        print 'Encoding: ', GiftiEncoding.encodings_inv[self.encoding]
        print 'Endian: ', GiftiEndian.endian_inv[self.endian]
        print 'ExternalFileName: ', self.ext_fname
        print 'ExternalFileOffset: ', self.ext_offset
        if not self.coordsys == None:
            print '----'
            print 'Coordinate System:'
            print self.coordsys.print_summary()

    def get_meta_as_dict(self):
        return self.meta.get_data_as_dict()

class GiftiImage(object):
    
    numDA = int
    version = str
    filename = str

    def __init__(self):
        
        # list of GiftiDataArray
        self.darrays = []
        self.meta = GiftiMetaData()
        self.labeltable = GiftiLabelTable()
        
    # add getter and setter methods?
    def get_metadata(self):
        
        return self.meta
    
    def set_metadata(self, meta):
        
        # XXX: if it exists, make it readonly?
        # similar issues with nifti,
        # especially for GiftiDataArray metadata!
        # e.g. changing transformation matrix
        
        self.meta = meta
    
    def add_gifti_data_array(self, dataarr):
        self.darrays.append(dataarr)
        
        self.numDA += 1
        
        # XXX sanity checks
        
    def remove_gifti_data_array(self, dataarr):
        self.darrays.remove(dataarr)
        # XXX update
    
    def getArraysFromIntent(self, intent):
        """ Returns a a list of GiftiDataArray elements matching
        the given intent """
        
        # if it is integer do not convert
        if type(intent)=='int':
            it = GiftiIntentCode.intents[intent]
        else:
            it = intent
            
        return [x for x in self.darrays if x.intent == it]
        
    
    def print_summary(self):
        
        print '----start----'
        print 'Source filename: ', self.filename
        print 'Number of data arrays: ', self.numDA
        print 'Version: ', self.version
        if not self.meta == None:
            print '----'
            print 'Metadata:'
            print self.meta.print_summary()
        if not self.labeltable == None:
            print '----'
            print 'Labeltable:'
            print self.labeltable.print_summary()
        for i, da in enumerate(self.darrays):
            print '----'
            print 'DataArray %s:' % i
            print da.print_summary()
        print '----end----'
    

##############
# General Gifti Input - Output to the filesystem
##############

def loadImage(filename):
    """ Load a Gifti image from a file """
    import os.path
    if not os.path.exists(filename):
        raise IOError("No such file or directory: '%s'" % filename)
    else:
        import parse_gifti_fast as pg
        giifile = pg.parse_gifti_file(filename)
        return giifile
     
            
def saveImage(image, filename):
    """ 
    Save the current image to a new file
        
	If the image was created using array data (not loaded from a file) one
	has to specify a filename
	
	Note that the Gifti spec suggests using the following suffixes to your
	filename when saving each specific type of data:
    
    - Generic GIFTI File    .gii
	- Coordinates           .coord.gii
	- Functional            .func.gii
	- Labels                .label.gii
	- RGB or RGBA           .rgba.gii
	- Shape                 .shape.gii
	- Surface               .surf.gii
	- Tensors               .tensor.gii
	- Time Series           .time.gii
	- Topology              .topo.gii	
	"""
        
    doc = Document()
    root_node = doc.createElement('GIFTI')
    root_node.setAttribute('Version', '1.03')
    root_node.setAttribute('NumberOfDataArrays', str(image.numDA))
    
    #TODO Add metadata writing
    metadata_root = doc.createElement('MetaData')
    metadata = image.meta.get_data_as_dict()
    for key in metadata:
        MD_node = doc.createElement('MD')
        name_node = doc.createElement('Name')
        name_value = doc.createCDATASection(str(key))
        name_node.appendChild(name_value)
        MD_node.appendChild(name_node)
        value_node = doc.createElement('Value')
        value_value = doc.createCDATASection(str(metadata[key]))
        value_node.appendChild(value_value)
        MD_node.appendChild(value_node)
        metadata_root.appendChild(MD_node)
    root_node.appendChild(metadata_root)
    
    #TODO Add label writing
    label_root = doc.createElement('LabelTable')
    #Append label data to this
    root_node.appendChild(label_root)
    
    for i in xrange(int(image.numDA)):
        # Intent code is stored in the DataArray struct
        darray = image.darrays[i]
        array_node = doc.createElement('DataArray')
        array_node.setAttribute('Intent', str(GiftiIntentCode.intents_inv[darray.intent]))
        array_node.setAttribute('DataType', str(GiftiDataType.datatypes_inv[darray.datatype]))
        array_node.setAttribute('ArrayIndexingOrder', "RowMajorOrder")
        array_node.setAttribute('Dimensionality', str(len(darray.dims)))
        for dim_idx, dimension in enumerate(darray.dims):
            dim_name = 'Dim' + str(dim_idx)
            array_node.setAttribute(dim_name, str(dimension))
        array_node.setAttribute("Encoding", "ASCII")
        array_node.setAttribute('endian', 'LittleEndian')
        array_node.setAttribute('ExternalFileName', '')
        array_node.setAttribute('ExternalFileOffset', '')
        
        array_data_node = doc.createElement('Data')
        
        folder = os.path.split(filename)[0]
        store_list_data(darray.data, "tmp", folder)
        tmp_file = open(os.path.join(folder, "tmp"), "r")
        data_string = tmp_file.read()
        array_data = doc.createTextNode(data_string)
        tmp_file.close()
        os.remove(os.path.join(folder, "tmp"))
        print "Created node"
        array_data_node.appendChild(array_data)
        array_node.appendChild(array_data_node)
        root_node.appendChild(array_node)
    doc.appendChild(root_node)
    file_obj = open(filename, 'wb')
    doc.writexml(file_obj, addindent="    ", newl="\n")
    file_obj.close()



##############
# special purpose GiftiImage / GiftiDataArray creation methods
##############

#def GiftiImage_fromarray(data, intent = GiftiIntentCode.NIFTI_INTENT_NONE, encoding=GiftiEncoding.GIFTI_ENCODING_B64GZ, endian = GiftiEndian.GIFTI_ENDIAN_LITTLE):
#    """ Returns a GiftiImage from a Numpy array with a given intent code and
#    encoding """
#    pass
#    
#def GiftiImage_fromTriangles(vertices, triangles, cs = None, encoding=GiftiEncoding.GIFTI_ENCODING_B64GZ, endian = GiftiEndian.GIFTI_ENDIAN_LITTLE):
#    """ Returns a GiftiImage from two numpy arrays representing the vertices
#    and the triangles. Additionally defining the coordinate system and encoding """
#    pass
#    
#def GiftiDataArray_fromarray(self, data, intent = GiftiIntentCode.NIFTI_INTENT_NONE, encoding=GiftiEncoding.GIFTI_ENCODING_B64GZ, endian = GiftiEndian.GIFTI_ENDIAN_LITTLE):
#    """ Returns as GiftiDataArray from a Numpy array with a given intent code and
#    encoding """
#    pass
