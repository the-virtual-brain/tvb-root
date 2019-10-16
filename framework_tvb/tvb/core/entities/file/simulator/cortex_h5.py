from tvb.core.neotraits.h5 import H5File, Reference, Scalar
from tvb.datatypes.cortex import Cortex


class CortexH5(H5File):

    def __init__(self, path):
        super(CortexH5, self).__init__(path)
        self.local_connectivity = Reference(Cortex.local_connectivity, self)
        self.region_mapping_data = Reference(Cortex.region_mapping_data, self)
        self.coupling_strength = Scalar(Cortex.coupling_strength, self)
