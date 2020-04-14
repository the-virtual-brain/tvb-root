# coding=utf-8

from tvb.contrib.scripts.datatypes.base import BaseModel
from tvb.datatypes.connectivity import Connectivity as TVBConnectivity


class ConnectivityH5Field(object):
    WEIGHTS = "weights"
    TRACTS = "tract_lengths"
    CENTERS = "centres"
    CENTRES = "centres"
    REGION_LABELS = "region_labels"
    ORIENTATIONS = "orientations"
    HEMISPHERES = "hemispheres"
    AREAS = "areas"


class Connectivity(TVBConnectivity, BaseModel):

    # The following two methods help avoid problems with centers vs centres writing
    def __setattr__(self, key, value):
        if key == "centers":
            super(Connectivity, self).__setattr__("centres", value)
        else:
            super(Connectivity, self).__setattr__(key, value)

    @property
    def centers(self):
        return self.centres

    # A usefull method for addressing subsets of the connectome by label:
    def get_regions_inds_by_labels(self, labels):
        return self.labels2inds(self.region_labels, labels)

    def to_tvb_instance(self, **kwargs):
        return super(Connectivity, self).to_tvb_instance(TVBConnectivity, **kwargs)
