# coding=utf-8

from tvb.contrib.scripts.datatypes.base import BaseModel
from tvb.datatypes.local_connectivity import LocalConnectivity as TVBLocalConnectivity


class LocalConnectivity(TVBLocalConnectivity, BaseModel):

    def to_tvb_instance(self, **kwargs):
        return super(LocalConnectivity, self).to_tvb_instance(TVBLocalConnectivity, **kwargs)
