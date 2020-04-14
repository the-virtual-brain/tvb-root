# coding=utf-8

from tvb.contrib.scripts.datatypes.base import BaseModel
from tvb.datatypes.region_mapping import RegionMapping as TVBRegionMapping
from tvb.datatypes.region_mapping import RegionVolumeMapping as TVBRegionVolumeMapping


class RegionMapping(TVBRegionMapping, BaseModel):

    def to_tvb_instance(self, datatype=TVBRegionMapping, **kwargs):
        return super(RegionMapping, self).to_tvb_instance(datatype, **kwargs)


class CorticalRegionMapping(RegionMapping):
    pass


class SubcorticalRegionMapping(RegionMapping):
    pass


class RegionVolumeMapping(TVBRegionVolumeMapping, BaseModel):

    def to_tvb_instance(self, **kwargs):
        return super(RegionVolumeMapping, self).to_tvb_instance(TVBRegionVolumeMapping, **kwargs)