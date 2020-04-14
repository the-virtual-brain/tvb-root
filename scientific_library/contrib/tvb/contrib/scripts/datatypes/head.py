# coding=utf-8

import os

from tvb.contrib.scripts.datatypes.base import BaseModel
from tvb.contrib.scripts.datatypes.connectivity import Connectivity
from tvb.contrib.scripts.datatypes.local_connectivity import LocalConnectivity
from tvb.contrib.scripts.datatypes.projections import ProjectionSurfaceEEG, ProjectionSurfaceSEEG, \
    ProjectionSurfaceMEG
from tvb.contrib.scripts.datatypes.region_mapping import CorticalRegionMapping, SubcorticalRegionMapping, \
    RegionVolumeMapping
from tvb.contrib.scripts.datatypes.sensors import SensorsEEG, SensorsSEEG, SensorsMEG
from tvb.contrib.scripts.datatypes.structural import T1, T2, Flair, B0
from tvb.contrib.scripts.datatypes.surface import Surface, CorticalSurface, SubcorticalSurface
from tvb.contrib.scripts.utils.file_utils import insensitive_glob
from tvb.basic.logger.builder import get_logger
from tvb.basic.neotraits.api import HasTraits, Attr
from tvb.datatypes.connectivity import Connectivity as TVBConnectivity
from tvb.datatypes.cortex import Cortex
from tvb.datatypes.local_connectivity import LocalConnectivity as TVBLocalConnectivity
from tvb.datatypes.projections import EEG_POLYMORPHIC_IDENTITY, SEEG_POLYMORPHIC_IDENTITY, MEG_POLYMORPHIC_IDENTITY
from tvb.datatypes.projections import ProjectionMatrix as TVBProjectionMatrix
from tvb.datatypes.region_mapping import RegionMapping as TVBRegionMapping
from tvb.datatypes.region_mapping import RegionVolumeMapping as TVBRegionVolumeMapping
from tvb.datatypes.sensors import Sensors as TVBSensors
from tvb.datatypes.structural import StructuralMRI as TVBStructuralMRI
from tvb.datatypes.surfaces import Surface as TVBSurface
from tvb.simulator.plot.utils import raise_value_error


class Head(HasTraits):
    logger = get_logger(__name__)

    """
    One patient virtualization. Fully configured for defining hypothesis on it.
    """
    # TODO: find a solution with cross-references between tvb-scripts and TVB datatypes
    title = Attr(str, default="Head", required=False)
    path = Attr(str, default="path", required=False)
    connectivity = Attr(field_type=Connectivity)
    cortical_surface = Attr(field_type=CorticalSurface, required=False)
    subcortical_surface = Attr(field_type=SubcorticalSurface, required=False)
    cortical_region_mapping = Attr(field_type=CorticalRegionMapping, required=False)
    subcortical_region_mapping = Attr(field_type=SubcorticalRegionMapping, required=False)
    region_volume_mapping = Attr(field_type=RegionVolumeMapping, required=False)
    local_connectivity = Attr(field_type=LocalConnectivity, required=False)
    t1 = Attr(field_type=T1, required=False)
    t2 = Attr(field_type=T2, required=False)
    flair = Attr(field_type=Flair, required=False)
    b0 = Attr(field_type=B0, required=False)
    eeg_sensors = Attr(field_type=SensorsEEG, required=False)
    seeg_sensors = Attr(field_type=SensorsSEEG, required=False)
    meg_sensors = Attr(field_type=SensorsMEG, required=False)
    eeg_projection = Attr(field_type=ProjectionSurfaceEEG, required=False)
    seeg_projection = Attr(field_type=ProjectionSurfaceSEEG, required=False)
    meg_projection = Attr(field_type=ProjectionSurfaceMEG, required=False)
    _cortex = None

    def __init__(self, **kwargs):
        super(Head, self).__init__(**kwargs)

    def configure(self):
        if isinstance(self.connectivity, TVBConnectivity):
            self.connectivity.configure()
        if isinstance(self.connectivity, TVBLocalConnectivity):
            self.local_connectivity.configure()
        if isinstance(self.cortical_surface, TVBSurface):
            self.cortical_surface.configure()
            if isinstance(self.cortical_region_mapping, TVBRegionMapping):
                self.cortical_region_mapping.connectivity = self.connectivity.to_tvb_instance()
                self.cortical_region_mapping.surface = self.cortical_surface.to_tvb_instance()
                self.cortical_region_mapping.configure()
        if isinstance(self.subcortical_surface, TVBSurface):
            self.subcortical_surface.configure()
            if isinstance(self.subcortical_region_mapping, TVBRegionMapping):
                self.subcortical_region_mapping.connectivity = self.connectivity.to_tvb_instance()
                self.subcortical_region_mapping.surface = self.subcortical_surface.to_tvb_instance()
                self.subcortical_region_mapping.configure()
        structural = None
        for s_type in ["b0", "flair", "t2", "t1"]:
            instance = getattr(self, s_type)
            if isinstance(instance, TVBStructuralMRI):
                instance.configure()
                structural = instance
        if structural is not None:
            if isinstance(self.region_volume_mapping, TVBRegionVolumeMapping):
                self.region_volume_mapping.connectivity = self.connectivity
                self.region_volume_mapping.volume = structural.volume
                self.region_volume_mapping.configure()
        for s_type, p_type in zip(["eeg", "seeg", "meg"],
                                  [EEG_POLYMORPHIC_IDENTITY, SEEG_POLYMORPHIC_IDENTITY, MEG_POLYMORPHIC_IDENTITY]):
            sensor = "%s_sensors" % s_type
            sensors = getattr(self, sensor)
            if isinstance(sensors, TVBSensors):
                sensors.configure()
                projection = "%s_projection" % s_type
                projection = getattr(self, projection)
                if isinstance(projection, TVBProjectionMatrix):
                    projection.sensors = sensors.to_tvb_instance()
                    if isinstance(self.surface, Surface):
                        projection.sources = self.surface.to_tvb_instance()
                    projection.projection_type = p_type
                    projection.configure()

    def filter_regions(self, filter_arr):
        return self.connectivity.region_labels[filter_arr]

    def _get_filepath(self, filename, patterns, used_filepaths):
        # Search for default names if there is no filename provided
        if filename is None:
            for pattern in patterns:
                filepaths = insensitive_glob(os.path.join(self.path, "*%s*" % pattern))
                if len(filepaths) > 0:
                    for filepath in filepaths:
                        if filepath not in used_filepaths and os.path.isfile(filepath):
                            return filepath
            return None
        else:
            try:
                return insensitive_glob(os.path.join(self.path, "*%s*" % filename))[0]
            except:
                self.lowarning("No *%s* file found in %s path!" % (filename, self.path))

    def _load_reference(self, datatype, arg_name, patterns, used_filepaths, **kwargs):
        # Load from file
        filepath = self._get_filepath(kwargs.pop(arg_name, None), patterns, used_filepaths)
        if filepath is not None:
            used_filepaths.append(filepath)
            if issubclass(datatype, BaseModel):
                if filepath.endswith("h5"):
                    return datatype.from_h5_file(filepath), kwargs
                else:
                    return datatype.from_tvb_file(filepath), kwargs
            else:
                return datatype.from_file(filepath), kwargs
        else:
            return None, kwargs

    @classmethod
    def from_folder(cls, path=None, head=None, **kwargs):
        # TODO confirm the filetypes and add (h5 and other) readers to all TVB classes .from_file methods
        # Default patterns:
        # *conn* for zip/h5 files
        # (*cort/subcort*)surf*(*cort/subcort*) / (*cort/subcort*)srf*(*cort/subcort*) for zip/h5 files
        # (*cort/subcort*)reg*map(*cort/subcort*) for txt files
        # *map*vol* / *vol*map* for txt files
        # *t1/t2/flair/b0 for ??? files
        # *eeg/seeg/meg*sensors/locations* / *sensors/locations*eeg/seeg/meg for txt files
        # # *eeg/seeg/meg*proj/gain* / *proj/gain*eeg/seeg/meg for npy/mat

        used_filepaths = []

        if head is None:
            head = Head()
            head.path = path
            title = os.path.basename(path)
            if len(title) > 0:
                head.title = title

        # We need to read local_connectivity first to avoid confusing it with connectivity:
        head.local_connectivity, kwargs = \
            head._load_reference(LocalConnectivity, 'local_connectivity', ["loc*conn", "conn*loc"],
                                 used_filepaths, **kwargs)

        # Connectivity is required
        # conn_instances
        connectivity, kwargs = \
            head._load_reference(Connectivity, "connectivity", ["conn"], used_filepaths, **kwargs)
        if connectivity is None:
            raise_value_error("A Connectivity instance is minimally required for a Head instance!")
        head.connectivity = connectivity

        # TVB only volume datatypes: do before region_mappings to avoid confusing them with volume_mapping
        structural = None
        for datatype, arg_name, patterns in zip([B0, Flair, T2, T1],
                                                ["b0", "flair", "t2", "t1", ],
                                                [["b0"], ["flair"], ["t2"], ["t1"]]):
            try:
                datatype.from_file
                instance, kwargs = head._load_reference(datatype, arg_name, patterns, used_filepaths, **kwargs)
            except:
                cls.logger.warning("No 'from_file' method yet for %s!" % datatype.__class__.__name__)
                instance = None
            if instance is not None:
                setattr(head, arg_name, instance)
                volume_instance = instance
        if structural is not None:
            head.region_volume_mapping, kwargs = \
                head._load_reference(RegionVolumeMapping, "region_volume_mapping", ["vol*map", "map*vol"],
                                     used_filepaths, **kwargs)

        # Surfaces and mappings
        # (read subcortical ones first to avoid confusion):
        head.subcortical_surface, kwargs = \
            head._load_reference(SubcorticalSurface, "subcortical_surface",
                                 ["subcort*surf", "surf*subcort", "subcort*srf", "srf*subcort"],
                                 used_filepaths, **kwargs)
        if head.subcortical_surface is not None:
            # Region Mapping requires Connectivity and Surface
            head.subcortical_region_mapping, kwargs = \
                head._load_reference(SubcorticalRegionMapping, "subcortical_region_mapping",
                                     ["subcort*reg*map", "reg*map*subcort"],
                                     used_filepaths, **kwargs)

        head.cortical_surface, kwargs = \
            head._load_reference(CorticalSurface, "cortical_surface",
                                 ["cort*surf", "surf*cort", "cort*srf", "srf*cort", "surf", "srf"],
                                 used_filepaths, **kwargs)
        if head.cortical_surface is not None:
            # Region Mapping requires Connectivity and Surface
            head.cortical_region_mapping, kwargs = \
                head._load_reference(CorticalRegionMapping, "cortical_region_mapping",
                                     ["cort*reg*map", "reg*map*cort", "reg*map"], used_filepaths, **kwargs)

        # Sensors and projections
        # (read seeg before eeg to avoid confusion!)
        for s_datatype, p_datatype, s_type in zip([SensorsSEEG, SensorsEEG, SensorsMEG],
                                                  [ProjectionSurfaceSEEG, ProjectionSurfaceEEG, ProjectionSurfaceMEG],
                                                  ["seeg", "eeg", "meg"]):
            arg_name = "%s_sensors" % s_type
            patterns = ["%s*sensors" % s_type, "sensors*%s" % s_type,
                        "%s*locations" % s_type, "locations*%s" % s_type]
            sensors, kwargs = head._load_reference(s_datatype, arg_name, patterns, used_filepaths, **kwargs)
            if sensors is not None:
                setattr(head, arg_name, sensors)
                arg_name = "%s_projection" % s_type
                patterns = ["%s*proj" % s_type, "proj*%s" % s_type, "%s*gain" % s_type, "gain*%s" % s_type]
                projection, kwargs = head._load_reference(p_datatype, arg_name, patterns, used_filepaths, **kwargs)
                setattr(head, arg_name, projection)

        return head

    @classmethod
    def from_file(cls, path=None, **kwargs):
        filename = os.path.basename(path)
        dirname = os.path.dirname(path)
        if "head" in filename.lower():
            import h5py
            head = Head()
            head.path = path
            h5file = h5py.File(path, 'r', libver='latest')
            for field in []:
                try:
                    setattr(head, field, h5file['/' + field][()])
                except:
                    cls.logger.warning("Failed to read Head field %s from file %s!" % (field, path))
            for attr in ["title"]:
                try:
                    setattr(head, attr, h5file.attrs.get(attr, h5file.attrs.get("TVB_%s" % attr)))
                except:
                    cls.logger.warning("Failed to read Head attribute %s from file %s!" % (attr, path))
            head.path = dirname
        else:
            kwargs["connectivity"] = filename
            head = None
        return cls.from_folder(dirname, head, **kwargs)

    @classmethod
    def from_tvb_file(cls, path=None, **kwargs):
        return cls.from_file(path, **kwargs)

    def make_cortex(self, local_connectivity=None, coupling_strength=None):
        self._cortex = Cortex()
        self._cortex.region_mapping_data = self.cortical_region_mapping
        if isinstance(local_connectivity, LocalConnectivity):
            self._cortex.local_connectivity = local_connectivity
        if coupling_strength is not None:
            self._cortex.coupling_strength = coupling_strength
        self._cortex.configure()
        return self._cortex

    def cortex(self, local_connectivity=None, coupling_strength=None):
        if not isinstance(self._cortex, Cortex):
            self.make_cortex(local_connectivity, coupling_strength)
        return self._cortex

    @property
    def surface(self):
        return self.cortical_surface

    @property
    def number_of_regions(self):
        return self.connectivity.number_of_regions
