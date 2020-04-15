# coding=utf-8

from tvb.contrib.scripts.datatypes.base import BaseModel
from tvb.datatypes.projections import ProjectionMatrix as TVBProjectionMatrix
from tvb.datatypes.projections import ProjectionSurfaceEEG as TVBProjectionSurfaceEEG
from tvb.datatypes.projections import ProjectionSurfaceSEEG as TVBProjectionSurfaceSEEG
from tvb.datatypes.projections import ProjectionSurfaceMEG as TVBProjectionSurfaceMEG


def get_TVB_proj_type(s_type):
    if s_type.lower() == "eeg":
        return TVBProjectionSurfaceEEG
    elif s_type.lower() == "seeg":
        return TVBProjectionSurfaceSEEG
    elif s_type.lower() == "meg":
        return TVBProjectionSurfaceMEG
    else:
        return TVBProjectionMatrix


class ProjectionMatrix(TVBProjectionMatrix, BaseModel):

    def to_tvb_instance(self, datatype=TVBProjectionMatrix, **kwargs):
        return super(ProjectionMatrix, self).to_tvb_instance(datatype, **kwargs)

    @classmethod
    def _from_tvb_projection_file(cls, s_type, source_file, matlab_data_name=None, is_brainstorm=False,
                      return_tvb_instance=False, **kwargs):
        if source_file.endswith("mat") and matlab_data_name is None:
            if len(s_type) > 0:
                matlab_data_name = kwargs.pop("%s_matlab_data_name" % s_type.lower(),
                                              kwargs.get("matlab_data_name", "ProjectionMatrix"))
            else:
                matlab_data_name = kwargs.get("matlab_data_name", "ProjectionMatrix")
        tvb_instance = get_TVB_proj_type(s_type).from_file(source_file, matlab_data_name, is_brainstorm)
        result = cls.from_tvb_instance(tvb_instance, **kwargs)
        if return_tvb_instance:
            return result, tvb_instance
        else:
            return result

    @classmethod
    def from_tvb_file(cls, source_file, matlab_data_name=None, is_brainstorm=False,
                      return_tvb_instance=False, **kwargs):
        return cls._from_tvb_projection_file("", source_file, matlab_data_name, is_brainstorm,
                                             return_tvb_instance, **kwargs)


class ProjectionSurfaceEEG(ProjectionMatrix, TVBProjectionSurfaceEEG):

    def to_tvb_instance(self, **kwargs):
        return super(ProjectionSurfaceEEG, self).to_tvb_instance(TVBProjectionSurfaceEEG, **kwargs)

    @classmethod
    def from_tvb_file(cls, source_file, matlab_data_name=None, is_brainstorm=False,
                      return_tvb_instance=False, **kwargs):
        return cls._from_tvb_projection_file("eeg", source_file, matlab_data_name, is_brainstorm,
                                             return_tvb_instance, **kwargs)


class ProjectionSurfaceSEEG(ProjectionMatrix, TVBProjectionSurfaceSEEG):

    def to_tvb_instance(self, **kwargs):
        return super(ProjectionSurfaceSEEG, self).to_tvb_instance(TVBProjectionSurfaceSEEG, **kwargs)

    @classmethod
    def from_tvb_file(cls, source_file, matlab_data_name=None, is_brainstorm=False,
                      return_tvb_instance=False, **kwargs):
        return cls._from_tvb_projection_file("seeg", source_file, matlab_data_name, is_brainstorm,
                                             return_tvb_instance, **kwargs)


class ProjectionSurfaceMEG(ProjectionMatrix, TVBProjectionSurfaceMEG):

    def to_tvb_instance(self, **kwargs):
        return super(ProjectionSurfaceMEG, self).to_tvb_instance(TVBProjectionSurfaceMEG, **kwargs)

    @classmethod
    def from_tvb_file(cls, source_file, matlab_data_name=None, is_brainstorm=False,
                      return_tvb_instance=False, **kwargs):
        return cls._from_tvb_projection_file("meg", source_file, matlab_data_name, is_brainstorm,
                                             return_tvb_instance, **kwargs)
