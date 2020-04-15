# coding=utf-8

from tvb.contrib.scripts.datatypes.base import BaseModel
from tvb.basic.neotraits.api import Attr
from tvb.datatypes.structural import StructuralMRI as TVBStructuralMRI


class StructuralMRI(TVBStructuralMRI, BaseModel):

    def to_tvb_instance(self, datatype=TVBStructuralMRI, **kwargs):
        return super(StructuralMRI, self).to_tvb_instance(datatype, **kwargs)


class T1(StructuralMRI):
    weighting = Attr(str, label="MRI weighting", default="T1")  # eg, "T1", "T2", "T2*", "PD", ...


class T2(StructuralMRI):
    weighting = Attr(str, label="MRI weighting", default="T2")  # eg, "T1", "T2", "T2*", "PD", ...


class Flair(StructuralMRI):
    weighting = Attr(str, label="MRI weighting", default="Flair")  # eg, "T1", "T2", "T2*", "PD", ...


class B0(StructuralMRI):
    weighting = Attr(str, label="MRI weighting", default="B0")  # eg, "T1", "T2", "T2*", "PD", ...
