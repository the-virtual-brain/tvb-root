# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Contributors Package. This package holds simulator extensions.
#  See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

"""
.. moduleauthor:: Dionysios Perdikis <Denis@tvb.invalid>
"""

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
