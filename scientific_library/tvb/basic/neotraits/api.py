"""
The public api of the neotraits package.
"""

from ._core import HasTraits, traitproperty
from .info import narray_describe, narray_summary_info
from ._attr import Attr, Int, Float, NArray, Const, List, Range, LinspaceRange
