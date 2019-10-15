"""
The public api of the neotraits package.
"""

from ._core import HasTraits, trait_property, cached_trait_property
from .info import narray_describe, narray_summary_info
from ._attr import Attr, Int, Float, NArray, Const, List, Range, LinspaceRange
