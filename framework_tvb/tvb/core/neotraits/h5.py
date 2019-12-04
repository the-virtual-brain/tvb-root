from ._h5accessors import DataSet, DataSetMetaData
from ._h5accessors import Scalar, Reference, Accessor
from ._h5accessors import SparseMatrix, SparseMatrixMetaData
from ._h5accessors import Json, JsonFinal, EquationScalar
from ._h5core import H5File

import h5py
STORE_STRING = h5py.string_dtype(encoding='utf-8')
MEMORY_STRING = "U128"

