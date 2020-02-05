# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

from tvb.adapters.visualizers.fourier_spectrum import FourierSpectrumModel
from tvb.basic.neotraits.api import Attr
from tvb.core.neotraits.h5 import ViewModelH5
from tvb.core.neotraits.uploader_view_model import UploaderViewModel
from tvb.core.neotraits.view_model import Str
from tvb.datatypes.spectral import FourierSpectrum


class DummyImporterViewModel(UploaderViewModel):
    uploaded = Str(
        label='File to upload'
    )

    dummy_scalar = Attr(
        field_type=float,
        required=False,
        label='Dummy int scalar'
    )


def test_dummy_importer_mv_to_h5(tmph5factory):
    dummy_file_name = 'file_name.zip'
    dummy_scalar = 1.0

    divm = DummyImporterViewModel(uploaded=dummy_file_name, dummy_scalar=dummy_scalar)
    path = tmph5factory()
    h5_file = ViewModelH5(path, divm)
    h5_file.store(divm)
    h5_file.close()

    loaded_divm = DummyImporterViewModel()
    assert not hasattr(loaded_divm, 'uploaded')
    assert loaded_divm.dummy_scalar is None
    h5_file = ViewModelH5(path, loaded_divm)
    h5_file.load_into(loaded_divm)
    assert loaded_divm.uploaded == dummy_file_name
    assert loaded_divm.dummy_scalar == dummy_scalar


def test_fourier_spectrum_model_to_h5(tmph5factory, time_series_index_factory):
    fs = FourierSpectrum()

    fsm = FourierSpectrumModel(input_data=fs.gid)
    path = tmph5factory()
    h5_file = ViewModelH5(path, fsm)

    h5_file.store(fsm)
    h5_file.close()

    loaded_dt = FourierSpectrumModel()
    h5_file = ViewModelH5(path, loaded_dt)
    h5_file.load_into(loaded_dt)
    assert loaded_dt.input_data == fs.gid
    assert loaded_dt.gid == fsm.gid
