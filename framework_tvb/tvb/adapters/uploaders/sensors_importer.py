# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

"""
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

from tvb.adapters.datatypes.db.sensors import SensorsIndex
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.adapters.abcuploader import ABCUploader, ABCUploaderForm
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import TraitUploadField, SelectField
from tvb.core.neotraits.h5 import MEMORY_STRING
from tvb.core.neotraits.uploader_view_model import UploaderViewModel
from tvb.core.neotraits.view_model import Str
from tvb.datatypes.sensors import SensorsEEG, SensorsMEG, SensorsInternal


class SensorsImporterModel(UploaderViewModel):
    OPTIONS = {'EEG Sensors': SensorsEEG.sensors_type.default,
               'MEG Sensors': SensorsMEG.sensors_type.default,
               'Internal Sensors': SensorsInternal.sensors_type.default}

    sensors_file = Str(
        label='Please upload sensors file (txt or bz2 format)',
        doc='Expected a text/bz2 file containing sensor measurements.'
    )

    sensors_type = Str(
        label='Sensors type: ',
        choices=tuple(OPTIONS.values()),
        default=tuple(OPTIONS.values())[0]
    )


class SensorsImporterForm(ABCUploaderForm):

    def __init__(self, project_id=None):
        super(SensorsImporterForm, self).__init__(project_id)

        self.sensors_file = TraitUploadField(SensorsImporterModel.sensors_file, ('.txt', '.bz2'), self.project_id,
                                             'sensors_file', self.temporary_files)
        self.sensors_type = SelectField(SensorsImporterModel.sensors_type, self.project_id, name='sensors_type',
                                        choices=SensorsImporterModel.OPTIONS)

    @staticmethod
    def get_view_model():
        return SensorsImporterModel

    @staticmethod
    def get_upload_information():
        return {
            'sensors_file': ('.txt', '.bz2')
        }


class SensorsImporter(ABCUploader):
    """
    Upload Sensors from a TXT file.
    """
    _ui_name = "Sensors"
    _ui_subsection = "sensors_importer"
    _ui_description = "Import Sensor locations from TXT or BZ2"

    logger = get_logger(__name__)

    def get_form_class(self):
        return SensorsImporterForm

    def get_output(self):
        return [SensorsIndex]

    def launch(self, view_model):
        # type: (SensorsImporterModel) -> [SensorsIndex]
        """
        Creates required sensors from the uploaded file.
        :returns: a list of sensors instances of the specified type

        :raises LaunchException: when
                    * no sensors_file specified
                    * sensors_type is invalid (not one of the mentioned options)
                    * sensors_type is "MEG sensors" and no orientation is specified
        """
        if view_model.sensors_file is None:
            raise LaunchException("Please select sensors file which contains data to import")

        self.logger.debug("Create sensors instance")
        if view_model.sensors_type == SensorsEEG.sensors_type.default:
            sensors_inst = SensorsEEG()
        elif view_model.sensors_type == SensorsMEG.sensors_type.default:
            sensors_inst = SensorsMEG()
        elif view_model.sensors_type == SensorsInternal.sensors_type.default:
            sensors_inst = SensorsInternal()
        else:
            exception_str = "Could not determine sensors type (selected option %s)" % view_model.sensors_type
            raise LaunchException(exception_str)

        locations = self.read_list_data(view_model.sensors_file, usecols=[1, 2, 3])

        # NOTE: TVB has the nose pointing -y and left ear pointing +x
        # If the sensors are in CTF coordinates : nose pointing +x left ear +y
        # to rotate the sensors by -90 along z uncomment below
        # locations = numpy.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]).dot(locations.T).T
        sensors_inst.locations = locations
        sensors_inst.labels = self.read_list_data(view_model.sensors_file, dtype=MEMORY_STRING, usecols=[0])
        sensors_inst.number_of_sensors = sensors_inst.labels.size

        if isinstance(sensors_inst, SensorsMEG):
            try:
                sensors_inst.orientations = self.read_list_data(view_model.sensors_file, usecols=[4, 5, 6])
                sensors_inst.has_orientation = True
            except IndexError:
                raise LaunchException("Uploaded file does not contains sensors orientation.")

        sensors_inst.configure()
        self.logger.debug("Sensors instance ready to be stored")

        return h5.store_complete(sensors_inst, self.storage_path)
