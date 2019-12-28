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
import importlib
import os
import uuid
from tvb.core.entities.file.simulator.h5_factory import config_h5_factory
from tvb.core.neotraits.h5 import H5File
from tvb.core.neocom import h5


class SimulatorConfigurationH5(H5File):
    @staticmethod
    def get_full_class_name(class_entity):
        return class_entity.__module__ + '.' + class_entity.__name__

    def store_config_as_reference(self, config):
        gid = uuid.uuid4()

        config_h5_class = config_h5_factory(type(config))
        config_path = h5.path_for(os.path.dirname(self.path), config_h5_class, gid)

        with config_h5_class(config_path) as config_h5:
            config_h5.store(config)
            config_h5.gid.store(gid)
            config_h5.type.store(self.get_full_class_name(type(config)))

        return gid

    def get_reference_path(self, gid):
        dir_loader = h5.DirLoader(os.path.dirname(self.path), h5.REGISTRY)
        config_filename = dir_loader.find_file_name(gid)
        config_path = os.path.join(dir_loader.base_dir, config_filename)
        return config_path

    def load_from_reference(self, gid):
        config_path = self.get_reference_path(gid)

        config_h5 = H5File.from_file(config_path)

        config_type = config_h5.type.load()
        package, cls_name = config_type.rsplit('.', 1)
        module = importlib.import_module(package)
        config_class = getattr(module, cls_name)

        config_instance = config_class()
        config_h5.load_into(config_instance)
        config_h5.close()

        return config_instance
