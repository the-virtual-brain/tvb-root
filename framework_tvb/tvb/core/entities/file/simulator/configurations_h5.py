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
