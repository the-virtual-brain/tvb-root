from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.file.exceptions import FileStructureException, UnsupportedFileStorageException
from tvb.file.hdf5_storage_manager import HDF5StorageManager


class FileStorageFactory(object):

    @staticmethod
    def get_file_storage(storage_folder, file_name, buffer_size=600000):
        if storage_folder is None:
            raise FileStructureException("Please provide the folder where to store data")
        if file_name is None:
            raise FileStructureException("Please provide the file name where to store data")

        if TvbProfile.current.file_storage == 'h5':
            return HDF5StorageManager(storage_folder, file_name, buffer_size)

        raise UnsupportedFileStorageException("Unsupported file storage was chosen! Currently only H5 storage is "
                                              "supported!")

