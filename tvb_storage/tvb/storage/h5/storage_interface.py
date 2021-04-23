from tvb.storage.h5.encryption.data_encryption_handler import DataEncryptionHandler, FoldersQueueConsumer
from tvb.storage.h5.encryption.encryption_handler import EncryptionHandler
from tvb.storage.h5.file.files_helper import FilesHelper
from tvb.storage.h5.file.hdf5_storage_manager import HDF5StorageManager
from tvb.storage.h5.file.xml_metadata_handlers import XMLReader, XMLWriter
from tvb.storage.h5.exceptions import *
from tvb.storage.h5.decorators import synchronized
from tvb.storage.h5.utils import string2date, date2string, string2bool


class StorageInterface:

    def __init__(self):
        self.files_helper = FilesHelper()
        self.storage_manager = HDF5StorageManager()
        self.xml_reader = XMLReader()
        self.xml_writer = XMLWriter()
        self.encryption_handler = EncryptionHandler()
        self.data_encryption_handler = DataEncryptionHandler()
        self.folders_queue_consumer = FoldersQueueConsumer()
