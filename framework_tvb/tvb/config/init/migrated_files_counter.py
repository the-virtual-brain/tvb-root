from tvb.config.init.initializer import initialize
from tvb.core.entities.file.files_update_manager import FilesUpdateManager


if __name__ == '__main__':
    # migrate the database
    initialize()

    # migrate H5 files
    FilesUpdateManager().run_all_updates()

    # test if there are any files which were not migrated
    number_of_unmigrated_files = len(FilesUpdateManager.get_all_h5_paths())
    assert number_of_unmigrated_files == 0, 'Some files could not been migrated!'
