from tvb.core.entities.file.files_update_manager import FilesUpdateManager

if __name__ == '__main__':
    number_of_unmigrated_files = len(FilesUpdateManager._get_all_h5_paths())
    assert number_of_unmigrated_files == 0, 'Some files could not been migrated'
