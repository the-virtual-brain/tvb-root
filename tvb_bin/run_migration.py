import shutil
import os
import sys

from tvb.basic.profile import TvbProfile
from tvb.config.init.initializer import initialize
from tvb.core.entities.file.files_update_manager import FilesUpdateManager


if __name__ == '__main__':
    """
    Script written for testing the migration from version 1.5.8 to 2.1.0.
    """

    # set web profile
    TvbProfile.set_profile(TvbProfile.WEB_PROFILE)

    # migrate the database and h5 files
    h5_migrating_thread = initialize()

    # wait for thread to finish before processing
    h5_migrating_thread.join()

    # copy files in tvb_root folder so Jenkins can find them
    shutil.copytree(TvbProfile.current.TVB_LOG_FOLDER, os.path.join(TvbProfile.current.EXTERNALS_FOLDER_PARENT, 'logs'))

    # test if there are any files which were not migrated
    number_of_unmigrated_files = len(FilesUpdateManager.get_all_h5_paths())
    if number_of_unmigrated_files != 0:
        sys.exit(-1)
    sys.exit(0)
