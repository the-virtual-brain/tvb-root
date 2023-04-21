BIDS data builder and directory monitor
=======================================

This module contains helper classes which can build BIDS datasets and import them into TVB projects. Along with this, the module also provides a way to monitor a BIDS directory and build datasets accordingly when new files are added.

BIDSDataBuilder
---------------

- Class for building BIDS dataset by providing 1) a set of json files and BIDS root directory or 2) a datatype from the subject directory (e.g. net, ts, coord, spatial)
- Contains utils for finding all dependencies of a subject json file.
- Produces a zip file.
- Takes following args
   - BIDS_ROOT_DIR (reqd.)- BIDS root dir from which dataset will be created
   - BIDS_DATA_TO_IMPORT - Accepts BIDSImporter net, ts, coord, spatial token  to build subject type specific dataset For e.g. if BIDSImporter.TS_TOKEN is provided then, it'll build a dataset containing only TS data and their dependencies
   - INITIAL_JSON_FILES - A set of initial json files with which bids dataset to be built. This is used by BIDSDirWatcher

BIDSDirWatcher
--------------

- Class for monitoring a BIDS directory and also builds datasets accordingly on new files
- Runs two threads in parallel for 1) observing a directory and 2) building/importing dataset into TVB
- Importing thread runs on a fixed interval (configurable), which also acts as a buffer time when large files are added
- Contains utils for observing specific (subjects) directories changes only
- Takes following args
    - DIRECTORY_TO_WATCH (reqd.)- BIDS root dir on which is to be monitored
    - UPLOAD_TRIGGER_INTERVAL - Importer thread interval in seconds
    - IMPORT_DATA_IN_TVB - A flag for only creating dataset and storing and not importing into TVB projects
    - TVB_PROJECT_ID - ID of TVB project on which dataset is to be imported


[launch_bids_monitor.py](tvb_framework/tvb/interfaces/rest/bids_monitor/launch_bids_monitor.py) contains sample code and it also accepts command line arguments. To start monitoring a bids directory run below command

.. code-block::

    $ python launch_bids_monitor.py --rest-url=http://localhost:9090 --bids-dir=user/doc/BIDS_SAMPLE
..

where  `-rest-url` is the url on which TVB rest server is running and `-bids-dir` is the BIDS root directory which is to be monitored for new files. Please note that bids monitor will only trigger when files are added inside subject directories.

Build BIDS dataset for time_series

.. code-block:: python

    from tvb.interfaces.rest.bids_monitor.bids_data_builder import BIDSDataBuilder
    from tvb.adapters.uploaders.bids_importer import BIDSImporter

    bids_data_builder = BIDSDataBuilder(BIDSImporter.TS_TOKEN, BIDS_DIR)
    zip_file_location = bids_data_builder.create_dataset_subjects()
    print(zip_file_location)
..

Monitor a BIDS directory for new files

.. code-block:: python

    from tvb.interfaces.rest.bids_monitor.bids_dir_monitor import BIDSDirWatcher

    bids_dir_watcher = BIDSDirWatcher(
          DIRECTORY_TO_WATCH=BIDS_DIR,
          UPLOAD_TRIGGER_INTERVAL=20,
          IMPORT_DATA_IN_TVB=True
    )
    bids_dir_watcher.init_watcher()
..

Some implementation details
---------------------------
* The module triggers an import in TVB only for new files added to the BIDS_DIR, not for the files that already exist in the BIDS_DIR at start-up
* Files are imported by default in the first project of the user
* If the imported file has dependencies, all dependencies and the file will be imported in TVB, even if the dependencies might already exist

Acknowledgments
---------------
This project has received funding from GSOC program 2022.