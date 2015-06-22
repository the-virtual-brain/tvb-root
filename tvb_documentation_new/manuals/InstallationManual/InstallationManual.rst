.. |TITLE| replace:: Installation Manual
.. |DESCRIPTION| replace:: Installation Manual for |TVB|
.. |VERSION| replace:: 1.0
.. |REVISION| replace:: 0.1

.. include:: ../templates/pdf_template.rst
      

This file will guide you in installing and using The Virtual Brain project (TVB)
correctly.  For a more detailed solution to any of your problems related to TVB
software, please contact the mailing list at tvb-users@googlegroups.com.


1. INSTALL REQUIREMENTS
=======================

TVB has been packaged into stand-alone distributions, and using one of these
distributions is often the most convenient choice because it does not require
downloading or installing anything else. 

Running TVB from the sources requires setting up a Python environment with the
various dependencies, including standard scientific libraries such as NumPy, 
SciPy, MatPlotLib, etc. Where available (Mac OS X, Windows & Linux), the Anaconda
Python distribution (http://continuum.io/downloads) is the easiest way to get
started, as it provides many of the tools. 

Next, obtain TVB sources via the GitHub repo pack containing all the necessary
module and a few useful tools:

`git clone https://github.com/the-virtual-brain/tvb-pack`

Next, initialize and update the submodules

`git submodule init`
`git submodule update`

The easiest way to set up the dependencies is to create a virtual environment
with the provided script:

`scripts/mkenv.sh my_env_path`

This environment can be activated with 

`source scripts/env.sh my_env_path`

after which TVB should function correctly. 

The extra dependencies can otherwise be installed all at once via `pip`

`pip install cherrypy formencode sqlalchemy sqlalchemy-migrate genshi simplejson cfflib networkx nibabel apscheduler mod_pywebsocket psutil minixsv h5py`

where `pip` is a standard Python package manager.

If you have a non-standard situation, please try installing the packages by hand:

- Python 2.7.1 : other versions of python might not work as expected, so we recommend that you have this version installed with all the dependencies. You should find instructions on installing Python here: http://diveintopython.org/installing_python/

- CherryPy 3.2 : you can download it from here: http://www.cherrypy.org/wiki/CherryPyDownload

- FormEncode 1.2.3 or 1.2.4 : these are the versions the product was tested on, other versions might work also. Using python's 'easy_install' should bring you a good version, otherwise you can download and install it from here: http://pypi.python.org/pypi/FormEncode

- SQLAlchemy-Migrate 0.6.6 or 0.7.1 : 'easy_install' should work for this too, otherwise download and install from: http://www.sqlalchemy.org/download.html

- SQLAlchemy 0.7.1 : doing an 'easy_install sqlalchemy-migrate' should work. You can also download and install from: http://code.google.com/p/sqlalchemy-migrate/downloads/list

- Genshi 0.6 : again this should be available trough easy_install. You can also find the sources here: http://genshi.edgewall.org/wiki/Download

- SimpleJSON 2.1.3 or 2.1.6 : available trough easy_install or from: http://pypi.python.org/pypi/simplejson/2.1.3

- Matplotlib 1.0.1 : easy_install and pip might not work. You can find a step by step install procedure here: http://matplotlib.sourceforge.net/users/installing.html

- Numpy 1.5.1 or 1.6.0 and Scipy 0.9: you can find them here: http://www.scipy.org/Download

- Scikits.learn 0.8.1 : easy_install might work. You can also find it here: http://pypi.python.org/pypi/scikits.learn#downloads

- Numexpr 1.4.2 : easy_install should work, otherwise sources are found here: http://pypi.python.org/pypi/numexpr

- Cfflib 2.0.2 or 2.0.5 : you can find the sources here: http://pypi.python.org/pypi/cfflib/2.0.5

- NetworkX 1.4 : easy_install or pip should work. You can also find sources here: http://pypi.python.org/pypi/networkx

- Nibabel 1.0.1 or 1.0.2 : easy_install or pip should work. You can also find sources here: http://pypi.python.org/pypi/nibabel

- APScheduler 2.0.2 : easy_install or pip should work. You can also find sources here: http://pypi.python.org/pypi/APScheduler/

- PyWebSocket 0.7 : easy_install/pip mod_pywebsocket, or download code from http://code.google.com/p/pywebsocket/ then execute python setup.py install

- psutil : easy_install, pip

- minixsv (for genxmlif) : easy_install, pip

- h5py: easy_install, pip

In order to run the tests, ``pip install BeautifulSoup``.

2. OTHER REQUIREMENTS
=====================

A. Modern web browser
---------------------

The Virtual Brain's web interface uses several newer technologies for the
web, including Scalable Vector Graphics and WebGL, so you will need a browser
that supports these standards. The following browsers have been tested and 
are available for most platforms:

	- Google Chrome (http://www.google.com/chrome?hl=en), versions 17 - 23
	- Safari 5.1.1 (with Developer -Enable WebGL checked)
	- and Mozilla Firefox (http://www.mozilla.org/en-US/firefox/new/), version 17 - 18.

Because current versions of Internet Explorer do not support the required standards, it is
not advised or support to use Internet Explorer.

Some other browsers might work but the browser should support WebGL as a
requirement for part of the visualisers.  A list of browsers that support WebGL
is found at the following link: http://en.wikipedia.org/wiki/WebGL.  Further
details about WebGL can be found here: http://learningwebgl.com/blog/?p=11 .

Your default browser will be used when starting the application.  If you have
an unsupported browser as default, you will need to copy the URL
(http://127.0.0.1:8080/ by default) into a supported browser.

B. Octave or Matlab (optional)
------------------------------

Some of our analyzers (Brain Connectivity Toolbox) need Matlab or Octave to be installed and added to system path(available directly from console).
If this is not the case, these specific analyzers will be hidden from the UI, as they can not be executed.
You do not need to manually "link" between TVB and Matlab, that will be done behind the scene, you only need to have Matlab available in PATH.
https://sites.google.com/a/brain-connectivity-toolbox.net/bct/Home

C. WebSocket compatible browser
-------------------------------

Most of the modern browsers have this capability.


3. START / STOP PROCEDURE
=========================

Starting the application is done by executing a "tvb start" script (name and form depends on the current OS - see bellow).
After executing the script, a browser interface should be fired. In case that does not happen, you might see an error in file ~/TVB/TVB_Logger.
In case you are accessing the application from a different machine, or the default browser is not found, you could manually type in your prefered browser:

  - http://[localIPwhereTVBwasInstalled]:8080/setting/settings
  - validate that the default settings are ok for you, or change some (see bellow some explanations).
  - in case you want to be able to access the application from a different machine, make sure you change "server IP" from "127.0.0.1" into your IP, or a name that you can use for accessing the machine where TVB is installed.
    In case you do not change this IP, when accessing from a different machine, you will still be able to access most of TVB pages, except for visualizers that are using WebSockets.
  - TVB_STORAGE - this is a local path on your file system that will be used to store TVB related projects and the sqllite database.
              This must be a valid path on your file system and you need to have write access on it.
              Default value it's ~/TVB. A log files will be placed there also: ~/TVB/TVB_logger.log
              Make sure the user you start TVB with can create this folder, or change the parameter towards an empty folder you have write access to!!!
  - DB_URL - this is the  database that will be used. In case you want to choose Postgres, you need to install that separately yoruself.
  - RPC_SERVER_PORT, WEB_SERVER_PORT - the ports that will be used by the backend server and the cherrypy server.
  	These must be valid ports and not used by any other process.
  - press Apply button.
  - you will be redirected to a new URL. If that does not happen, you can try it yourself: http://[serverIP]:[new port if you changed it, or default 8080]/tvb

A. MAC OS
---------
In order to start the application simply double click 'tvb.app'. This will run the application with default settings and start a default browser with the starting URL. 
You can also start an IDLE console with the proper environment to use TVB from a console mode by using the 'tvb_command' or 'tvb_library' in 'bin' folder.

If by some reason you need to reset the database (for example when an important release is done, and previous DB structure can not be reused - highly not probable to happen), 
you can start with a fresh database by running the 'tvb_clean.command' and after that start normally.
NOTE: This will reset you database and delete all your data folders so you will lose all previous data unless you have a previous backup.

The application might still will run in background. In order to stop it you need to use the 'tvb_stop.command'.

B. WINDOWS
----------

In order to start the application simply execute tvb_start.bat. This will start a command terminal and run the application with default settings. 
You can also start an IDLE console with the proper environment to use TVB from a console mode by executing 'tvb_console.bat' or 'tvb_library.bat' in 'bin' folder.

If by some reason you need to reset the database (for example when an important release is done, and previous DB structure can not be reused - highly not probable to happen) 
you can start with a fresh database by running the 'tvb_clean.bat' file located in 'bin' folder, and after that start normally.
NOTE: This will reset you database and delete all your data folders so you will lose all previous data unless you have a previous backup.

In order to stop the application all you need to do is close the terminal that was started initially.

! When starting with a different user than the Administrator, on windows 7, you will be asked if you give permission to Python (the main programming language used in TVB) 
to start a server. You should answer positively if you want to have TVB running locally.

C. LINUX
--------

In order to start the application simply execute: 'sh tvb_start.sh' in 'bin' folder of TVB Distribution.
This will run the application with default settings and start a default browser with the starting URL. 
You can also start an IDLE console with the proper environment to use TVB from a console mode by using the 'tvb_command.sh' or 'tvb_library.sh'.

If by some reason you need to reset the database you can start with a fresh database by executing 'sh tvb_clean.sh' and after that start normally.
(NOTE: This will reset you database and delete all your data folders so you will lose all previous data unless you have a backup)

In order to stop them you need to use the script file located at 'sh tvb_stop.sh'.


4. DEFAULT DATA
===============

The Virtual Brain comes with a set of example structural data that you can use.
However, in order to use all this data you will need as a prerequisite to:

    - login with the default user admin/pass
    - go to "project" area, create a new project, save it
    - same "project" area, select "Data Structure" link near you newly created project
    - button "CFF Importer" on the right
    - select "demo_data/dataset_XX.cff" from TVB distribution package you've previously downloaded and unpacked.
    - you should see now one Connectivity, one or 2 surfaces and one Cortex entities in your Project Structure.

However, with each new user created, a "default_project" is created, and it has a full CFF uploaded inside.

