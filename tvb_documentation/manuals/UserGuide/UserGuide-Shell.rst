Shell Interface of |TVB|
========================


Getting started with Python IDLE 
--------------------------------

.. figure:: screenshots/linux_shell.jpg
   :width: 90%
   :align: center

   Interactive Python Shell

IDLE font size, syntax higlighting and keys can be configured according to your 
needs. Go to the Options menu in the menu bar and select `Configure IDLE`.

.. figure:: screenshots/linux_shell_idle_options.jpg
   :width: 70%
   :align: center

   Configure IDLE options.
   
   
There is a number of scripting demos to show how to build a network model and
run a simulation. 

To run any demo use the `execfile` command::

    execfile('/home/user/Downloads/TVB_Distribution/tvb_data/tvb/simulator/demos/region_deterministic.py')

The above command should work on Linux and Windows, as long as you replace '/home/user/Downloads/TVB_Distribution'
with your personal path towards the folder where TVB was being downloaded.
On Mac OS the path is just a little different::

    execfile('../Resources/lib/python2.7/tvb/simulator/demos/region_deterministic.py')
    execfile('/home/user/Downloads/TVB_Distribution/tvb.app/Contents/Resources/lib/python2.7/tvb/simulator/demos/region_deterministic.py')

Here is an illustration for the above.

.. figure:: screenshots/linux_shell_run_demo.jpg
   :width: 90%
   :align: center

   Run a demo with execfile
   
   
Another way to run a script, that also allows to see and edit the code, is opening 
the file from the File menu. A new window will pop out. Then select Run Module 
from the Run menu. The script will be executed.


.. figure:: screenshots/linux_shell_run_demo_2.jpg
   :width: 90%
   :align: center

   Run a demo from the Run Menu option



To work interactively in the Python shell you need a few modules::

    from tvb.simulator.lab import *


This will import all the scientific simulator modules as well as some datatypes
that wrap important data as the `Connectivity` matrix and cortical `Surface`.


Terminal Interface of |TVB|
===========================
If you are using |TVB| on a headless machine then Python IDLE is not an option.
To launch a python terminal in the command profile use ::

   distribution start COMMAND_PROFILE -headless

And in the library profile ::

   distribution start LIBRARY_PROFILE -headless

The scripts are located in the `bin` folder and they have platform specific terminations.

The `distribution` script can be used to launch other profiles as well.
The `DEPLOYMENT_PROFILE` will start the web interface. The following has the same effect as `tvb_start` ::

    distribution start DEPLOYMENT_PROFILE

Using the distribution script allows you to give additional options.
The `-reset` option will clean the |TVB| folder before starting the web interface ::

    distribution start DEPLOYMENT_PROFILE -reset
