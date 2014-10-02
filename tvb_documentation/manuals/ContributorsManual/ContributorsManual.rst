.. |TITLE| replace:: TVB Contributors Manual
.. |DESCRIPTION| replace:: Provides a tutorial with the steps you need to take in order to start contributing into TVB code, as well as a demo of using TVB Framework in console mode.
.. |VERSION| replace:: 1.0
.. |REVISION| replace:: 3

.. include:: ../templates/pdf_template.rst   
      
.. _TVB Web Page: http://www.thevirtualbrain.org
.. _TVB Git Repository: https://github.com/tvb-admin/tvb_scientific_library


TVB Contributors manual
=======================

The purpose of this document is to provide a step by step flow that will get TVB set-up and ready for code contributions, 
and to offer a short tutorial of how TVB can be used from a console script with storage activated as part of TVB framework.
Current document is created and maintained by developers. You should expect changes as the contribution flow evolves.
For details about TVB architecture, please check the adjacent document, or send us an email at tvb.admin@thevirtualbrain.org.


GIT Setup
-----------	

The following steps assume you have the latest distribution package from `TVB Web Page`_. You also need to make sure you have a GIT client installed and available from command line. 
You can test this by trying to execute *git --version*.

You should access `TVB Git Repository`_. The first step you need to do is to create a free GitHub account, then fork TVB-Library repository for your account.

.. figure:: images/fork_repo.jpg
   :width: 90%
   :align: center

   Fork `TVB Git Repository`_

You should now have your own git clone of TVB-Library. 

Now, assuming you have your TVB Distribution package unpacked in a folder *TVB_Distribution*, go into *TVB_Distribution/bin*.
Depending on the operating system you are using, open a terminal or command line prompt in this directory and then execute the following:

- On Unix systems: *sh contributor_setup.sh ${github_url}*

- On Windows machines: *contributor_setup.bat ${github_url}*

In the commands above replace *${github_url}* with the URL of your previously forked repository on GitHub.

.. figure:: images/clone_repo.jpg
   :width: 90%
   :align: center

   Clone your TVB fork

The steps above should create a folder *TVB_Distribution/scientific_library* which contains Simulator, Analyzers, Basic and DataTypes subfolders. 
This is a clone of your previously Git forked repository. You are now ready to contribute to TVB. Good luck!

NOTE: Each time you do a clean of TVB using the tvb_clean.sh script, make sure to re-run the above described commands in order to re-initialize TVB_PATH properly. This will give you some GIT related warning which you can just ignore.


Contribution guidelines
------------------------

- By default, the only branch available is 'trunk'. You should **always** create a separate branch with a self-explanatory name for the new features you want to add to TVB. In order to do this just execute (from *TVB_Distribution/scientific_library* folder): *git checkout {my-awesome-new-feature-url}*. 

.. figure:: images/create_branch.jpg
   :width: 90%
   :align: center

   Create a new branch within your local cloned repo


- While making your modifications/contributions, make sure that 1) you are working in the right branch and 2) you make pull requests from master ('trunk') often, in order to quickly solve any conflicts which might appear.

- If you have problems, send us an email, and we will do our best to help you.

- You should put explanatory comments and documentation in your code.

- You should attach unit-tests for your new code, to prove that it is correct and that it fits into the overall architecture of TVB.

- Once you are done with your changes and you believe that they can be integrated into TVB master repository, go to your GitHub repository, switch to your feature branch and issue a *pull request*, describing the improvements you did. We will later test that your changes are fit to be included, and notify you of the integration process.


	
Use TVB Framework from Console Interface
-----------------------------------------
	
You can use TVB Framework Distribution package through two different interfaces: one is the web interface 
(where you will have an HTML face with buttons and pages for manipulating TVB objects), and the other is through a console interface.

For the second available interface, TVB Distribution package comes with a working IDLE Console which you can use as a low-level alternative to the Web Interface. 
In the console interface you can directly play with TVB Python objects and execute the same steps as in the web interface, or more; but you need to have programming knowledge.


Examples of using TVB Console without Storage (Library Profile)
******************************************************************

TVB Console can work in two manners: with or without storage enabled. We call the mode with storage enabled *Command Profile*, and the one without storage *Library Profile*.
You can fire one of the two profiles, by launching the corresponding command file from inside *TVB_Distribution/bin*: *tvb_command* or *tvb_library*

When storage is disabled (in Library Profile), the manipulation of objects is entirely at your hand. 
When you close the console any computed data will be lost, unless you store results yourself.

Examples of how TVB Library Profile is in action, are found in folder *TVB_Distribution/tvb_scientific_library/tvb/simulator/demos/*. Please try them, in the IDLE console of TVB.

Even more details about TVB Library Profile (without storage) are found in file *TVB_Distribution/tvb_scientific_library/tvb/simulator/README.txt*.


Example of using TVB Console with Storage enabled (Command Profile)
*********************************************************************

You can launch TVB Command profile (with File and DB storage enabled) by using *tvb_command* in *bin* folder. 
An example of what you can further write in the IDLE console which will get fired is placed bellow.

Note that we'll use placeholders in the demo script below::

    $TMP_STORAGE$ = a string representing the path to a folder on your disk that will be used as temporary storage

    $ZIP_ATCHIVE_PATH$ = a string representing the path to a ZIP archive on your disk with a valid Connectivity DataType.


The following is a demo script for using TVB Command mode for various operations, with storage mode enabled ::

    ## First lines should always be setting use_storage flag to true if you want to persisted data.
	from tvb.basic.profile import TvbProfile
    TvbProfile.set_profile(TvbProfile.COMMAND_PROFILE)
	
    ## We need and user and a project. You can initialize your database with
	## the following. Care this will also reset all the data from your database.
	from tvb.core.services.initializer import reset, initialize
    reset()
    initialize(['tvb.config'])
	
    ## Import necessary libraries
	import numpy
    from tvb.basic.filters.chain import FilterChain
    from tvb.core.traits import db_events
    from tvb.core.entities import model
    from tvb.core.entities.storage import dao
    from tvb.core.services.operation_service import OperationService
    from tvb.core.adapters.abcadapter import ABCAdapter
    from tvb.simulator import simulator, models, coupling, integrators, monitors, noise
	
    db_events.attach_db_events()
	
    ## We need a user and a project in order to run operations and store results
	## A default user named 'admin' should already exist in the system.
	user = dao.get_user_by_name('admin')
    project = model.Project('my_project_name', user.id)
    project = dao.store_entity(project)
	
    ##----------------------------------------------------------------------------##
	##-                      Import a Connectivity ZIP                           -##
	##----------------------------------------------------------------------------##
	## Here is an example of how one would launch a TVB uploader:
	tmp_storage = $TMP_STORAGE$
    launcher = OperationService()
    algo_group = dao.find_group('tvb.adapters.uploaders.zip_connectivity_importer', 'ZIPConnectivityImporter')
    adapter = ABCAdapter.build_adapter(algo_group)
    launch_args = {'uploaded' : $ZIP_ATCHIVE_PATH$}
    result = launcher.initiate_operation(user, project.id, adapter, tmp_storage, **launch_args)
    ## You should see as output: "Operation X has finished" where X is the operation ID
	## You can later on use the Operation ID, to get resulted datatypes if you want.
	op_id = [int(s) for s in result.split() if s.isdigit()][0]
	
    ## Lets retrieve the result of the upload operation: in this case a Connectivity; and edit an attribute on the Connectivity.
	conn_result = dao.get_results_for_operation(op_id)[0]
    conn_result.subject = "My fancy subject"
    dao.store_entity(conn_result)
	
    ## We can also retrieve a datatype by specific filters .
	## E.g. let's get all the connectivities for a specific subject
	from tvb.datatypes.connectivity import Connectivity
    dt_filter = FilterChain(fields = [FilterChain.datatype + '.subject'], operations=["=="], values=['My fancy subject'])
    returned_values = dao.get_values_of_datatype(project.id, Connectivity, dt_filter)
    print "Got from database values: %s" %(returned_values,)
	
    ##----------------------------------------------------------------------------##
	##-                      Perform a Simulation                                -##
	##----------------------------------------------------------------------------##
	## Configuring: Initialise a Model, Coupling, and Connectivity.
	oscilator = models.Generic2dOscillator(a=1.42)
    ## You can also load a datatype if you have the GID or ID for it.
	white_matter = ABCAdapter.load_entity_by_gid(returned_values[0][2])   
    white_matter.speed = numpy.array([4.0])
    white_matter_coupling = coupling.Linear(a=0.016)
	
    ## Initialise an Integrator
	hiss = noise.Additive(nsig = numpy.array([2**-10,]))
    heunint = integrators.HeunStochastic(dt=0.06103515625, noise=hiss)
	
    ## Initialise a Monitor with period in physical time
	what_to_watch = monitors.TemporalAverage(period=0.48828125) 
	
    ## Initialise a Simulator -- Model, Connectivity, Integrator, and Monitors.
	sim = simulator.Simulator(model=oscilator, connectivity=white_matter,
	                          coupling=white_matter_coupling, integrator=heunint, monitors=what_to_watch)
    sim.configure()
    ## Perform the simulation
	tavg_data = []
    tavg_time = []
    ## Starting simulation
	for tavg in sim(simulation_length=1600):
        if tavg is not None:
            tavg_time.append(tavg[0][0])
            tavg_data.append(tavg[0][1]) ## The first [0] is the first monitor's result
	
    ##----------------------------------------------------------------------------##
	##-                      Persist TimeSeries DataType                         -##
	##----------------------------------------------------------------------------##
	## At this point the simulation computation is done but data is not yet stored.
	## We have the data computed, but it's still transient unless we store it in a TimeSeries DataType.
	import tvb.datatypes.time_series as time_series
    data_result = time_series.TimeSeriesRegion()
    ## Need an operation for each datatypes that is to be stored in database. Just use dummy operation here for demo purposes.
	data_result.set_operation_id(1) 
    ## A TimeSeries needs a reference to the connectivity that was generated.
	data_result.connectivity = white_matter
    data_result.write_data_slice(tavg_data)
    data_result.write_time_slice(tavg_time)
    data_result.close_file()
    ## Saving simulator result into DB
	dao.store_entity(data_result)
    ## Loading from db to check results are properly stored
	loaded_dt = ABCAdapter.load_entity_by_gid(data_result.gid)
    print "Time shape is %s" % (loaded_dt.get_data_shape('time'),)
    print "Data shape is %s" % (loaded_dt.get_data_shape('data'),)
    print loaded_dt.get_data('time')
    print loaded_dt.get_data('data')


.. raw:: pdf

   PageBreak

.. COPYRIGHT

.. include:: ../templates/copyright_notice.rst
