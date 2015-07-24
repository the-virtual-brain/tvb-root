.. |TITLE| replace:: TVB Contributors Manual
.. |DESCRIPTION| replace:: Provides a tutorial with the steps you need to take in order to start contributing into TVB code, as well as a demo of using TVB Framework in console mode.
.. |VERSION| replace:: 1.0
.. |REVISION| replace:: 3

.. _TVB Web Page: http://www.thevirtualbrain.org
.. _TVB Git Repository: https://github.com/the-virtual-brain/tvb-library


TVB Contributors manual
=======================

The purpose of this document is to provide a step by step flow that will get TVB set-up and ready for code contributions, 
and to offer a short tutorial of how TVB can be used from a console script with storage activated as part of TVB framework.
Current document is created and maintained by developers. You should expect changes as the contribution flow evolves.
For details about TVB architecture, please check the adjacent document, or send us an email at tvb.admin@thevirtualbrain.org.


GIT Setup
---------

The following steps assume you have the latest distribution package from `TVB Web Page`_. You also need to make sure you have a GIT client installed and available from command line. 
You can test this by trying to execute *git --version*.

You should access `TVB Git Repository`_. The first step you need to do is to create a free GitHub account, then fork TVB-Library repository for your account.

.. figure:: images/fork_repo.jpg
   :align: center

   Fork `TVB Git Repository`_

You should now have your own git clone of TVB-Library. 

Now, assuming you have your TVB Distribution package unpacked in a folder *TVB_Distribution*, go into *TVB_Distribution/bin*.
Depending on the operating system you are using, open a terminal or command line prompt in this directory and then execute the following:

- On Unix systems: *sh contributor_setup.sh ${github_url}*

- On Windows machines: *contributor_setup.bat ${github_url}*

In the commands above replace *${github_url}* with the URL of your previously forked repository on GitHub.

.. figure:: images/clone_repo.jpg
   :align: center

   Clone your TVB fork

The steps above should create a folder *TVB_Distribution/scientific_library* which contains Simulator, Analyzers, Basic and DataTypes subfolders. 
This is a clone of your previously Git forked repository. You are now ready to contribute to TVB. Good luck!

NOTE: Each time you do a clean of TVB using the tvb_clean.sh script, make sure to re-run the above described commands in order to re-initialize TVB_PATH properly. This will give you some GIT related warning which you can just ignore.


Contribution guidelines
-----------------------

- By default, the only branch available is 'trunk'. You should **always** create a separate branch with a self-explanatory name for the new features you want to add to TVB. In order to do this just execute (from *TVB_Distribution/scientific_library* folder): *git checkout {my-awesome-new-feature-url}*. 

.. figure:: images/create_branch.jpg
   :align: center

   Create a new branch within your local cloned repo


- While making your modifications/contributions, make sure that 1) you are working in the right branch and 2) you make pull requests from master ('trunk') often, in order to quickly solve any conflicts which might appear.

- If you have problems, send us an email, and we will do our best to help you.

- You should put explanatory comments and documentation in your code.

- You should attach unit-tests for your new code, to prove that it is correct and that it fits into the overall architecture of TVB.

- Once you are done with your changes and you believe that they can be integrated into TVB master repository, go to your GitHub repository, switch to your feature branch and issue a *pull request*, describing the improvements you did. We will later test that your changes are fit to be included, and notify you of the integration process.


	
Use TVB Framework from Console Interface
----------------------------------------
	
You can use TVB Framework Distribution package through two different interfaces: one is the web interface 
(where you will have an HTML face with buttons and pages for manipulating TVB objects), and the other is through a console interface.

For the second available interface, TVB Distribution package comes with a working IDLE Console which you can use as a low-level alternative to the Web Interface. 
In the console interface you can directly play with TVB Python objects and execute the same steps as in the web interface, or more; but you need to have programming knowledge.


Examples of using TVB Console without Storage (Library Profile)
***************************************************************

TVB Console can work in two manners: with or without storage enabled. We call the mode with storage enabled *Command Profile*, and the one without storage *Library Profile*.
You can fire one of the two profiles, by launching the corresponding command file from inside *TVB_Distribution/bin*: *tvb_command* or *tvb_library*

When storage is disabled (in Library Profile), the manipulation of objects is entirely at your hand. 
When you close the console any computed data will be lost, unless you store results yourself.

Examples of how TVB Library Profile is in action, are found in folder *TVB_Distribution/tvb_scientific_library/tvb/simulator/demos/*. Please try them, in the IDLE console of TVB.

Even more details about TVB Library Profile (without storage) are found in file *TVB_Distribution/tvb_scientific_library/tvb/simulator/README.txt*.


Example of using TVB Console with Storage enabled (Command Profile)
*******************************************************************

You can launch TVB Command profile (with File and DB storage enabled) by using *distribution* in *bin* folder.

You can find examples of Command profile usage at https://github.com/the-virtual-brain/tvb-framework/tree/master/tvb/interfaces/command/demos
