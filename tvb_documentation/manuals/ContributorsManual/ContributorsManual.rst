.. |TITLE| replace:: TVB Contributors Manual
.. |DESCRIPTION| replace:: Provides a tutorial with the steps you need to take in order to start contributing into TVB code, as well as a demo of using TVB Framework in console mode.
.. |VERSION| replace:: 1.0
.. |REVISION| replace:: 3

.. |open_issues| raw:: html

   <a href="http://req.thevirtualbrain.org/issues/?filter=10421" target="_blank">open issues</a>


.. _TVB Web Page: http://www.thevirtualbrain.org
.. _TVB Library Repository: https://github.com/the-virtual-brain/tvb-library
.. _mailing list: https://groups.google.com/forum/#!forum/tvb-users
.. _contributors_manual:

TVB Contributors manual
=======================

So you want to contribute to TVB. Maybe add a model or another feature.
Thank you for helping! We welcome and appreciate contributions.

Get in touch with the |TVB| team, we are glad to help tvb.admin@thevirtualbrain.org.

Sign up for the `mailing list`_ and introduce yourself.

Read trough these docs. Get to know TVB by installing a TVB distribution and playing with the GUI or following tutorials.

Have a look at the open tasks in our Jira |open_issues|.

Finally revisit this document and find out how to set up |TVB| for a developer.


The source code
---------------

First you will need to fork the source code.
You will need to have git installed. And we recommend a github account as it makes collaboration easy.

TVB's source is hosted on `github <https://github.com/the-virtual-brain>`_ .

It is spread in several repositories. These are the most important:

* tvb-library contains the scientific code.
* tvb-framework contains data management services and the web interface of TVB.
* tvb-data contains demonstration data.

Fork the `TVB Library Repository`_ for your account.

.. figure:: images/fork_repo.jpg
   :align: center


If you want to contribute to the framework then fork that repository as well.

Do not clone your forks yet, read about the contributor setup.


The contributor setup
---------------------

You can just clone your forks then install |TVB|'s distutils packages.
That approach is described in `The unaided setup`_.
It seems easy but |TVB| has some heavy dependencies.
To avoid having contributors deal with installing those we have created the contributor setup.

In the contributor setup you will have to :ref:`install <installing_tvb>` the latest |TVB| distribution.
This is the same install that end users will use.

Then use a special script to clone the repositories you want to modify.
This setup will use the python and the dependencies from the |TVB| distribution, sidestepping the need to install them.
You will run |TVB| from the distribution and the changes you have made to your local git repo will be visible.
This works by placing your repository in PYTHONPATH ahead of the code from the distribution.

Below are the commands for getting a contributor setup for the tvb-library.
You should do the same for tvb-framework if you need to change that.

The commands below are for Linux, adapt the extensions for your operating system.
Also replace [github_account] with your github account name to get a valid url to your fork.

Assuming you have your TVB Distribution package unpacked in a folder ``TVB_Distribution`` run:

.. code-block:: bash

   $ cd TVB_Distribution/bin
   $ sh contributor_setup.sh https://github.com/[github_account]/tvb-library.git

The steps above will create a folder *TVB_Distribution/tvb-library*.
This is a clone of your forked repository. You are now ready to contribute to TVB. Good luck!

NOTE: Each time you do a clean of TVB using the tvb_clean.sh script, make sure to re-run the above described commands in order to re-initialize TVB_PATH properly. This will give you some GIT related warning which you can just ignore.


The unaided setup
-----------------

.. _anaconda: https://store.continuum.io/cshop/anaconda/
.. _virtualenv: https://virtualenv.pypa.io/en/latest/index.html

The contributor setup avoids having to deal with dependencies. But you might want to do exactly that, adding a dependency to |TVB| or changing the ones it already has.

The unaided setup is the usual way to install python packages.

Clone the repositories (after forking them in a github account of your own), noting that now it is likely that you will need all three.

.. code-block:: bash

   $ cd my_tvb_workspace
   $ git clone https://github.com/[github_account]/tvb-library.git
   $ # these might be optional
   $ git clone https://github.com/[github_account]/tvb-framework.git
   $ git clone https://github.com/[github_account]/tvb-data.git

|TVB| depends on numpy and scipy, heavy native libraries.
If you can please install them using you operating system package manager.
On Linux apt-get, yum, dnf etc.

.. code-block:: bash

   $ sudo yum install Cython numpy scipy

If such native package managers are not available please install the `anaconda`_ python distribution and use TVB with it.

If you leave the installation of these dependencies to distutils then it will try to compile them from source.
For that to work you will need C and Fortran compilers, and development libraries, not an easy task.

Using a virtual python environment is a good idea.
For vanilla python get `virtualenv`_ then create and activate an enviroment:

.. code-block:: bash

   $ virtualenv tvb_venv
   $ source tvb_venv/bin/activate

Anaconda has it's own way of creating environments, see `anaconda`_ site.


Now to install the |TVB| packages in develop mode using distutils :

.. code-block:: bash

   $ cd my_tvb_workspace
   $ cd scientific_library
   $ python setup.py develop
   $ cd ../framework_tvb
   $ python setup.py develop


Support
-------

If you have problems, send us an email, and we will do our best to help you.
You can see open issues on TVB's Jira |open_issues|. You may also create a new issue.


Test suite
----------

TVB's test suite takes a long time to run, but a patch will have to pass it.
We recommend running tests before submitting changes that touch code that you have not written.

.. code-block:: bash

   $ cd my_tvb_workspace/tvb_bin
   $ sh run_tests.sh


Contribution guidelines
-----------------------

You should put explanatory comments and documentation in your code.
Document every public function with a docstring.
Use english for both comments and names.

Avoid cryptic short names. You may relax this if implementing a mathematical formula.
But then please document it using latex docstrings.

Try to adhere to the Python code style. Indent with 4 spaces. We are ok with 120 long lines.
Naming: module_name, ClassName, function_name, CONSTANT_NAME function_parameter_name, local_var_name

You should attach unit-tests for your new code, to prove that it is correct and that it fits into the overall architecture of TVB.

Prefer small commits. Add a meaningful commit message.
We strongly recommend that the commit message start with the Jira task id. (e.g. TVB-1963 Add FCT analyser).

Use logging instead of print statements.

If code is indented more than 6 levels your function is too complex.
If a function has more than 50 lines it is too long. Split these functions.

Do not copy paste code.
Avoid reinventing the wheel. Use the python built in functions, the standard library and numpy.


Git guidelines
--------------

By default, the only branch available is 'trunk'. You should **always** create a separate branch with a self-explanatory name for the new features you want to add to TVB.
In order to do this assuming you are using the contributor setup do :

.. code-block:: bash

   $ cd TVB_Distribution/scientific_library
   $ git checkout -b my-awesome-new-feature-url


While making your modifications/contributions, make sure that

1) you are working in the right branch and
2) you make pull requests from master ('trunk') often, in order to quickly solve any conflicts which might appear.
3) You follow the `Contribution guidelines`_

Once you are done with your changes and you believe that they can be integrated into TVB master repository, go to your GitHub repository,
switch to your feature branch and issue a *pull request*, describing the improvements you did.
We will later test that your changes are fit to be included, and notify you of the integration process.


Tools
-----

We use pycharm to develop and debug TVB.
To test quick ideas we like ipython.


Technologies used by TVB
------------------------

TVB uses numpy extensively.
Numpy is quite different from other python libraries.
Learn a bit about it before trying to understand TVB code.

The TVB framework uses sqlalchemy for ORM mapping, cherrypy as a web framework and server and genshi for html templating.
Numeric arrays are stored in the hdf5 format.
Client side we use jquery, d3 and webgl.

TVB uses some advanced python features to implement it's `Traits` system: metaclasses and data descriptors.


Glossary of terms used by TVB code
----------------------------------

Datatype:

   The way TVB represents data. Similar to entities in a database model.
   They usually contain numeric arrays.
   Many algorithms receive and produce Datatypes.

   Tvb framework organizes them into projects, stores the numeric data in .h5 files and metadata in MAPPED_TYPE... tables in a database.

   Example: Surface, Connectivity
   Code: scientific_library/tvb/datatypes/

Adapter:

   A TVB framework plugin, similar to a runnable task. It has a launch method.
   It declares what inputs it requires and what Datatypes it produces.
   Asynchronous Adapters will be run in a different process, possibly on a cluster.

   Adapters may be of different types: analysers, creators, uploaders, visualizers

   These plugins are discovered at TVB startup and recorded in the database table ALGORITHMS.

   Example:  SimulatorAdapter
   code: framework_tvb/tvb/adapters

Operation:

   Running an Adapter produces an Operation. It will contain the Datatypes produced by the Adapter.

Project:

   Organizes the data of an user. It will contain all Operations and Datatypes.
   Stored on disk in ~/TVB/PROJECTS. The numerically named folders correspond to operations with that id, the h5 files in them correspond to datatypes.

