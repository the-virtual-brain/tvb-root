.. |TITLE| replace:: TVB Contributors Manual
.. |DESCRIPTION| replace:: Provides a tutorial with the steps you need to take in order to start contributing into TVB code.
.. |VERSION| replace:: 1.1
.. |REVISION| replace:: 1

.. |open_issues| raw:: html

   <a href="http://req.thevirtualbrain.org/issues/?filter=10421" target="_blank">open issues</a>


.. _TVB Web Page: http://www.thevirtualbrain.org
.. _mailing list: https://groups.google.com/forum/#!forum/tvb-users
.. _contributors_manual:

TVB Contributors manual
=======================

So you want to contribute to TVB, or simply use the latest code from sources.
We welcome contributions, or using for testing. Thank you!

Get in touch with the |TVB| team, we are glad to help tvb.admin@thevirtualbrain.org.
Sign up for the `mailing list`_ and introduce yourself.

Read trough these docs. Get to know TVB by installing a TVB Distribution
(from `TVB Web Page`_) and playing with the GUI or following tutorials first.

Have a look at the open tasks in our Jira |open_issues|.

Finally revisit this document and find out how to set up |TVB| as a developer.


Source code and working environment
------------------------------------

.. _github: https://github.com/the-virtual-brain/tvb-root
.. _anaconda: https://store.continuum.io/cshop/anaconda/

TVB's source code is hosted on `github`_ . Fork the **tvb-root** into your account.
You will need to have `git` installed locally and a Github account prepared. Then clone the repo locally.

For preparing the local work environment, as well as for linking TVB sources into it,
the following steps should be done.
We recommend preparing your local Python environment for TVB with `anaconda`_.
Using a virtual environment inside Anaconda is a good idea.

.. code-block:: bash

   $ envname="tvb-run"
   $ conda create -y --name $envname python=3 nomkl numba scipy numpy networkx scikit-learn cython pip numexpr psutil psycopg2 pytables scikit-image==0.14.2 simplejson cherrypy docutils werkzeug matplotlib-base
   $ source activate $envname
   $ consta install -c conda-forge jupyterlab flask gevent
   $ pip install h5py formencode cfflib jinja2 nibabel sqlalchemy sqlalchemy-migrate allensdk tvb-gdist typing BeautifulSoup4 subprocess32 flask-restplus python-keycloak mako
   $ cd [tvb-root]/tvb_build/
   $ sh install_full_tvb.sh

The above list of dependencies might change. We recommend you to check the latest version here:
https://github.com/the-virtual-brain/tvb-root/tree/master/tvb_build/docker

In case you do not want the sources, but only the latest released version of TVB, we recommend you to use Pypi:

.. code-block:: bash

   $ source activate $envname
   $ pip install tvb-library               # In case you do not need the web interface. not storage, you can stop here
   $ pip install tvb-framework
   $ python -m tvb.interfaces.web.run WEB_PROFILE  # Launch TVB web server locally


Support
-------

If you have problems, send us an email, and we will do our best to help you.
You can see open issues on TVB's Jira |open_issues|. You may also create a new issue.


Test suite
----------

TVB's test suite takes a long time to run, but a patch will have to pass it.
We recommend running tests before submitting changes that touch code that you have not written::

   $ pip install pytest pytest-cov pytest-benchmark pytest-mock
   $ pytest --pyargs tvb.tests.library
   $ pytest --pyargs tvb.tests.framework


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

By default, the only branch available is 'trunk'. You should **always** create a separate branch with a self-explanatory
name for the new features you want to add to TVB.

While making your modifications/contributions, make sure that

1) you are working in the right branch and
2) you make pull requests from master often, in order to quickly solve any conflicts which might appear.
3) You follow the `Contribution guidelines`_

Once you are done with your changes and you believe that they can be integrated into TVB master repository, go to your GitHub repository,
switch to your feature branch and issue a *pull request*, describing the improvements you did.
We will later test that your changes are fit to be included, and notify you of the integration process.


Tools
-----

We use pycharm to develop and debug TVB.
To test quick ideas we like ipython notebooks.


Technologies used by TVB
------------------------

TVB uses numpy extensively.
Numpy is quite different from other python libraries.
Learn a bit about it before trying to understand TVB code.

The TVB framework uses sqlalchemy for ORM mapping, cherrypy as a web framework and server and jinja2 for html templating.
Numeric arrays are stored in the hdf5 format.
Client side we use jquery, d3 and webgl.

TVB uses some advanced python features to implement it's `Traits` system: metaclasses and data descriptors.


Glossary of terms used by TVB code
----------------------------------

Datatype:

   The way TVB represents data. Similar to entities in a database model.
   They usually contain numeric arrays.
   Many algorithms receive and produce Datatypes.

   Tvb framework organizes them into projects, stores the numeric data in .h5 files and metadata in
   tables in a relational database (sqlite or postgresql).

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

