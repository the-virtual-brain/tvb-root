.. |TITLE| replace:: TVB Contributors Manual
.. |DESCRIPTION| replace:: Provides a tutorial with the steps you need to take in order to start contributing into TVB code, as well as a demo of using TVB Framework in console mode.
.. |VERSION| replace:: 1.0
.. |REVISION| replace:: 3

.. _TVB Web Page: http://www.thevirtualbrain.org
.. _TVB Library Repository: https://github.com/the-virtual-brain/tvb-library

.. _contributors_manual:

TVB Contributors manual
=======================

So you want to contribute code to TVB. Maybe add a model or another feature.
Thank you for helping |TVB|! We welcome and appreciate contributions.
Please get in touch with the |TVB| team, we are glad to help tvb.admin@thevirtualbrain.org.

This document describes how to set up |TVB| for a developer.


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
If such native package managers are not available please install the `anaconda`_ python distribution and use TVB with it.

If you leave the installation of these dependencies to distutils then it will try to compile them from source.
For that to work you will need C and Fortran compilers, and development libraries, not an easy task.

Now to install the |TVB| packages in develop mode using distutils :

.. code-block:: bash

   $ cd my_tvb_workspace
   $ cd scientific_library
   $ python setup.py develop
   $ cd ../framework_tvb
   $ python setup.py develop


Contribution guidelines
-----------------------

By default, the only branch available is 'trunk'. You should **always** create a separate branch with a self-explanatory name for the new features you want to add to TVB.
In order to do this assuming you are using the contributor setup do :

.. code-block:: bash

   $ cd TVB_Distribution/scientific_library
   $ git checkout -b my-awesome-new-feature-url


While making your modifications/contributions, make sure that

1) you are working in the right branch and
2) you make pull requests from master ('trunk') often, in order to quickly solve any conflicts which might appear.

If you have problems, send us an email, and we will do our best to help you.

You should put explanatory comments and documentation in your code.

You should attach unit-tests for your new code, to prove that it is correct and that it fits into the overall architecture of TVB.

Once you are done with your changes and you believe that they can be integrated into TVB master repository, go to your GitHub repository,
switch to your feature branch and issue a *pull request*, describing the improvements you did.
We will later test that your changes are fit to be included, and notify you of the integration process.

