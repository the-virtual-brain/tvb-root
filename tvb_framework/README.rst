TVB FRAMEWORK
=============

The Virtual Brain framework is a complete framework including:

-  a plugable workflow manager;
-  a data persistence layer (with a relational DB and File Storage);
-  an HTML5 based user interface;
-  visualizers for neuro-science related entities.

The easiest way to make use of this code, is to obtain
a Distribution of TVB with Python and all the necessary packages linked,
and then clone this repo nearby. More details `in our
documentation <http://docs.thevirtualbrain.org/manuals/ContributorsManual/ContributorsManual.html>`__.

Alternatively, if you plan to develop long term with TVB, extensively
modify code, add new dependencies, or you simply prefer to use your own
Python installation, you may want to read this:
`here <http://docs.thevirtualbrain.org/manuals/ContributorsManual/ContributorsManual.html#the-unaided-setup>`__.

If you don't require the framework features listed above, the simulator
and associated scientific modules can be used independently; please see
the `tvb-library <https://github.com/the-virtual-brain/tvb-root/tree/master/tvb_library>`__
folder.

Framework structure
-------------------

::

    tvb-gdist
        |
        |
    tvb-library     tvb-data
          \          /
            \       /
            tvb.config
                |
                |
            tvb.core
                |
                |
            tvb.adapters
                |
                |
            tvb.config.init
                |
                |
            tvb.interfaces

**tvb-data** should be installed from **Zenodo**: https://zenodo.org/record/7574266/files/tvb_data.zip?download=1.
After download, unzip and execute `python setup.py develop` in the correct env.

Usage
-----

To use TVB code, clone from GitHub (https://github.com/the-virtual-brain/tvb-root), or get from Pypi::

    pip install tvb-framework
    python -m tvb.interfaces.web.run WEB_PROFILE tvb.config


Your port **8080** should be free, as a CherryPy service will try to run there.
Your default browser should automatically open http://localhost:8080/ which is the way to
interact with TVB Web Interface.

Testing
=======

For testing the package, the `Pytest  <https://docs.pytest.org/>`_
framework is used. Pytest can be installed using pip.

In addition to tvb-framework needed just for
launching the web, tests require few extra dependencies just for test (e.g. pytest-benchmark), which
need to be installed manually. The entire list of these dependencies for testing is found in `tvb_framework/setup.py`
under **extras_require**.

Pytest will run all files in the current directory and its subdirectories
of the form test_*.py or \*_test.py.
More generally, it follows `standard test discovery rules
<https://docs.pytest.org/en/latest/getting-started.html>`_

The command for running our tests has two forms.
Recommendation when working with a git clone of tvb-framework::

    cd [folder_where_tvb_framework_is]
    pytest tvb/test/framework [--profile=TEST_POSTGRES_PROFILE] [--junitxml=path]
    # default profile value is TEST_SQLITE_PROFILE

The second alternative form of running TVB tests, when installing TVB from Pypi, is::

    pip install -U tvb-framework
    pytest --pyargs tvb.tests.framework


Coverage
========

A coverage report can be generated with::

    pip install pytest-cov
    cd [folder_where_tvb_framework_is]
    py.test --cov=tvb tvb/tests/ --cov-branch --cov-report xml:[file_where_xml_will_be_generated]


Further Resources
=================

-  For issue tracking we are using Jira: http://req.thevirtualbrain.org
-  For API documentation and live demos, have a look here:
   http://docs.thevirtualbrain.org
-  A public mailing list for users of The Virtual Brain can be joined
   and followed using: tvb-users@googlegroups.com
-  Raw demo IPython Notebooks can be found under:
   https://github.com/the-virtual-brain/tvb-root/tree/master/tvb_documentation/demos


Acknowledgments
===============
This project has received funding from the European Union’s Horizon 2020 Framework Programme for Research and
Innovation under the Specific Grant Agreement Nos. 785907 (Human Brain Project SGA2), 945539 (Human Brain Project SGA3)
and VirtualBrainCloud 826421.