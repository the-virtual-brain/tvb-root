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
the `tvb-library <https://github.com/the-virtual-brain/tvb-library>`__
repo.

Usage
-----

To use TVB code, clone from GitHub (https://github.com/the-virtual-brain/tvb-framework), or get from Pypi::

    pip install tvb-framework
    python -m tvb.interfaces.web.run WEB_PROFILE tvb.config


Your port **8080** should be free, as a CherryPy service will try to run there.
Your default browser should automatically open http://localhost:8080/ which is the way to
interact with TVB Web Interface.

When using from sources (pypi or Github, not TVB_Distribution), if you want BCT adapters enabled, you should
manually download BCT https://sites.google.com/site/bctnet/
and set env variable **BCT_PATH** towards the directory where you unzip BCT, plus also have Octave or
Matlab installed with command line API enabled.


Testing
=======

For testing the package, the `Pytest  <https://docs.pytest.org/>`_
framework is used. Pytest can be installed using pip command::

  pip install -U pytest

Pytest will run all files in the current directory and its subdirectories
of the form test_*.py or \*_test.py.
More generally, it follows `standard test discovery rules
<https://docs.pytest.org/en/latest/getting-started.html>`_

The command for running our tests has two forms.
Recommandation when working with a git clone of tvb-framework::

  cd [folder_where_tvb_framework_is]
  pytest tvb/test [--profile=TEST_POSTGRES_PROFILE] [--junitxml=path]

When installing TVB from Pypi, the recommandation is to run our tests with::

pip install -U tvb-framework
pip install -U tvb-data
pytest --pyargs tvb.tests.framework
pytest --pyargs tvb.tests.framework


For changing the testing profile at runtime use the command::

  pytest --profile=TEST_POSTGRES_PROFILE
  default value is TEST_SQLITE_PROFILE

Coverage
========



A coverage report can be generated with::

pip install pytest-cov
py.test --cov=[folder_where_tvb_framework_is] tvb/tests/ --cov-branch --cov-report xml:[file_where_xml_will_be_generated]

Further Resources
=================

-  For issue tracking we are using Jira: http://req.thevirtualbrain.org
-  For API documentation and live demos, have a look here:
   http://docs.thevirtualbrain.org
-  A public mailing list for users of The Virtual Brain can be joined
   and followed using: tvb-users@googlegroups.com
-  Raw demo IPython Notebooks can be found under:
   https://github.com/the-virtual-brain/tvb-documentation/tree/master/demos
