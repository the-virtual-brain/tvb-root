
# The Virtual Brain

The Virtual Brain Project (TVB Project) has the purpose of offering some
modern tools to the Neurosciences community, for computing, simulating
and analyzing functional and structural data of human brains.


This repository holds the sources for TVB codebase.

To install these in your env, using our pre-packed Pypi releases is recommended, in a separated env.
Our main deliverable are shared on Pypi under the following names:

    pip install tvb-library
    pip install tvb-framework
   

More details on our
documentation site: <http://docs.thevirtualbrain.org>.

   
### TVB Scientific library

"TVB Scientific Library" is the most important scientific contribution
of TVB Project, but only a part of our code. 

"TVB Scientific Library" is a light-weight, stand-alone Python library
that contains all the needed packages in order to run simulations and
analysis on data without the need for the entire TVB Framework. This
implies that no storage will be provided so data from each session will
be lost on close. You need to either persist it yourself in some manner
or use the full TVBFramework where HDF5 / database storage is provided
as default.

   
### TVB Framework

The Virtual Brain framework is a complete framework, wrapped over tvb-library, and offering extra:

-  a plugable workflow manager;
-  a data persistence layer (with a relational DB and File Storage);
-  an HTML5 based user interface;
-  visualizers for neuro-science related entities.
 
You can launch the web interface with the command:

    python -m tvb.interfaces.web.run WEB_PROFILE tvb.config
    
Your port **8080** should be free, as a CherryPy service will try to run there.
Your default browser should automatically open http://localhost:8080/ which is the way to
interact with TVB Web Interface.

When using from sources (pypi or Github, not TVB_Distribution), if you want BCT adapters enabled, you should
manually download BCT https://sites.google.com/site/bctnet/
and set env variable **BCT_PATH** towards the directory where you unzip BCT, plus also have Octave or
Matlab installed with command line API enabled.
    
### Testing

For testing the package, the Pytest framework is used. 

Pytest will run all files in the current directory and its subdirectories
of the form test_*.py or *_test.py.

The command for running our tests has two forms.
Recommendation when working with a git clone of this tvb github repo:

    cd [folder_where_tvb_framework_is]
    pytest tvb/test/framework [--profile=TEST_POSTGRES_PROFILE] [--junitxml=path]
    # default profile value is TEST_SQLITE_PROFILE
    
    cd [folder_where_tvb_library_is]
    pytest tvb/test/library [--junitxml=path]

The second alternative form of running TVB tests, when installing TVB from Pypi, is::

    pip install -U tvb-framework
    pytest --pyargs tvb.tests.framework
    
    pip install -U tvb-library
    pytest --pyargs tvb.tests.library


### Coverage

A coverage report can be generated with::

    pip install pytest-cov
    cd [folder_where_tvb_framework_is]
    py.test --cov=tvb tvb/tests/ --cov-branch --cov-report xml:[file_where_xml_will_be_generated]

    cd [folder_where_tvb_library_is]
    py.test --cov-config .coveragerc --cov=tvb tvb/tests/ --cov-branch --cov-report xml:[file_where_xml_will_be_generated]


# Relevant TVB Resources

- For issue tracking we are using Jira: http://req.thevirtualbrain.org
- For API documentation and live demos, have a look here: http://docs.thevirtualbrain.org
- A public mailing list for users of The Virtual Brain can be joined and followed using: tvb-users@googlegroups.com
- Raw demo IPython Notebooks can be found under: https://github.com/the-virtual-brain/tvb-documentation/tree/master/demos
