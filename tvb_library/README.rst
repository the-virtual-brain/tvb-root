THE VIRTUAL BRAIN SCIENTIFIC LIBRARY
====================================

The Virtual Brain Project (TVB Project) has the purpose of offering some
modern tools to the Neurosciences community, for computing, simulating
and analyzing functional and structural data of human brains.

"TVB Scientific Library" is the most important scientific contribution
of TVB Project, but only a part of our code. In order to use this TVB
Python Library (modify/run/test), you are advised to follow the steps
described here:
http://docs.thevirtualbrain.com/manuals/ContributorsManual/ContributorsManual.html#contributors-manual

The interaction with TVB source code is recommended for advanced users
only, for which the concepts proposed by TVB are known and understood,
and which also have some programming knowledge.

"TVB Scientific Library" comes in two major forms for code-contributors:
as a light-weight Python library, or plugged into TVB framework
(including Web interface and Persistence of data layers). Depending on
your needs and resources, you could use any of the two available options
for contributing and testing new code.

"TVB Scientific Library" is a light-weight, stand-alone Python library
that contains all the needed packages in order to run simulations and
analysis on data without the need for the entire TVB Framework. This
implies that no storage will be provided so data from each session will
be lost on close. You need to either persist it yourself in some manner
or use the full TVBFramework where HDF5 / database storage is provided
as default.

For more details, check: http://www.thevirtualbrain.org/

"TVB Scientific Library" contains the following packages: basic,
datatypes, simulator and analyzers. The dependencies between these
packages can be seen in tvb-package-diagram.jpg . Following is a short
description of each of these packages:

-  tvb.basic

This package is the base of TVB and holds sub-packages that are used by
most of the other packages like logging, global settings and the TVB
traits package. You should rarely (if at all) need to change code in
this package, and should know exactly what you are doing before
attempting to change anything from here.

-  tvb.datatypes

The simulator and analyzers packages (as well as uploaders and
visualizers in TVB Framework) will need to have a common "language" in
order to work with the same data. In TVB architecture, that "common
language" is represented by Data Types. TVB Data Types declarations are
located in this package.

-  tvb.simulator

The Simulation Component is the most important component in The Virtual
Brain solution, as it is the component responsible for all the
scientific computation related to brain models and data.

You can find various demos of using the simulator here:
http://docs.thevirtualbrain.org/demos/Demos.html .

-  tvb.analyzers

Holds modules that can run various analysis of data resulted from the
simulator. TVB is not strong in
doing data analysis, we barely have a minimum set of analyzers for
immediate needs.

NOTE:

To contribute to this repo, you are advised to first fork it under
GitHub (through the web ui), and next clone that fork on your machine,
and use it as you use any other git repository.

Testing
=======

For testing the package, the `Pytest  <https://docs.pytest.org/>`_
framework is used. Pytest can be installed using pip.

Pytest will run all files in the current directory and its subdirectories
of the form test_*.py or \*_test.py.
More generally, it follows `standard test discovery rules
<https://docs.pytest.org/en/latest/getting-started.html>`_

The command for running our tests has two forms.
Recommendation when working with a git clone of tvb-library::

    cd [folder_where_tvb_library_is]
    pytest tvb/test [--junitxml=path]

The second alternative form of running TVB tests, when installing TVB from Pypi, is::

    pip install -U tvb-library
    pytest --pyargs tvb.tests.library


Coverage
--------

A coverage report can be generated with::

    pip install pytest-cov
    cd [folder_where_tvb_library_is]
    py.test --cov-config .coveragerc --cov=tvb tvb/tests/ --cov-branch --cov-report xml:[file_where_xml_will_be_generated]


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
