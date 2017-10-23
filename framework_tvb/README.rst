TVB FRAMEWORK
=============

The Virtual Brain framework is a complete framework including:

-  a plugable workflow manager;
-  a data persistence layer (with a relational DB and File Storage);
-  an HTML5 based user interface;
-  visualizers for neuro-science related entities.

The easiest way to make use of the code from this Git repo is to obtain
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

To use TVB sources, clone from GitHub (https://github.com/the-virtual-brain/tvb-framework), or get from Pypi::

    pip install tvb-framework
    python -m tvb.interfaces.web.run WEB_PROFILE tvb.config


Your port **8080** should be free, as a CherryPy service will try to run there.
Your default browser should automatically open http://localhost:8080/ which is the way to
interact with TVB Web Interface.

When using from sources (pypi or Github, no TVB_Distribution), if you want BCT adapters enabled, you should
manually download BCT https://sites.google.com/site/bctnet/
and set env variable **BCT_PATH** towards the directory when you unzip BCT, plus also have Octave or
Matlab installed with command line API enabled.


Further Resources
=================

-  For issue tracking we are using Jira: http://req.thevirtualbrain.org
-  For API documentation and live demos, have a look here:
   http://docs.thevirtualbrain.org
-  A public mailing list for users of The Virtual Brain can be joined
   and followed using: tvb-users@googlegroups.com
-  Raw demo IPython Notebooks can be found under:
   https://github.com/the-virtual-brain/tvb-documentation/tree/master/demos
