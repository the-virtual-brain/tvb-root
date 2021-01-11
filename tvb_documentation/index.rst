.. tvb_documentation documentation master file, created by
   sphinx-quickstart on Mon Nov 25 22:11:09 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. IMPORTANT: Sphinx expects a non dummy toc tree in the index.rst
   This index page is not only the main documentation page.
   It is mostly the top level node of the document hierarchy
   Without this toc sphinx is lost, does not render navigations etc.
   The toctree is hidden, not rendered, it just defines structure

.. toctree::
    :hidden:

    /manuals/UserGuide/UserGuide-Installation
    /manuals/UserGuide/UserGuide-Config
    /tutorials/Tutorials
    /demos/Demos
    /manuals/UserGuide/UserGuide-UI
    /manuals/UserGuide/UserGuide-Shell
    /doc_site/top_developers


.. include:: /manuals/UserGuide/UserGuide-Overview.rst

Helpful external resources
--------------------------

The **TVB main website** on `www.thevirtualbrain.org <https://www.thevirtualbrain.org/>`_ contains a lot of helpful resources:


* `Download <https://www.thevirtualbrain.org/tvb/zwei/brainsimulator-software>`_ **TVB software packages** for macOS, Windows and Linux.
* `Learn <https://www.thevirtualbrain.org/tvb/zwei/neuroscience-simulation>`_ about the **scientific background and clinical applications** of TVB.
* `Study <https://www.thevirtualbrain.org/tvb/zwei/newswire-educase>`_ dozens of **TVB EduCases with video lectures**, explaining the software step by step.
* `Read <https://www.thevirtualbrain.org/tvb/zwei/newswire-blog>`_ the **TVB blog** about the latest news and achievements.
* `Follow <https://www.thevirtualbrain.org/tvb/zwei/newswire-event>`_ international **TVB events** to meet other developers and scientists working with TVB.


If you're familiar with Docker and/or Python environments, you can try early releases of the TVB software:


* We publish **Docker containers** on the `TVB DockerHub <https://hub.docker.com/u/thevirtualbrain>`_.
* We publish **Python packages** on the `TVB PyPI profile <https://pypi.org/user/tvb/>`_.


These versions of TVB are updated more frequently and contain all the latest new features and bugfixes. You can follow all the latest changes on our |our_github| page.

On CERN's data sharing platform *Zenodo*, we offer `various demonstration datasets <https://zenodo.org/record/4263723>`_, readily packaged for usage with The Virtual Brain software.

If you have specific questions, also about how to use TVB for your current research activity, you can use our `public discussion forum <https://groups.google.com/g/tvb-users>`_, which doubles as a *mailing list* if you prefer this channel.


In this forum, you can meet and discuss with other TVB users, as well as experts from our own support team. It's a perfect place to ask things like *"Does anyone else see that both the EEG and the BOLD have similar shape in terms of the 1/f type of drop off towards the higher frequency range?"*.



We are grateful to
------------------

 - our contributors (check their names on |our_github|)
 - our sponsors (check their names on the |our_sponsors_page|)
 - all |third_party| tools that we used (licenses are also included in TVB_Distribution)
 - JetBrains for |pycharm_ide|
 - |jenkins| team for their continuous integration tool
 - Atlassian company for |jira| software
 - and to you for reading these :-)


    .. image:: _static/logo_python.svg
        :height: 64px
    .. image:: _static/logo_jira.png
        :height: 64px
    .. image:: _static/logo_pycharm.png
        :height: 64px
    .. image:: _static/logo_github.png
        :height: 64px
    .. image:: _static/logo_jenkins.png
        :height: 64px


.. |our_sponsors_page| raw:: html

    <a href="http://www.thevirtualbrain.org/tvb/zwei/teamwork-sponsors" target="_blank">our sponsors page</a>


.. |our_github| raw:: html

   <a href="https://github.com/the-virtual-brain" target="_blank">GitHub</a>


.. |third_party| raw:: html

   <a href="http://www.thevirtualbrain.org/tvb/zwei/brainsimulator-requirements" target="_blank">3rd party</a>


.. |pycharm_ide| raw:: html

    <a href="https://www.jetbrains.com/pycharm/" target="_blank">PyCharm IDE</a>


.. |jenkins| raw:: html

    <a href="https://www.jenkins.io/" target="_blank">Jenkins</a>


.. |jira| raw:: html

    <a href="https://www.atlassian.com/software/jira" target="_blank">Jira</a>
