.. include:: /manuals/templates/pdf_constants.rst

.. _top_gui_guide:

Web User Interface of |TVB|
============================

The workflow of |TVB| is divided in six major activities.
The main menu of the web interface lays at the bottom of the page. It links to these six activities.

1. In **User**, the user can manage their account and |TVB| settings.

2. In **Project**, the individual projects are managed with all their data and infrastructure.

3. In **Simulator** large-scale simulations are defined and launched. Analysers and visualizers
   associated with a simulation are defined there.
   Their results, structural and functional data, are shown in panels.
   Having multiple panels allows having a quick overview of the
   current |TVB| model parameters, simulations and results.
   We consider the Simulator to be the core of |TVB|.

4. In **Analysis** experimental and simulated data can be analyzed.

5. In **Stimulus**, patterns of spatiotemporal stimuli can be generated.

6. And finally in **Connectivity**, the user can edit the connectivity matrices assisted by interactive visualization tools.

..
    Sphinx quirk:
    Section title commented because sphinx assumes the toctree to be a child of this.
    The resulting nested structure is akward for navigation and pdf index.
    If we find a way around this behaviour then we may reintroduce a section title
    Main Web Interface Description and Typical Workflow
    ---------------------------------------------------

The typical workflow within |TVB| GUI proceeds through these steps:

1. a project is defined and/or selected and user data, (e. g. a connectivity matrix), are uploaded into this project

2. new data is obtained by simulating large scale brain dynamics with some set of parameters

3. results are analyzed and visualized

A history of launched simulations is kept to have the traceability of any
modifications that took place in the simulation chain.

.. toctree::
    :maxdepth: 2

    /manuals/UserGuide/UserGuide-UI_User.rst
    /manuals/UserGuide/UserGuide-UI_Project.rst
    /manuals/UserGuide/UserGuide-UI_Simulator.rst
    /manuals/UserGuide/UserGuide-UI_Analyze.rst
    /manuals/UserGuide/UserGuide-UI_Stimulus.rst
    /manuals/UserGuide/UserGuide-UI_Connectivity.rst
