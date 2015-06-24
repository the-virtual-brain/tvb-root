Web User Interface of |TVB|
============================

The workflow of TVB is divided in **User**, **Project**, **Simulator**,
**Analysis**, **Stimulus** and **Conectivity**. In **User**, the user can
manage their account and |TVB| settings. In **Project**, the individual projects
are defined with all their data and infrastructure. In **Simulator** the
large-scale simulation is defined and different options to view structural and
functional data are offered in 2D, 3D, as well other representations to
visualize results. Having multiple panels allows having a quick overview of the
current |TVB| model parameters, simulations and results. In **Analysis** certain
options for data and connectivity analysis are offered. In **Stimulus**,
patterns of stimuli can be generated. And finally in **Connectivity**, the user
can edit the connectivity matrices assisted by interactive visualization tools.


Main Web Interface Description and Typical Workflow
----------------------------------------------------

On the right, the `login` button has changed to a `logout` button with obvious
functionality.

The main menu of |TVB| interface lays at the bottom of the page and is composed
of six basic options:

User:
    where user's details are managed.

Project:
    where projects are defined and administered.

Simulator:
    where simulations are launched, combined with the analyzers and visualizers. 
    It allows to have quick overview of the ongoing Project, which explains why 
    we consider the Simulator to be the core of |TVB|.

Stimulus:
    where spatiotemporal stimuli can be generated.

Analyze:
    where experimental and simulated data can be analyzed.

Connectivity:
    where connectivity visualization and editing facilities of |TVB| are stored.

These options sum up the typical workflow within |TVB| framework which proceeds
through these steps:

1. a project is defined and/or selected and user data, (e. g. a connectivity matrix), are uploaded into this project;

2. new data is obtained by simulating large scale brain dynamics with some set of parameters;

3. results are analyzed and visualized;

A history of launched simulations is kept to have the traceability of any
modifications that took place in the simulation chain.
