User Interface of |TVB|
=======================


Main Interface Description and Typical Workflow
-----------------------------------------------

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

.. raw:: pdf

   PageBreak

.. include:: UserGuide-UI_User.rst 

.. raw:: pdf

   PageBreak
   
.. include:: UserGuide-UI_Project.rst

.. raw:: pdf

   PageBreak

.. include:: UserGuide-UI_Simulator.rst

.. raw:: pdf

   PageBreak

.. include:: UserGuide-UI_Analyze.rst

.. raw:: pdf

   PageBreak

.. include:: UserGuide-UI_Stimulus.rst

.. raw:: pdf

   PageBreak

.. include:: UserGuide-UI_Connectivity.rst
