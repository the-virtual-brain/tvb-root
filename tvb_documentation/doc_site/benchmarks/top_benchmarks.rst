:orphan:

.. _top_benchmarks:


**********
Benchmarks
**********

In this section we provide some benchmarks about the execution times for some
simulation cases. This information is both to give you an idea of how long your
simulations may take and to show how variable can the running times be depending
on your hardware resources.

If you time your own simulations do not hesitate to send us your reports! As a
general note, if you are running simulations for benchmark purposes, check if
there are other processes that might be competing for resources. That might
result in biased results. Also do not forget to provide the system info:

  -  Operating System.
  -  Processor.
  -  Memory Available.
  -  TVB version.
  -  Web Browser (if applicable).
  -  Python version (if applicable).


Timeline
--------

Performance evolution from a TVB version to another.

.. toctree::
   :maxdepth: 2

   tvb_2.6.1_mac
   tvb_2.0_mac
   tvb_1.5.9_mac
   tvb_1.5.4_mac
   tvb_1.5_mac
   tvb_1.4.1_mac
   tvb_1.4_mac
   tvb_1.4
   tvb_1.3.1
   tvb_1.3
   tvb_1.2


.. figure:: /_static/benchmarks-neotraits.png
    :width: 1000px
    :figclass: demo-figure

    Only tvb-library Version 1.* vs 2.0 (* neo) shows a factor 2 improvement here.


.. figure:: /_static/benchmarks-evolution-mac.png
    :width: 1000px
    :figclass: demo-figure

    Recent versions - Mac station. Version 2.0 does not do very good with storage attached, but this was expected, we will work on optimizing this in future releases


.. figure:: /_static/benchmarks-evolution.png

    Older Versions - Linux station


.. figure:: /_static/benchmarks-evolution.png

    Recent Version - Mac station

