#!/bin/sh
set -o verbose

##############################################################################
# Reverse engineer the classes of tvb.simulator, show only public attributes #
##############################################################################

SIM_ROOT=../../tvb_library/tvb/simulator

## Simulator ##
pyreverse --project=simulator $SIM_ROOT/simulator.py

dot -Tsvg classes_simulator.dot > img/classes_simulator.svg
dot -Tfig classes_simulator.dot > img/classes_simulator.fig


## Integrators ##
pyreverse --project=integrators $SIM_ROOT/integrators.py

dot -Tsvg classes_integrators.dot > img/classes_integrators.svg
dot -Tfig classes_integrators.dot > img/classes_integrators.fig


## Models ##
pyreverse --project=models $SIM_ROOT/models

dot -Tsvg classes_models.dot > img/classes_models.svg
dot -Tfig classes_models.dot > img/classes_models.fig


## Monitors ##
pyreverse --project=monitors $SIM_ROOT/monitors.py

dot -Tsvg classes_monitors.dot > img/classes_monitors.svg
dot -Tfig classes_monitors.dot > img/classes_monitors.fig


## Couplings ##
pyreverse --project=coupling $SIM_ROOT/coupling.py

dot -Tsvg classes_coupling.dot > img/classes_coupling.svg
dot -Tfig classes_coupling.dot > img/classes_coupling.fig


## Noise ##
pyreverse --project=noise $SIM_ROOT/noise.py

dot -Tsvg classes_noise.dot > img/classes_noise.svg
dot -Tfig classes_noise.dot > img/classes_noise.fig

## Clean up ##
rm *dot
rm img/*.fig
