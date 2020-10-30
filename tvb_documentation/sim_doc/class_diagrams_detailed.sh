#!/bin/sh
set -o verbose

############################################################################
# Reverse engineer the classes of tvb.simulator, showing hidden attributes #
############################################################################


## Simulator ##
pyreverse --filter-mode=ALL --project=simulator_detailed ../simulator.py

dot -Tsvg classes_simulator_detailed.dot > img/classes_simulator_detailed.svg
dot -Tfig classes_simulator_detailed.dot > img/classes_simulator_detailed.fig


## Integrators ##
pyreverse --filter-mode=ALL --project=integrators_detailed ../integrators.py

dot -Tsvg classes_integrators_detailed.dot > img/classes_integrators_detailed.svg
dot -Tfig classes_integrators_detailed.dot > img/classes_integrators_detailed.fig


## Models ##
pyreverse --filter-mode=ALL --project=models_detailed ../models.py

dot -Tsvg classes_models_detailed.dot > img/classes_models_detailed.svg
dot -Tfig classes_models_detailed.dot > img/classes_models_detailed.fig


## Monitors ##
pyreverse --filter-mode=ALL --project=monitors_detailed ../monitors.py

dot -Tsvg classes_monitors_detailed.dot > img/classes_monitors_detailed.svg
dot -Tfig classes_monitors_detailed.dot > img/classes_monitors_detailed.fig


## Couplings ##
pyreverse --filter-mode=ALL --project=coupling_detailed ../coupling.py

dot -Tsvg classes_coupling_detailed.dot > img/classes_coupling_detailed.svg
dot -Tfig classes_coupling_detailed.dot > img/classes_coupling_detailed.fig


## Noise ##
pyreverse --filter-mode=ALL --project=noise_detailed ../noise.py

dot -Tsvg classes_noise_detailed.dot > img/classes_noise_detailed.svg
dot -Tfig classes_noise_detailed.dot > img/classes_noise_detailed.fig


## Clean up ##
rm *dot
