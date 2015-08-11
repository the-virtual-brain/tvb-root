#!/bin/sh
set -o verbose

##############################################################################
# Reverse engineer the classes of tvb.simulator, show only public attributes #
##############################################################################


## Simulator ##
pyreverse --project=simulator ../simulator.py

dot -Tsvg classes_simulator.dot > img/classes_simulator.svg
dot -Tfig classes_simulator.dot > img/classes_simulator.fig


## Integrators ##
pyreverse --project=integrators ../integrators.py

dot -Tsvg classes_integrators.dot > img/classes_integrators.svg
dot -Tfig classes_integrators.dot > img/classes_integrators.fig


## Models ##
pyreverse --project=models ../models.py

dot -Tsvg classes_models.dot > img/classes_models.svg
dot -Tfig classes_models.dot > img/classes_models.fig


## Monitors ##
pyreverse --project=monitors ../monitors.py

dot -Tsvg classes_monitors.dot > img/classes_monitors.svg
dot -Tfig classes_monitors.dot > img/classes_monitors.fig


## Couplings ##
pyreverse --project=coupling ../coupling.py

dot -Tsvg classes_coupling.dot > img/classes_coupling.svg
dot -Tfig classes_coupling.dot > img/classes_coupling.fig


## Noise ##
pyreverse --project=noise ../noise.py

dot -Tsvg classes_noise.dot > img/classes_noise.svg
dot -Tfig classes_noise.dot > img/classes_noise.fig


## Phase-Plane Interactive ##
pyreverse --project=phase_plane_interactive ../plot/phase_plane_interactive.py

dot -Tsvg classes_phase_plane_interactive.dot > img/classes_phase_plane_interactive.svg
dot -Tfig classes_phase_plane_interactive.dot > img/classes_phase_plane_interactive.fig


## Power-Spectra Interactive ##
pyreverse --project=power_spectra_interactive ../plot/power_spectra_interactive.py

dot -Tsvg classes_power_spectra_interactive.dot > img/classes_power_spectra_interactive.svg
dot -Tfig classes_power_spectra_interactive.dot > img/classes_power_spectra_interactive.fig


## Time-Series Interactive ##
pyreverse --project=timeseries_interactive ../plot/timeseries_interactive.py

dot -Tsvg classes_timeseries_interactive.dot > img/classes_timeseries_interactive.svg
dot -Tfig classes_timeseries_interactive.dot > img/classes_timeseries_interactive.fig


## Clean up ##
rm *dot
#rm img/*.fig
