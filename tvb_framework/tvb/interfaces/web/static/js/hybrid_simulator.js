/**
 * TheVirtualBrain-Framework Package. This package holds all Data Management, and
 * Web-UI helpful to run brain-simulations. To use it, you also need to download
 * TheVirtualBrain-Scientific Package (for simulators). See content of the
 * documentation-folder for more details. See also http://www.thevirtualbrain.org
 *
 * (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
 *
 * This program is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License along with this
 * program.  If not, see <http://www.gnu.org/licenses/>.
 **/

/* globals doAjaxCall, renderWithMathjax, displayMessage, setupMenuEvents */

function _renderHybridSimulatorParameters(response) {
    const simParamElem = $("#div-hybrid-simulator-parameters");
    renderWithMathjax(simParamElem, response, true);
    if (typeof setupMenuEvents === "function") {
        setupMenuEvents();
    }
}

function resetToNewHybridSimulator() {
    doAjaxCall({
        type: "POST",
        url: "/burst/hybrid/reset_hybrid_simulator_configuration/",
        success: function (response) {
            _renderHybridSimulatorParameters(response);
            displayBurstTree(undefined);
            displayMessage("New hybrid simulator configuration loaded!");
        },
        error: function () {
            displayMessage("Hybrid simulator configuration could not be reset.", "errorMessage");
        }
    });
}

function loadHybridBurstHistory() {
    doAjaxCall({
        type: "POST",
        url: "/burst/hybrid/load_hybrid_history/",
        cache: false,
        success: function (response) {
            const historyElem = $("#section-view-history");
            renderWithMathjax(historyElem, response, true);
        },
        error: function () {
            displayMessage("Hybrid simulator history could not be loaded.", "errorMessage");
        }
    });
}

function hybridSubmit(currentForm) {
    event.preventDefault();
    const formData = $(currentForm).serialize();
    doAjaxCall({
        type: "POST",
        url: $(currentForm).attr("action"),
        data: formData,
        traditional: true,
        success: _renderHybridSimulatorParameters,
        error: function () {
            displayMessage("Hybrid simulator parameters could not be submitted.", "errorMessage");
        }
    });
}

function hybridPrevious(currentForm, previousAction) {
    doAjaxCall({
        type: "GET",
        url: previousAction,
        success: _renderHybridSimulatorParameters,
        error: function () {
            displayMessage("Hybrid simulator parameters could not be loaded.", "errorMessage");
        }
    });
}
