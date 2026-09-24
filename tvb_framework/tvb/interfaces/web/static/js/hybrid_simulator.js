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

/* globals doAjaxCall, renderWithMathjax, displayMessage, setupMenuEvents, updateTree */

/**
 * The Hybrid Simulator wizard, following the classic Simulator Cockpit behaviour: pressing Next keeps
 * the step you just filled in on screen as a disabled, read-only form and appends the next one under
 * it, so the whole configuration stays visible while it is built up.
 *
 * The third column follows the step being configured. Each fragment names the configuration it wants
 * there through data-hybrid-context-url; a step naming none leaves the column empty and the Results
 * view visible. This is what replaced the separate, full width Subnetwork configuration page.
 */

const HYBRID_FORMS_DIV = "hybrid-simulator-forms";
const HYBRID_CONTEXT_DIV = "hybrid-context-column";
const HYBRID_RESULTS_DIV = "hybrid-results-view";
const HYBRID_SUBNETWORKS_STEP_URL = "/burst/hybrid/set_subnetworks";
const HYBRID_DYNAMICS_STEP_URL = "/burst/hybrid/set_subnetwork_dynamics";
// The cockpit steps, in wizard order, used to rebuild the stack when a step is no longer on screen.
//
// It deliberately stops at the Subnetwork dynamics. Rebuilding walks this list and renders every step
// on it, which for the Projections would mean regenerating the whole NetworkSet on any step back, and
// for the Monitors would mean rendering them out of order, since the Monitor parameter steps depend on
// what was selected and have no fixed urls at all. A step that is not on the list falls back to
// rebuilding up to the dynamics, which is where a user can reach all of them again.
const HYBRID_WIZARD_STEPS = ["/burst/hybrid/set_connectivity", HYBRID_SUBNETWORKS_STEP_URL,
    HYBRID_DYNAMICS_STEP_URL];

function _hybridFormsDiv() {
    return document.getElementById(HYBRID_FORMS_DIV);
}

/** Every step of the wizard, in the order they are stacked. */
function _hybridForms() {
    return Array.prototype.slice.call(_hybridFormsDiv().querySelectorAll("form"));
}

/** The step currently being configured: the last one of the wizard stack. */
function _activeHybridForm() {
    const forms = _hybridForms();
    return forms.length === 0 ? null : forms[forms.length - 1];
}

function _asFragment(response) {
    // createContextualFragment keeps inline scripts executable, which the Subnetwork board needs
    return document.createRange().createContextualFragment(response);
}

function _afterHybridRender() {
    if (typeof setupMenuEvents === "function") {
        setupMenuEvents();
    }
    _syncHybridContextColumn();
    $("button.btn-next").last().focus();
}

// ---------------------------------------------------------------- contextual configuration column

function _setHybridContextTitle(title) {
    const action = document.getElementById("hybrid-context-action");
    const subject = document.getElementById("title-visualizers");
    if (action === null || subject === null) {
        return;
    }
    action.textContent = title === "" ? "Visualize" : "Configure";
    subject.textContent = title === "" ? "Hybrid simulation" : title;
}

function _clearHybridContextColumn() {
    const contextDiv = document.getElementById(HYBRID_CONTEXT_DIV);
    if (contextDiv === null) {
        return;
    }
    contextDiv.innerHTML = "";
    contextDiv.style.display = "none";
    $("#" + HYBRID_RESULTS_DIV).show();
    _setHybridContextTitle("");
}

/**
 * Show in the third column whatever the step currently being configured asked for, or empty that
 * column and hand it back to the Results view when the step configures nothing there.
 */
function _syncHybridContextColumn() {
    const contextDiv = document.getElementById(HYBRID_CONTEXT_DIV);
    if (contextDiv === null) {
        return;
    }
    const form = _activeHybridForm();
    const contextUrl = form === null ? "" : (form.dataset.hybridContextUrl || "");

    if (contextUrl === "") {
        _clearHybridContextColumn();
        return;
    }

    doAjaxCall({
        type: "GET",
        url: contextUrl,
        success: function (response) {
            $("#" + HYBRID_RESULTS_DIV).hide();
            contextDiv.style.display = "";
            renderWithMathjax($(contextDiv), _asFragment(response), true);
            _setHybridContextTitle(form.dataset.hybridContextTitle || "");
        },
        error: function () {
            _clearHybridContextColumn();
            displayMessage("The configuration of this step could not be loaded.", "errorMessage");
        }
    });
}

/**
 * Show the Set up region Model configuration in the third column. Unlike the configurations a step
 * declares through data-hybrid-context-url, this one is opened on demand by its button, so nothing
 * clears it until the user leaves the step - moving on re-syncs the column, which is what closes it.
 */
function hybridConfigureRegionModel() {
    const contextDiv = document.getElementById(HYBRID_CONTEXT_DIV);
    if (contextDiv === null) {
        return;
    }

    doAjaxCall({
        type: "GET",
        url: "/burst/hybrid/configure_region_model",
        success: function (response) {
            $("#" + HYBRID_RESULTS_DIV).hide();
            contextDiv.style.display = "";
            renderWithMathjax($(contextDiv), _asFragment(response), true);
            _setHybridContextTitle("Region Model");
        },
        error: function () {
            displayMessage("The region Model configuration could not be loaded.", "errorMessage");
        }
    });
}

/**
 * Replace one wizard step with a freshly rendered version of itself, leaving the third column as it is.
 *
 * Deliberately not _afterHybridRender: that re-syncs the column, and the Set up region Model panel is
 * open in it while this is called, so it would close the panel the user is still working in.
 */
function hybridReplaceStep(stepUrl, response) {
    const form = document.getElementById(stepUrl);
    if (form === null) {
        return false;
    }
    const fragment = _asFragment(response);
    const newForm = fragment.querySelector("form");
    if (newForm === null || newForm.id !== stepUrl) {
        return false;
    }
    form.replaceWith(fragment);
    if (typeof setupMenuEvents === "function") {
        setupMenuEvents();
    }
    return true;
}

/** The Results tree of this page. bursts.js, which owns the cockpit one, is not loaded here. */
// The Hybrid simulation whose results the third column is showing. Set when one is launched from this
// page; until then the tree has no burst to filter on.
let HYBRID_RESULTS_BURST_ID = null;

function displayHybridResultsTree(burstId) {
    if (burstId !== undefined && burstId !== null) {
        HYBRID_RESULTS_BURST_ID = burstId;
    }
    const filterValue = HYBRID_RESULTS_BURST_ID === null ? "0" : String(HYBRID_RESULTS_BURST_ID);
    updateTree("#treeOverlay", null, JSON.stringify({'type': 'from_burst', 'value': filterValue}));
    $("#div-burst-tree").show();
}

/**
 * Launch the configured Hybrid simulation. The answer carries the new simulation's id, which is what
 * the history and the results tree are then pointed at - the simulation itself runs on the server.
 */
function hybridLaunchSimulation(currentForm) {
    const launchButton = currentForm.elements.namedItem("launch_simulation");
    if (launchButton !== null) {
        launchButton.disabled = true;
    }

    displayMessage("Hybrid simulation submitted. Please wait for the preprocessing steps...", "warningMessage");

    doAjaxCall({
        type: "POST",
        url: "/burst/hybrid/launch_simulation/",
        data: $(currentForm).serialize(),
        traditional: true,
        success: function (response) {
            const result = $.parseJSON(response);
            if ('error' in result) {
                displayMessage(result.error, "errorMessage");
                if (launchButton !== null) {
                    launchButton.disabled = false;
                }
                return;
            }
            loadHybridBurstHistory();
            if ('id' in result) {
                displayHybridResultsTree(result.id);
            }
            displayMessage("Hybrid simulation launched.");
        },
        error: function () {
            displayMessage("The Hybrid simulation could not be launched.", "errorMessage");
            if (launchButton !== null) {
                launchButton.disabled = false;
            }
        }
    });
}

// ---------------------------------------------------------------- wizard stack

/**
 * Turn a form into the read-only record of a step that is already done: fields greyed out, buttons
 * hidden. This is what the classic cockpit does when you move on.
 */
function _staysEnabled(element) {
    // the Subnetwork selector keeps working on a finished step: switching Subnetwork is what rebuilds
    // the steps configuring it, so locking it away would strand the user on the first one picked
    return element.closest("[data-hybrid-keep-enabled]") !== null;
}

function _lockHybridForm(form) {
    form.querySelectorAll("button").forEach(function (button) {
        if (!_staysEnabled(button)) {
            button.style.visibility = "hidden";
        }
    });
    form.querySelectorAll("fieldset").forEach(function (fieldset) {
        fieldset.disabled = true;
    });
}

function _unlockHybridForm(form) {
    form.querySelectorAll("button").forEach(function (button) {
        button.style.visibility = "visible";
    });
    form.querySelectorAll("fieldset").forEach(function (fieldset) {
        fieldset.disabled = false;
    });
}

/** Append one more step under the ones already on screen. */
function _appendHybridFragment(fragment) {
    renderWithMathjax($(_hybridFormsDiv()), fragment);
    _afterHybridRender();
}

/** Replace everything on screen with a single fragment. */
function _replaceHybridFragments(response) {
    renderWithMathjax($(_hybridFormsDiv()), _asFragment(response), true);
    _afterHybridRender();
}

/**
 * Rebuild the wizard stack from scratch, loading the given steps in order and leaving every step but
 * the last one read-only.
 */
function _renderHybridStack(stepUrls) {
    const container = _hybridFormsDiv();
    container.innerHTML = "";

    let index = 0;

    function loadNext() {
        if (index >= stepUrls.length) {
            _afterHybridRender();
            return;
        }
        const isLastStep = index === stepUrls.length - 1;
        doAjaxCall({
            type: "GET",
            url: stepUrls[index],
            success: function (response) {
                const fragment = _asFragment(response);
                const form = fragment.querySelector("form");
                renderWithMathjax($(container), fragment);
                if (!isLastStep && form !== null) {
                    _lockHybridForm(form);
                }
                index += 1;
                loadNext();
            },
            error: function () {
                displayMessage("Hybrid simulator parameters could not be loaded.", "errorMessage");
            }
        });
    }

    loadNext();
}

function resetToNewHybridSimulator() {
    doAjaxCall({
        type: "POST",
        url: "/burst/hybrid/reset_hybrid_simulator_configuration/",
        success: function (response) {
            _replaceHybridFragments(response);
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

/**
 * Submit the current step and move to the next one, keeping the current step on screen as read-only.
 * When the server answers with the same step again the configuration was rejected, so that step is
 * replaced in place instead of being stacked on top of itself.
 */
function hybridSubmit(currentForm) {
    hybridSubmitTo(currentForm, $(currentForm).attr("action"));
}

/**
 * Submit the current step to the given url and move to the next one. A step whose own action url does
 * something other than moving on - the closing step of the Subnetwork configuration stores the dynamics
 * there - needs this to say where forward is.
 */
function hybridSubmitTo(currentForm, url) {
    // the wizard buttons are type="button" so nothing would submit anyway, but keep the guard for
    // any caller that does arrive through a real event. window.event only exists during dispatch.
    if (typeof event !== "undefined" && event !== null) {
        event.preventDefault();
    }
    // A disabled fieldset is not serialized, and a read-only step is exactly that, so it is enabled for
    // the length of the call. The classic wizzard_submit does the same.
    const disabledFieldsets = Array.prototype.filter.call(
        currentForm.querySelectorAll("fieldset"), function (fieldset) {
            return fieldset.disabled;
        });
    disabledFieldsets.forEach(function (fieldset) {
        fieldset.disabled = false;
    });
    const formData = $(currentForm).serialize();
    disabledFieldsets.forEach(function (fieldset) {
        fieldset.disabled = true;
    });

    doAjaxCall({
        type: "POST",
        url: url,
        data: formData,
        traditional: true,
        success: function (response) {
            const fragment = _asFragment(response);
            const newForm = fragment.querySelector("form");

            if (newForm !== null && newForm.id === currentForm.id) {
                currentForm.replaceWith(fragment);
                _afterHybridRender();
                return;
            }

            _lockHybridForm(currentForm);
            _appendHybridFragment(fragment);
        },
        error: function () {
            displayMessage("Hybrid simulator parameters could not be submitted.", "errorMessage");
        }
    });
}

/**
 * Step back: drop the current step and hand control back to the one above it, which is already on
 * screen. A form's id is its action url, which is how the previous step is found.
 */
function hybridPreviousStep(currentForm, previousUrl) {
    const previousForm = document.getElementById(previousUrl);

    if (previousForm === null) {
        // the step above is not on screen, so rebuild the stack up to and including it
        const upTo = HYBRID_WIZARD_STEPS.indexOf(previousUrl);
        _renderHybridStack(HYBRID_WIZARD_STEPS.slice(0, upTo === -1 ? undefined : upTo + 1));
        return;
    }

    currentForm.remove();
    _unlockHybridForm(previousForm);
    _afterHybridRender();
}

/**
 * Store the Subnetwork grouping edited in the third column. Only then does the wizard step listing the
 * Subnetworks change, which is why the answer replaces that step. Rendering it also reloads the board,
 * so the two always agree about what is configured.
 */
function hybridSaveSubnetworks() {
    doAjaxCall({
        type: "POST",
        url: "/burst/hybrid/save_subnetworks/",
        success: function (response) {
            const currentForm = document.getElementById(HYBRID_SUBNETWORKS_STEP_URL);
            const fragment = _asFragment(response);
            const newForm = fragment.querySelector("form");

            if (currentForm === null || newForm === null || newForm.id !== HYBRID_SUBNETWORKS_STEP_URL) {
                // the configuration is no longer where we left it, e.g. the Connectivity went missing
                _replaceHybridFragments(response);
                return;
            }

            currentForm.replaceWith(fragment);
            _afterHybridRender();
            displayMessage("Subnetwork configuration saved.");
        },
        error: function () {
            displayMessage("The Subnetwork configuration could not be saved.", "errorMessage");
        }
    });
}

// ---------------------------------------------------------------- Subnetwork dynamics

/** Drop every step stacked under the given one. */
function _dropHybridStepsAfter(form) {
    const forms = _hybridForms();
    const from = forms.indexOf(form);
    if (from === -1) {
        return;
    }
    forms.slice(from + 1).forEach(function (later) {
        later.remove();
    });
}

/**
 * Configure another Subnetwork. The steps that were configuring the previous one are dropped and this
 * one's whole configuration takes their place - every step, not just the first, so a Subnetwork that is
 * already set up is shown rather than stepped through again. Every field is on screen, and any of them
 * can be reached with Previous.
 *
 * The server sends the steps in one answer, already read only except the last, so nothing here has to
 * sequence requests or decide what to lock.
 *
 * What was edited for the Subnetwork being left is kept: the server holds the draft of every one.
 */
function hybridSelectSubnetwork(subnetworkId) {
    const dynamicsForm = document.getElementById(HYBRID_DYNAMICS_STEP_URL);
    if (dynamicsForm === null) {
        return;
    }

    doAjaxCall({
        type: "POST",
        url: "/burst/hybrid/select_subnetwork/",
        data: {subnetwork_id: subnetworkId},
        success: function (response) {
            _dropHybridStepsAfter(dynamicsForm);
            _appendHybridFragment(_asFragment(response));
            // the selector shows which Subnetwork is being configured, so it has to be redrawn too
            _refreshHybridDynamicsStep();
        },
        error: function () {
            displayMessage("This Subnetwork could not be opened.", "errorMessage");
        }
    });
}

/**
 * Reload the Subnetwork dynamics step in place, keeping the steps stacked under it. That step carries
 * the selector and the summary of what every Subnetwork is configured with, both of which go stale as
 * soon as something below it changes.
 */
function _refreshHybridDynamicsStep() {
    const dynamicsForm = document.getElementById(HYBRID_DYNAMICS_STEP_URL);
    if (dynamicsForm === null) {
        return;
    }
    const isLastStep = _activeHybridForm() === dynamicsForm;

    doAjaxCall({
        type: "GET",
        url: HYBRID_DYNAMICS_STEP_URL,
        success: function (response) {
            const fragment = _asFragment(response);
            const newForm = fragment.querySelector("form");
            if (newForm === null || newForm.id !== HYBRID_DYNAMICS_STEP_URL) {
                return;
            }
            dynamicsForm.replaceWith(fragment);
            if (!isLastStep) {
                // steps are still stacked under it, so it stays the read-only record of a finished step
                _lockHybridForm(document.getElementById(HYBRID_DYNAMICS_STEP_URL));
            }
        }
    });
}

/**
 * Store the dynamics configured for every Subnetwork. Only then does the wizard step summarising them
 * change, which is why the answer replaces that step; reloading it also refreshes the column, so the
 * two always agree about what is configured.
 */
function hybridSaveSubnetworkDynamics() {
    const dynamicsForm = document.getElementById(HYBRID_DYNAMICS_STEP_URL);

    doAjaxCall({
        type: "POST",
        url: "/burst/hybrid/save_subnetwork_dynamics/",
        success: function (response) {
            const fragment = _asFragment(response);
            const newForm = fragment.querySelector("form");

            if (dynamicsForm === null || newForm === null || newForm.id !== HYBRID_DYNAMICS_STEP_URL) {
                // the configuration is no longer where we left it, e.g. the Connectivity went missing
                _replaceHybridFragments(response);
                return;
            }

            // the summary on that step changes, the steps configuring the Subnetwork stay as they are
            const isLastStep = _activeHybridForm() === dynamicsForm;
            dynamicsForm.replaceWith(fragment);
            if (!isLastStep) {
                _lockHybridForm(document.getElementById(HYBRID_DYNAMICS_STEP_URL));
            }
            displayMessage("Subnetwork dynamics saved.");
        },
        error: function () {
            displayMessage("The Subnetwork dynamics could not be saved.", "errorMessage");
        }
    });
}
