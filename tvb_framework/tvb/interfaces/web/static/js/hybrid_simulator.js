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
// the cockpit steps, in wizard order, used to rebuild the stack when a step is no longer on screen
const HYBRID_WIZARD_STEPS = ["/burst/hybrid/set_connectivity", HYBRID_SUBNETWORKS_STEP_URL];

function _hybridFormsDiv() {
    return document.getElementById(HYBRID_FORMS_DIV);
}

/** The step currently being configured: the last one of the wizard stack. */
function _activeHybridForm() {
    const forms = _hybridFormsDiv().querySelectorAll("form");
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

/** The Results tree of this page. bursts.js, which owns the cockpit one, is not loaded here. */
function displayHybridResultsTree() {
    updateTree("#treeOverlay", null, JSON.stringify({'type': 'from_burst', 'value': "0"}));
    $("#div-burst-tree").show();
}

// ---------------------------------------------------------------- wizard stack

/**
 * Turn a form into the read-only record of a step that is already done: fields greyed out, buttons
 * hidden. This is what the classic cockpit does when you move on.
 */
function _lockHybridForm(form) {
    form.querySelectorAll("button").forEach(function (button) {
        button.style.visibility = "hidden";
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
    // the wizard buttons are type="button" so nothing would submit anyway, but keep the guard for
    // any caller that does arrive through a real event. window.event only exists during dispatch.
    if (typeof event !== "undefined" && event !== null) {
        event.preventDefault();
    }
    const formData = $(currentForm).serialize();
    doAjaxCall({
        type: "POST",
        url: $(currentForm).attr("action"),
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
