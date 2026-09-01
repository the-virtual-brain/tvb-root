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

/* globals doAjaxCall, renderWithMathjax, displayMessage, setupMenuEvents, displayBurstTree,
   HYBRID_SUBNETWORKS */

/**
 * The Hybrid Simulator wizard, following the classic Simulator Cockpit behaviour: pressing Next keeps
 * the step you just filled in on screen as a disabled, read-only form and appends the next one under
 * it, so the whole configuration stays visible while it is built up.
 *
 * The Subnetwork grouping board is the exception. It is a full width detour rather than a wizard step,
 * so it replaces the stack while it is open, and the stack is rebuilt when you come back from it.
 */

const HYBRID_FORMS_DIV = "hybrid-simulator-forms";
// the cockpit steps, in wizard order, used to rebuild the stack after the grouping board
const HYBRID_WIZARD_STEPS = ["/burst/hybrid/set_connectivity", "/burst/hybrid/set_subnetworks"];

function _hybridFormsDiv() {
    return document.getElementById(HYBRID_FORMS_DIV);
}

function _resetHybridLayout() {
    // The grouping board widens the page; every other fragment belongs in the cockpit layout.
    // Probed rather than assumed: a cached older hybrid_subnetworks.js would otherwise throw here
    // and leave the whole wizard unable to render.
    if (typeof HYBRID_SUBNETWORKS !== "undefined" && typeof HYBRID_SUBNETWORKS.resetLayout === "function") {
        HYBRID_SUBNETWORKS.resetLayout();
    }
}

function _asFragment(response) {
    // createContextualFragment keeps inline scripts executable, which the grouping board needs
    return document.createRange().createContextualFragment(response);
}

function _afterHybridRender() {
    if (typeof setupMenuEvents === "function") {
        setupMenuEvents();
    }
    $("button.btn-next").last().focus();
}

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
    _resetHybridLayout();
    renderWithMathjax($(_hybridFormsDiv()), fragment);
    _afterHybridRender();
}

/** Replace everything on screen with a single fragment. */
function _replaceHybridFragments(response) {
    _resetHybridLayout();
    renderWithMathjax($(_hybridFormsDiv()), _asFragment(response), true);
    _afterHybridRender();
}

/**
 * Rebuild the wizard stack from scratch, loading the given steps in order and leaving every step but
 * the last one read-only. Used when returning from the full width grouping board.
 */
function _renderHybridStack(stepUrls) {
    const container = _hybridFormsDiv();
    _resetHybridLayout();
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
                _resetHybridLayout();
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

/** Leave the full width grouping board and rebuild the cockpit wizard behind it. */
function hybridBackToSimulator() {
    _renderHybridStack(HYBRID_WIZARD_STEPS);
}

/** Replace the wizard with a single fragment, used to open the grouping board. */
function hybridLoadFragment(fragmentUrl) {
    doAjaxCall({
        type: "GET",
        url: fragmentUrl,
        success: _replaceHybridFragments,
        error: function () {
            displayMessage("Hybrid simulator parameters could not be loaded.", "errorMessage");
        }
    });
}
