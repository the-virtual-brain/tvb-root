/**
 * TheVirtualBrain-Framework Package. This package holds all Data Management, and
 * Web-UI helpful to run brain-simulations. To use it, you also need do download
 * TheVirtualBrain-Scientific Package (for simulators). See content of the
 * documentation-folder for more details. See also http://www.thevirtualbrain.org
 *
 * (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
 *
 * This program is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License along with this
 * program.  If not, see <http://www.gnu.org/licenses/>.
 *
 **/

/* global doAjaxCall, displayMessage, minimizeColumn, MathJax,
   PSEDiscreet_BurstDraw, PSEDiscreet_RedrawResize, Isocline_MainDraw */

/*
 * SIMULATOR (BURST) related JS calls.
 *  - Configure and launch a new burst; 
 *  - Load a previous executed burst; 
 *  - Show progress.
 */

var sessionStoredBurstID = "";

//A list of selected portlets id. Used to correctly change/update the portlet checkboxes for each tab.
var selectedPortlets;

// Class mapping to the active burst entry
var ACTIVE_BURST_CLASS = 'burst-active';
// Class mapping to a workflow from a group launch
var GROUP_BURST_CLASS = 'burst-group-expanded';
// When user edits a title, we won't overwrite that (e.g. by adding 'Copy of' prefix)
var user_edited_title = false;


function clone(object_) {
    return JSON.parse(JSON.stringify(object_));
}

/*************************************************************************************************************************
 *            HISTORY COLUMN
 *************************************************************************************************************************/

/*
 * When clicking on the New Burst Button reset to defaults for simulator interface and portlets.
 */
function resetToNewBurst() {
    doAjaxCall({
        type: "POST",
        url: '/burst/reset_simulator_configuration/',
        success: function (response) {
            let simParamElem = $("#div-simulator-parameters");
            simParamElem.html(response);
            displayBurstTree(undefined);
            displayMessage("Completely new configuration loaded!");
            changeBurstHistory(null, true);
            $("button.btn-next").first().focus();
        },
        error: function () {
            displayMessage("We encountered an error while generating the new simulation. Please try reload and then check the logs!", "errorMessage");
        }
    });
}

/*
 * When clicking the copy button on a burst-history entry, a clone of that burst is prepared.
 */
function copyBurst(burstID, first_wizzard_form_url) {
    doAjaxCall({
        type: "POST",
        url: '/burst/get_last_fragment_url/' + burstID,
        showBlockerOverlay: true,
        success: function (response) {
            stop_at_url = response;
            doAjaxCall({
                type: "POST",
                url: '/burst/copy_simulator_configuration/' + burstID,
                showBlockerOverlay: true,
                success: function (response) {
                    let simParamElem = $("#div-simulator-parameters");
                    simParamElem.html(response);
                    renderAllSimulatorForms(first_wizzard_form_url, stop_at_url, function() {
                        const newName = $("#input_simulation_name_id").val();
                        fill_burst_name(newName, false);
                    });
                    changeBurstHistory(null, true);
                    displayBurstTree(undefined);
                    displayMessage("A copy of previous simulation was prepared for you!");
                },
                error: function () {
                    displayMessage("We encountered an error while generating a copy of the simulation. Please try reload and then check the logs!", "errorMessage");
                }
            });
        }
    });
}

function renderAllSimulatorForms(url, stop_at_url = '', onFinishFunction = null) {
    const simulator_params = document.getElementById('div-simulator-parameters');
    if (stop_at_url !== url) {
        doAjaxCall({
            type: "GET",
            url: url,
            success: function (response) {
                const t = document.createRange().createContextualFragment(response);
                simulator_params.appendChild(t);
                MathJax.Hub.Queue(["Typeset", MathJax.Hub, "div-simulator-parameters"]);

                const next_url = $(response).attr("action");
                if (next_url && next_url.length > 0) {
                    renderAllSimulatorForms(next_url, stop_at_url, onFinishFunction);
                }
            }
        });
    } else {
        setInitialFocusOnButton(simulator_params);
        if (onFinishFunction != null) {
            onFinishFunction();
        }
    }
}

/*
 * Reload entire Burst-History column.
 */
function loadBurstHistory() {
    doAjaxCall({
        type: "POST",
        url: '/burst/load_burst_history',
        cache: false,
        async: false,
        success: function (r) {
            const historyElem = $('#section-view-history');
            historyElem.empty();
            historyElem.append(r);
            MathJax.Hub.Queue(["Typeset", MathJax.Hub, "section-view-history"]);
            //setupMenuEvents(historyElem);
        },
        error: function () {
            displayMessage("Simulator data could not be loaded properly..", "errorMessage");
        }
    });
    scheduleNewUpdate(false);
}

/*
 * Periodically look for updates in burst history status and update their classes accordingly.
 */
function updateBurstHistoryStatus() {
    let burst_ids = [];
    // todo: do not store app state in css classes
    $("#burst-history").find("li").each(function () {
        if (this.className.indexOf('burst-started') >= 0) {
            if (this.id.indexOf('burst_id_') >= 0) {
                const id = this.id.replace('burst_id_', '');
                burst_ids.push(id);
            }
        }
    });
    if (burst_ids.length > 0) {
        doAjaxCall({
            type: "POST",
            data: {'burst_ids': JSON.stringify(burst_ids)},
            url: '/burst/get_history_status',
            success: function (r) {
                let finalStatusReceived = false;
                let changedStatusOnCurrentBurst = false;
                let result = $.parseJSON(r);

                for (let i = 0; i < result.length; i++) {
                    if (result[i][1] !== 'running') {
                        finalStatusReceived = true;
                        if (result[i][0] === sessionStoredBurstID) {
                            changedStatusOnCurrentBurst = true;
                        }
                    }
                }
                _updateBurstHistoryElapsedTime(result);
                scheduleNewUpdate(finalStatusReceived, changedStatusOnCurrentBurst);
            },
            error: function () {
                displayMessage("Error during simulation status update...", "errorMessage");
                scheduleNewUpdate(false);
            }
        });
    }
}

/**
 * Update the burst detail pop-up of running simulations with the elapsed processing time.
 */
function _updateBurstHistoryElapsedTime(result) {
    for (let i = 0; i < result.length; i++) {
        if (result[i][1] === 'running') {
            let el = $('#burst-history').find(' li .burst-prop-processtime span')[i];
            $(el).text(' ~ ' + result[i][4]);
        }
    }
}

/**
 * Schedule burst-history section update, in the next 5 seconds.
 * If "withFullUpdate" is true, then a full history section replacement happens before the periodical update.
 */
function scheduleNewUpdate(withFullUpdate, refreshCurrent) {
    if ($('#burst-history').length !== 0) {
        if (withFullUpdate) {
            loadBurstHistory();
            changeBurstHistory(sessionStoredBurstID, false);
            if (refreshCurrent) {
                loadBurstReadOnly(sessionStoredBurstID,  '/burst/set_connectivity');
            }
        } else {
            setTimeout(updateBurstHistoryStatus, 5000);
        }
    }
}

/*
 * Cancel or Remove the burst entity given by burst_id. Also update the history column accordingly.
 */
function cancelOrRemoveBurst(burst_id) {
    doAjaxCall({
        type: "POST",
        url: '/burst/cancel_or_remove_burst/' + burst_id,
        showBlockerOverlay: true,
        success: function () {
            loadBurstHistory();
            if (sessionStoredBurstID === burst_id) {
                resetToNewBurst();
            } else {
                changeBurstHistory(sessionStoredBurstID, false);
            }
        },
        error: function () {
            displayMessage("Could not cancel/remove simulation.", "errorMessage");
        }
    });
}

/*
 * Create a input field that should be displayed on top of the HREF with the
 * burst name. On lost focus just set that name as the new burst name and update the
 * entity.
 */
function renameBurstEntry(burst_id, new_name_id) {
    const newValue = document.getElementById(new_name_id).value;
    doAjaxCall({
        type: "POST",
        async: false,
        url: '/burst/rename_burst/' + burst_id + '/' + newValue,
        success: function (r) {
            const result = $.parseJSON(r);
            if ('success' in result) {
                displayMessage(result.success);
                $("#burst_id_" + burst_id + " a").html(newValue);
                if (sessionStoredBurstID === burst_id + "") {
                    fill_burst_name(newValue, true);
                }
            } else {
                displayMessage(result.error, "errorMessage");
            }
        },
        error: function () {
            displayMessage("Error when renaming simulation.", "errorMessage");
        }
    });
}

/*
 * Load a given burst entry from history.
 */
function changeBurstHistory(burst_id, reset_name) {

    $("#burst-history").find("li").each(function () {
        $(this).removeClass(ACTIVE_BURST_CLASS);
        $(this).removeClass(GROUP_BURST_CLASS);
    });

    if (burst_id === null || burst_id === "") {
        sessionStoredBurstID = "";
        if (reset_name) {
            fill_burst_name("", false);
        }
        return;
    } else {
        sessionStoredBurstID = burst_id;
    }

    const selectedBurst = $("#burst_id_" + burst_id);
    fill_burst_name(selectedBurst[0].children[0].text, true);
    selectedBurst.addClass(ACTIVE_BURST_CLASS);
}

/*************************************************************************************************************************
 *          MIDDLE COLUMN (SIMULATION PARAMETERS)
 *************************************************************************************************************************/

/**
 * Submit currently set simulation parameters.
 * For Model-Visual-Setter important are: Connectivity and Model.
 */
function configureModel(actionUrl) {
    // Go to SetupModelParams page.
    const myForm = document.createElement("form");
    myForm.method = "POST";
    myForm.action = actionUrl;
    document.body.appendChild(myForm);
    myForm.submit();
    document.body.removeChild(myForm);
}

function configureModelParamsOnRegions() {
    configureModel("/burst/modelparameters/regions/");
}

function configureModelParamsOnSurface() {
    configureModel("/spatial/modelparameters/surface/edit_model_parameters/");
}

function configureNoiseParameters() {
    configureModel("/burst/noise/");
}

function toggleConfigSurfaceModelParamsButton() {
    var selectorSurfaceElem = $("select[name='surface']");
    selectorSurfaceElem.unbind('change.configureSurfaceModelParameters');
    selectorSurfaceElem.bind('change.configureSurfaceModelParameters', function () {
        var selectedValue = this.value;
        var show = selectedValue != null && selectedValue != 'None' && selectedValue.length !== 0;
        $("#configSurfaceModelParam").toggle(show);
    });
    selectorSurfaceElem.trigger("change");
    if (selectorSurfaceElem.length < 1) {
        $("#configSurfaceModelParam").hide();
    }
}

/*************************************************************************************************************************
 *          FUNCTIONS FOR RANGE PARAMETERS CALCULATION
 *************************************************************************************************************************/

function _calculateValuesInRange(pse_param_lo, pse_param_hi, pse_param_step){
    const param_difference = pse_param_hi - pse_param_lo;
    let pse_param_number = Math.floor(param_difference / pse_param_step);
    const remainder_param = param_difference % pse_param_step;
    if(remainder_param !== 0){
        pse_param_number = pse_param_number + 1;
    }

    return pse_param_number;
}

function _getRangeValueForGuidParameter(pse_param_guid){
    pse_param_guid = pse_param_guid[0];
    const guid_list = pse_param_guid.value.split(',');
    return guid_list.length;
}

function _computeRangeNumberForParamPrefix(prefix){
    //check if param exists and does not have guid
    let pse_param_lo = $("#".concat(prefix, "_lo"));
    if(pse_param_lo.length !== 0){
        pse_param_lo = pse_param_lo[0].valueAsNumber;
        const pse_param_hi = $("#".concat(prefix, "_hi"))[0].valueAsNumber;
        const pse_param_step = $("#".concat(prefix, "_step"))[0].valueAsNumber;
        return _calculateValuesInRange(pse_param_lo, pse_param_hi, pse_param_step);
    }

    //check if we have param with guid
    let pse_param_guid = $("#".concat(prefix, "_guid"));
    if(pse_param_guid.length !== 0){
        return _getRangeValueForGuidParameter(pse_param_guid);
    }

    return 1;
}

function _displayPseSimulationMessage() {
    const THREASHOLD_WARNING = 500;
    const THREASHOLD_ERROR = 50000;

    pse_param1_number = _computeRangeNumberForParamPrefix('pse_param1');
    pse_param2_number = _computeRangeNumberForParamPrefix('pse_param2');

    let nrOps = pse_param1_number * pse_param2_number;
    let className = "infoMessage";

    if (nrOps > THREASHOLD_WARNING) {
        className = "warningMessage";
    }
    if (nrOps > THREASHOLD_ERROR) {
        className = "errorMessage";
    }
    if (nrOps > 1) {
        // Unless greater than 1, it is not a range, so do not display a possible confusing message.
        displayMessage("Range configuration: " + nrOps + " operations.", className);
    }
}

function setPseRangeParameters(){
    document.addEventListener('change', function(e){
        if(e.target && (e.target.id === 'pse_param1_lo' ||
        e.target.id === 'pse_param1_hi' ||
        e.target.id === 'pse_param1_step' ||
        e.target.id === 'pse_param2_lo' ||
        e.target.id === 'pse_param2_hi' ||
        e.target.id === 'pse_param2_step' ||
        e.target.id === 'pse_param1_guid' ||
        e.target.id === 'pse_param2_guid')){
            _displayPseSimulationMessage();
        }
    });
}

/*************************************************************************************************************************
 *            GENERIC FUNCTIONS
 *************************************************************************************************************************/

/*
 * If a burst is stored in session then load from there. Called on coming to burst page from a valid session.
 */
function initBurstConfiguration(currentBurstID, currentBurstName, selectedTab) {
    setPseRangeParameters();

    loadBurstHistory();
    changeBurstHistory(currentBurstID, true);
    fill_burst_name(currentBurstName, currentBurstID !== "");
    toggleConfigSurfaceModelParamsButton();

    if ('-1' === selectedTab) {
        $("#tab-burst-tree").click();
    }
}

/*
 * Given a burst id, load the simulator configuration in read-only mode
 */
function loadBurstReadOnly(burst_id, first_wizzard_form_url) {
    doAjaxCall({
        type: "POST",
        url: '/burst/get_last_fragment_url/' + burst_id,
        showBlockerOverlay: true,
        success: function (response) {
            stop_at_url = response;

            doAjaxCall({
                type: "POST",
                url: '/burst/load_burst_read_only/' + burst_id,
                showBlockerOverlay: true,
                success: function (response) {
                    let simParamElem = $("#div-simulator-parameters");
                    simParamElem.html(response);
                    renderAllSimulatorForms(first_wizzard_form_url, stop_at_url);
                    displayBurstTree(burst_id);
                    displayMessage("The simulation configuration was loaded for you!");
                },
                error: function () {
                    displayMessage("We encountered an error while loading a the simulation configuration. Please try reload and then check the logs!", "errorMessage");
                }
            })
        }
    });
}

/**
 * Method for updating title area according to current selected burst and its state.
 * @param {String} burstName
 * @param {bool} isReadOnly
 */
function fill_burst_name(burstName, isReadOnly) {
    const inputBurstName = $("#input-burst-name-id");
    const titleSimulation = $("#title-simulation");
    const titlePSE = $("#title-pse");
    const titlePortlets = $("#title-visualizers");

    inputBurstName.val(burstName);
    titleSimulation.empty();
    titlePortlets.empty();
    titlePSE.empty();

    if (isReadOnly) {
        titleSimulation.append("<mark>Review</mark> Simulation configuration for " + burstName);
        titlePortlets.append(burstName);
        titlePSE.append(burstName);
        inputBurstName.parent().parent().removeClass('is-created');
    } else {
        if (burstName !== '') {
            titleSimulation.append("<mark>Edit</mark> Simulation configuration for " + burstName);
            titlePortlets.append(burstName);
            titlePSE.append(burstName);
        } else {
            titleSimulation.append("<mark>Configure</mark> New simulation");
            titlePortlets.append("New simulation");
            titlePSE.append("New simulation");
        }
        inputBurstName.parent().parent().addClass('is-created');
    }
    user_edited_title = false;
}

function hideButtonsAfterLaunch(form_elements){
    for(var i = 0; i < form_elements.length; i++){
        if(form_elements[i].type === "button"){
            form_elements[i].style.visibility = "hidden";
        }
    }
}

function launchNewPSEBurst(currentForm) {
    _displayPseSimulationMessage();
    var form_data = $(currentForm).serialize();
    hideButtonsAfterLaunch(currentForm.elements);

    doAjaxCall({
        type: "POST",
        url: '/burst/launch_pse/',
        data: form_data,
        traditional: true,
        success: function (r) {
            loadBurstHistory();
            const result = $.parseJSON(r);
            if ('id' in result) {
                changeBurstHistory(result.id, true);
            }
            if ('error' in result) {
                displayMessage(result.error, "errorMessage");
            }
        },
        error: function () {
            displayMessage("Error when launching simulation. Please check te logs or contact your administrator.", "errorMessage");
        }
    });
}

/*
 * Get the data from the simulator and launch a new burst. On success add a new entry in the burst-history.
 * @param launchMode: 'new' 'branch' or 'continue'
 */
function launchNewBurst(currentForm, launchMode) {
    var form_data = $(currentForm).serialize(); //Encode form elements for submission
    hideButtonsAfterLaunch(currentForm.elements);

    displayMessage("You've submitted parameters for simulation launch! Please wait for preprocessing steps...", 'warningMessage');
    doAjaxCall({
        type: "POST",
        url: '/burst/launch_simulation/' + launchMode,
        data: form_data,
        traditional: true,
        success: function (response) {
            loadBurstHistory();
            const result = $.parseJSON(response);
            if ('id' in result) {
                changeBurstHistory(result.id, true);
            }
            if ('error' in result) {
                displayMessage(result.error, "errorMessage");
            }
        },
        error: function () {
            displayMessage("Error when launching simulation. Please check te logs or contact your administrator.", "errorMessage");
        }
    });
}

function setInitialFocusOnButton(simulator_params) {
    const current_url = simulator_params.lastElementChild.action;
    if (current_url !== undefined && current_url.includes('setup_pse')) {
        $('#launch_simulation').focus();
    } else if (current_url !== undefined && current_url.includes('launch_pse')) {
        $('#launch_pse').focus();
    } else {
        $("button.btn-next").last().focus();
    }
}

function previousWizzardStep(currentForm, previous_action, div_id = 'div-simulator-parameters') {
    const simulator_params = document.getElementById(div_id);
    simulator_params.removeChild(currentForm);

    var previous_form = document.getElementById(previous_action);
    var next_button = previous_form.elements.namedItem('next');
    var previous_button = previous_form.elements.namedItem('previous');
    var config_region_param_button = previous_form.elements.namedItem('configRegionModelParam');
    var config_surface_param_button = previous_form.elements.namedItem('configSurfaceModelParam');
    var config_noise_button = previous_form.elements.namedItem('configNoiseValues');
    var config_launch_button = previous_form.elements.namedItem('launch_simulation');
    var config_pse_button = previous_form.elements.namedItem('setup_pse');
    var config_launch_pse_button = previous_form.elements.namedItem('launch_pse');
    var config_branch_button = previous_form.elements.namedItem('branch_simulation');
    var fieldset = previous_form.elements[0];

    if (next_button != null) {
        next_button.style.visibility = 'visible';
    }
    if (previous_button != null) {
        previous_button.style.visibility = 'visible';
    }
    if (config_region_param_button != null) {
        config_region_param_button.style.visibility = 'visible';
    }
    if (config_surface_param_button != null) {
        config_surface_param_button.style.visibility = 'visible';
    }
    if (config_noise_button != null) {
        config_noise_button.style.visibility = 'visible';
    }
    if (config_launch_button != null){
        config_launch_button.style.visibility = 'visible';
    }
    if (config_pse_button != null){
        config_pse_button.style.visibility = 'visible';
    }
    if (config_launch_pse_button != null){
        config_launch_pse_button.style.visibility = 'visible';
    }
    if (config_branch_button != null){
        config_branch_button.style.visibility = 'visible';
    }
    fieldset.disabled = false;
    setInitialFocusOnButton(simulator_params);
}

function wizzard_submit(currentForm, success_function = null, div_id = 'div-simulator-parameters') {
    event.preventDefault(); //prevent default action
    var post_url = $(currentForm).attr("action"); //get form action url
    var request_method = $(currentForm).attr("method"); //get form GET/POST method
    var form_data = $(currentForm).serialize(); //Encode form elements for submission
    var next_button = currentForm.elements.namedItem('next');
    var previous_button = currentForm.elements.namedItem('previous');
    var config_region_param_button = currentForm.elements.namedItem('configRegionModelParam');
    var config_surface_param_button = currentForm.elements.namedItem('configSurfaceModelParam');
    var config_noise_button = currentForm.elements.namedItem('configNoiseValues');
    var config_launch_button = currentForm.elements.namedItem('launch_simulation');
    var config_pse_button = currentForm.elements.namedItem('setup_pse');
    var config_launch_pse_button = currentForm.elements.namedItem('launch_pse');
    var config_branch_button = currentForm.elements.namedItem('branch_simulation');
    var fieldset = currentForm.elements[0];

    $.ajax({
        url: post_url,
        type: request_method,
        data: form_data,
        success: function (response) {
            if (success_function != null) {
                success_function();
            } else {
                if (next_button != null) {
                    next_button.style.visibility = 'hidden';
                }
                if (previous_button != null) {
                    previous_button.style.visibility = 'hidden';
                }
                if (config_region_param_button != null) {
                    config_region_param_button.style.visibility = 'hidden';
                }
                if (config_surface_param_button != null) {
                    config_surface_param_button.style.visibility = 'hidden';
                }
                if (config_noise_button != null) {
                    config_noise_button.style.visibility = 'hidden';
                }
                if (config_launch_button != null){
                    config_launch_button.style.visibility = 'hidden';
                }
                if (config_pse_button != null){
                    config_pse_button.style.visibility = 'hidden';
                }
                if(config_launch_pse_button != null){
                    config_launch_pse_button.style.visibility = 'hidden';
                }
                if(config_branch_button != null){
                    config_branch_button.style.visibility = 'hidden';
                }
                fieldset.disabled = true;
                var t = document.createRange().createContextualFragment(response);
                const simulator_params = document.getElementById(div_id);
                simulator_params.appendChild(t);
                MathJax.Hub.Queue(["Typeset", MathJax.Hub, div_id]);
                setInitialFocusOnButton(simulator_params);
            }
        }
    })
}

function displayBurstTree(selectedBurstID) {
    let filterValue = {'type': 'from_burst', 'value': selectedBurstID};
    if (filterValue.value === undefined) {
        filterValue = {'type': 'from_burst', 'value': "0"};
    }
    updateTree("#treeOverlay", null, JSON.stringify(filterValue));
    $("#portlets-display").hide();
    $("#portlets-configure").hide();
    $("#portlet-param-config").hide();
    $("#div-burst-tree").show();
}