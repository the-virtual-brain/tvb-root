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

/**
 * Initializes the legend buffers, the color scheme component, and binds the click event.
 * It does not initialize webgl as that is done in the included left_template_brain_pick.html
 * @param minValue Minimum value of the LC sparse matrix - used for color scheme
 * @param maxValue Maximum value of the LC sparse matrix - used for color scheme
 * @param [selectedLocalConnectivity] if absent we are in the local connectivity page step2 and this value is retrieved from the dom
 */
function LCON_viewer_init(minValue, maxValue, selectedLocalConnectivity) {
    ColSch_initColorSchemeGUI(minValue, maxValue);

    LEG_initMinMax(minValue, maxValue);
    LEG_generateLegendBuffers();
    BASE_PICK_initLegendInfo(maxValue, minValue);

    $('#GLcanvas').click(function () {
        _displayGradientForThePickedVertex(selectedLocalConnectivity);
    });
}

function LCON_page_init(minValue, maxValue) {
    $('#GLcanvas').click(function () {
        BS_displayIndexForThePickedVertex();
    });
    LCON_viewer_init(minValue, maxValue);
    $("select[name='existentEntitiesSelect']").change(function () {
        BS_loadEntity();
    });
    LCONN_disableView('You are already in VIEW mode. If you want to display a different Local Connectivity entity just pick it from the selector menu above the visualizer.');
    LCONN_disableCreate('If you want to create a new Local Connectivity entity, go back to the EDIT page to set a new set of parameters.');
}
/**
 * Displays a gradient on the surface used by the selected local connectivity.
 * @param [selectedLocalConnectivity] the gid of the local connectivity
 */
function _displayGradientForThePickedVertex(selectedLocalConnectivity) {

    if (TRIANGLE_pickedIndex >= 0) {
        if (selectedLocalConnectivity == null) {
            selectedLocalConnectivity = $("select[name='existentEntitiesSelect']").val();
        }

        if (selectedLocalConnectivity == null || selectedLocalConnectivity == "None" ||
            selectedLocalConnectivity.trim().length == 0) {
            _drawDefaultColorBuffers();
            return;
        }

        let url = '/spatial/localconnectivity/compute_data_for_gradient_view?local_connectivity_gid=';
        url += selectedLocalConnectivity + "&selected_triangle=" + TRIANGLE_pickedIndex;
        doAjaxCall({
            async: false,
            url: url,
            success: _updateBrainColors
        });
        TRIANGLE_pickedIndex = GL_NOTFOUND;             // disable data retrieval until next triangle pick
    }
}

/**
 * Disable the view button in case we don't have some existing entity loaded
 */
function LCONN_disableView(message) {

    const stepButton = $("#lconn_step_2");
    stepButton[0].onclick = null;
    stepButton.unbind("click");
    stepButton.click(function () {
        displayMessage(message, 'infoMessage');
        return false;
    });
    stepButton.addClass("action-idle");
}

/**
 * Disable the create button and remove action in case we just loaded an entity.
 */
function LCONN_disableCreate(message) {
    const stepButton = $('#lconn_step_3');
    stepButton[0].onclick = null;
    stepButton.unbind("click");
    stepButton.click(function () {
        displayMessage(message, 'infoMessage');
        return false;
    });
    stepButton.addClass("action-idle");
}

/**
 * Enable the create button and add the required action to it in case some parameters have changed.
 */
function LCONN_enableCreate() {

    const stepButton = $('#lconn_step_3');
    stepButton[0].onclick = null;
    stepButton.unbind("click");
    stepButton.click(function () {
        createLocalConnectivity();
        return false;
    });
    stepButton.removeClass("action-idle");
}


/**
 * Collects the data defined for the local connectivity and submit it to the server.
 *
 * @param actionURL the url at which will be submitted the data
 * @param formId Form to be submitted.
 */
function LCONN_submitLocalConnectivityData(actionURL, formId) {
    const parametersForm = document.getElementById(formId);
    parametersForm.method = "POST";
    parametersForm.action = actionURL;
    parametersForm.submit();
}


/*
 * ---------------------------------------Pick related code starts here:
 * */

/**
 * Displays a gradient on the surface.
 *
 * @param data_from_server a json object which contains the data needed
 * for drawing a gradient on the surface.
 */
function _updateBrainColors(data_from_server) {

    data_from_server = $.parseJSON(data_from_server);
    const data = $.parseJSON(data_from_server.data);
    console.info("Showing " + data_from_server);
    console.info(data.length);
    BASE_PICK_updateBrainColors(data);
    displayMessage("Done displaying Local Connectivity profile for selected focal point ...")
}

/**
 * In case something changed in the parameters or the loaded local_connectivity is
 * set to None, just use this method to draw the 'default' surface with the gray coloring.
 */
function _drawDefaultColorBuffers() {
    if (BASE_PICK_brainDisplayBuffers.length === 0) {
        displayMessage("The load operation for the surface data is not completed yet!", "infoMessage");
        return;
    }
    BASE_PICK_buffer_default_color();
    drawScene();
}

// Following functions are used for updating the current LCONN on server and redraw the chart
function change_equation_params_form(base_url, method_to_call, current_equation, equation_params_div, fields_with_events) {
    let url = base_url + "/" + method_to_call + "/" + current_equation;
    $.ajax({
        url: url,
        type: 'POST',
        success: function (response) {
            var t = document.createRange().createContextualFragment(response);
            $("#" + equation_params_div).empty().append(t);
            MathJax.Hub.Queue(["Typeset", MathJax.Hub, equation_params_div]);
            set_events_on_form_fields(fields_with_events, base_url, true);
            plot_equation(base_url)
        }
    })
}

function set_lconn_param_and_redraw_chart(base_url, method_to_call, field_name, field_value) {
    let current_param = field_name + '=' + field_value;
    let url = base_url + '/' + method_to_call + '?' + current_param;
    $.ajax({
        url: url,
        type: 'POST',
        success: function () {
            plot_equation(base_url)
        }
    })
}

function set_events_on_form_fields(fields_with_events, url, only_equation_params = false) {
    let SURFACE_FIELD = 'set_surface';
    let EQUATION_FIELD = 'set_equation';
    let CUTOFF_FIELD = 'set_cutoff_value';
    let DISPLAY_NAME_FIELD = 'set_display_name';
    let EQUATION_PARAMS_FIELD = 'set_equation_param';

    if (only_equation_params === false) {
        $('select[name^="' + fields_with_events[SURFACE_FIELD] + '"]').change(function () {
            set_lconn_param_and_redraw_chart(url, SURFACE_FIELD, this.name, this.value)
        });
        $('input[name^="' + fields_with_events[CUTOFF_FIELD] + '"]').change(function () {
            set_lconn_param_and_redraw_chart(url, CUTOFF_FIELD, this.name, this.value)
        });
        $('input[name^="' + fields_with_events[DISPLAY_NAME_FIELD] + '"]').change(function () {
            set_lconn_param_and_redraw_chart(url, DISPLAY_NAME_FIELD, this.name, this.value)
        });

        //TODO: we want to have also support fields for this/ extract hardcoded strings
        let equation_select_fields = document.getElementsByName(fields_with_events[EQUATION_FIELD]);
        for (let i=0; i<equation_select_fields.length; i++) {
            equation_select_fields[i].onclick = function () {
                change_equation_params_form(url, EQUATION_FIELD, this.value, 'spatial_params',
                    fields_with_events)
            };
        }
    }
    $('input[name^="' + fields_with_events[EQUATION_PARAMS_FIELD] + '"]').change(function () {
        set_lconn_param_and_redraw_chart(url, EQUATION_PARAMS_FIELD, this.name, this.value)
    });
}

function plot_equation(base_url) {
    let url = base_url + '/get_equation_chart';
    doAjaxCall({
        async: false,
        type: 'GET',
        url: url,
        success: function (data) {
            $("#" + 'equationDivId').empty().append(data);
        }
    });
}