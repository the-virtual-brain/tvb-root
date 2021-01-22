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

/* globals doAjaxCall, displayMessage */


/**
 * Creates the operation select, the input and the drop button
 */
function _FIL_createUiForFilterType(filter, newDiv, isDate){
    var operation = $('<select></select>');
    for (var j = 0; j < filter.operations.length; j++) {
        var op = filter.operations[j];
        operation.append(new Option(op, op));
    }
    var input = $('<input type="text" name="values"/>');
    var dropButton = $('<input type="button" value=" Drop Filter "/>');
    newDiv.append(operation, input, dropButton);

    dropButton.click(function () {
        newDiv.remove();
    });

    if (isDate){
        input.datepicker({
            changeMonth: true,
            changeYear: true,
            showOn: "both",
            buttonImage: "/static/style/img/calendar.png",
            buttonImageOnly: true,
            dateFormat : "mm-dd-yy",
            buttonText: "pick a date"
        });
    }
}

function addFilter(div_id, filters) {
    //Create a new div for the filter
    var newDiv = $('<div class="user_trigger"> <label> Filter : </label> </div>');
    $('#' + div_id).append(newDiv);

    //This will be the select row to filter by
    var filter = $('<select></select>');

    //Iterate over the filters dictionary and create the possible values
    var filter_names = Object.keys(filters).sort();
    for (var i = 0; i < filter_names.length; i++) {
        var k = filter_names[i];
        filter.append( new Option(filters[k].display, k));
    }

    newDiv.append(filter);
    var first_filter = filters[filter_names[0]];
    _FIL_createUiForFilterType(first_filter, newDiv, first_filter.type === 'date');

    filter.change(function () {
        // remove all nodes to the right of filter
        var children = newDiv.children();
        for (var i = 2; i < children.length; i++){
            $(children[i]).remove();
        }
        // recreate them
        _FIL_createUiForFilterType(filters[this.value], newDiv, filters[this.value].type === 'date');
    })
}



/** gather all the data from the filters */
function _FIL_gatherData(divId, uiValue){
    var children = $('#'+divId).children('div');
    var default_fields = [], default_operations = [], default_values = [];
    var user_fields = [], user_operations = [], user_values = [];
    var runtime_fields = [], runtime_operations = [], runtime_values = [], runtime_reverse_filtering_values = [];

    for (var i = 0; i < children.length; i++) {
        var elem = children[i].children;
        //Get info about the filters.
        if (elem[3].value.trim().length > 0) {
            var value = elem[3].value.trim();

            if (children[i].className === "user_trigger") {
                user_fields.push(elem[1].value);
                user_operations.push(elem[2].value);
                user_values.push(value);
            } else {
                if (children[i].className.endsWith('runtime_trigger')) {
                    let value_from_field = $('#' + children[i].className.replace('_runtime_trigger', '')).val();
                    if(value === "default_runtime_value"){
                        value = value_from_field;
                        runtime_reverse_filtering_values.push('');
                    }else{
                        runtime_reverse_filtering_values.push(value_from_field);
                    }

                    runtime_fields.push(elem[1].value);
                    runtime_operations.push(elem[2].value);
                    runtime_values.push(value);
                } else {
                    default_fields.push(elem[1].value);
                    default_operations.push(elem[2].value);
                    default_values.push(value);
                }
            }
            displayMessage("Filters processed");
        }
        else {
            displayMessage("Please set a value for all the filters.", "errorMessage");
            return;
        }
    }

    return {default_filters: {default_fields: default_fields, default_operations: default_operations, default_values:
            default_values}, user_filters: {user_fields: user_fields, user_operations: user_operations, user_values:
            user_values}, runtime_filters: {runtime_fields: runtime_fields, runtime_operations: runtime_operations,
            runtime_values: runtime_values, runtime_reverse_filtering_values: runtime_reverse_filtering_values}};
}

function applyUserFilters(datatypeIndex, divId, name, gatheredData) {
    if (!gatheredData) {
        //gather all the data from the filters and make an
        //ajax request to get new data
        gatheredData = _FIL_gatherData(divId);
        if (gatheredData == null) {
            return;
        }
    }

    if (document.getElementsByName(name).length < 1) {
        // Hidden elements from simulator interface or missing parameters will be ignored.
        displayMessage("Filter could not be applied! " + name, "infoMessage");
        return;
    }

    var dt_class_start_index = datatypeIndex.lastIndexOf('.');
    var dt_module = datatypeIndex.substring(0, dt_class_start_index);
    var dt_class = datatypeIndex.substring(dt_class_start_index + 1, datatypeIndex.length);

    var select_field = document.getElementById(name);
    var has_all_option = false;
    var has_none_option = false;

    if (select_field.options[0] && select_field.options[0].innerHTML === "None"){
        has_none_option = true;
    }

    if (select_field.options[select_field.options.length - 1] && select_field.options[select_field.options.length - 1].innerHTML === "All"){
        has_all_option = true;
    }

    //Make a request to get new data
    doAjaxCall({
        type: 'POST',
        url: "/flow/get_filtered_datatypes/" + dt_module + '/' + dt_class + '/' +
            $.toJSON(gatheredData.default_filters) + '/' + $.toJSON(gatheredData.user_filters) +
            '/' + $.toJSON(gatheredData.runtime_filters) + '/' + has_all_option + '/' + has_none_option,
        success: function (response) {
            if (!response) {
                displayMessage(`No results for the ${name} filtering!`, "warningMessage");

            }
            const t = document.createRange().createContextualFragment(response);
            let i, length = select_field.options.length - 1;
            for (i = length; i >= 0; i--) {
                select_field.remove(i);
            }
            select_field.appendChild(t);
        },
        error: function (response) {
            displayMessage("Invalid filter data.", "errorMessage");
        }
    });
}

function applyRuntimeFilters(name, selected_value){

    if($('.' + name + '_runtime_trigger').length === 0){
        return;
    }

    var form = $('#' + name).closest('form');
    let form_action = form[0].action;
    let algorithm_id_start = form_action.lastIndexOf('/');
    let algorithm_id = form_action.substring(algorithm_id_start + 1, form_action.length);

    algorithm_id_start = algorithm_id.lastIndexOf('?');
    if(algorithm_id_start!==-1){
        algorithm_id = algorithm_id.substring(0, algorithm_id_start)
    }

    let select_fields = form.find('select.dataset-selector');
    var fields_and_default_filters = {};
    var fields_and_user_filters = {};
    var fields_and_runtime_filters = {};

    var is_runtime_filtering = false;
    let filter_values;
    for (let i = 0; i < select_fields.length; i++) {
        filter_values = _FIL_gatherData(select_fields[i].id + 'data_select', selected_value);
        filter_values.runtime_filters['ui_value'] = select_fields[i].value;
        fields_and_default_filters[select_fields[i].id] = filter_values.default_filters;
        fields_and_user_filters[select_fields[i].id] = filter_values.user_filters;
        fields_and_runtime_filters[select_fields[i].id] = filter_values.runtime_filters;

        if(filter_values.runtime_filters['runtime_fields'].length > 0){
            is_runtime_filtering = true;
        }
    }

    if(is_runtime_filtering) {
        doAjaxCall({
            type: 'POST',
            url: "/flow/get_runtime_filtered_form/" + algorithm_id + '/' + $.toJSON(fields_and_default_filters) +
                '/' + $.toJSON(fields_and_user_filters) + '/' + $.toJSON(fields_and_runtime_filters),
            success: function (response) {
                const t = document.createRange().createContextualFragment(response);

                let adapters_div = $('.adaptersDiv');
                adapters_div.children('fieldset').replaceWith(t);

                for(var key in fields_and_user_filters){
                    const divId = key + 'data_select';
                    for(var i=0; i<fields_and_user_filters[key]['user_fields'].length; i++) {
                        let field_df = JSON.parse($('#' + key + '_df').val());
                        addFilter(divId, field_df);
                        var children = $('#'+divId).children('div');
                        var elem = children[children.length - 1].children;
                        elem[1].value = fields_and_user_filters[key]['user_fields'][i];
                        elem[2].value = fields_and_user_filters[key]['user_operations'][i];
                        elem[3].value = fields_and_user_filters[key]['user_values'][i];
                    }
                }
            },
            error: function (response) {
                displayMessage("Invalid filter data.", "errorMessage");
            }
        });
    }
}