/**
 * TheVirtualBrain-Framework Package. This package holds all Data Management, and
 * Web-UI helpful to run brain-simulations. To use it, you also need to download
 * TheVirtualBrain-Scientific Package (for simulators). See content of the
 * documentation-folder for more details. See also http://www.thevirtualbrain.org
 *
 * (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
            buttonImage: deploy_context + "/static/style/img/calendar.png",
            buttonImageOnly: true,
            dateFormat : "mm-dd-yy",
            buttonText: "pick a date"
        });
    }
}

function addFilter(div_id, filters) {
    //Create a new div for the filter
    var newDiv = $('<div> <label> Filter : </label> </div>');
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
    });
}

/** gather all the data from the filters */
function _FIL_gatherData(divId){
    var children = $('#'+divId).children('div');
    var fields = [];
    var operations = [];
    var values = [];

    for (var i = 0; i < children.length; i++) {
        var elem = children[i].children;
        //Get info about the filters.
        if (elem[3].value.trim().length > 0) {
            fields.push(elem[1].value);
            operations.push(elem[2].value);
            values.push(elem[3].value.trim());
            displayMessage("Filters processed");
        } else {
            displayMessage("Please set a value for all the filters.", "errorMessage");
            return;
        }
    }
    if (fields.length === 0 && operations.length === 0 && values.length === 0) {
        displayMessage("Cleared filters");
    }

    return { fields: fields, operations: operations, values: values};
}

function applyFilters(datatypeIndex, divId, name, gatheredData) {
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
        url: "/flow/get_filtered_datatypes/" + dt_module + '/' + dt_class + '/' + $.toJSON(gatheredData) + '/' +
            has_all_option + '/' + has_none_option,
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


/**
 * After the user executes a filter than we have to replace the select with the old option with
 * the select which contains only the options that satisfies the filters.
 *
 * @param parentDiv the parent div in which is located the select
 * @param newSelect the html that contains the new select
 * @param selectName the name of the old select
 */
function replaceSelect(parentDiv, newSelect, selectName) {
    var allChildren = parentDiv.children;
    for (var j = 0; j < allChildren.length; j++) {
        if (allChildren[j].nodeName == 'SELECT' && allChildren[j].name == selectName) {
            $(newSelect).insertAfter($(allChildren[j]));
            parentDiv.removeChild(allChildren[j]);
            break;
        }
    }
}

/**
 * Filter fields which are linked with current entity.
 * @param {list} linkedDataList list of lists.
 * @param {string} currentSelectedGID for current input
 * @param {string} treeSessionKey Key
 */
function filterLinked(linkedDataList, currentSelectedGID, treeSessionKey) {
    if (currentSelectedGID.length < 1) {
        return;
    }
    for (var i = 0; i < linkedDataList.length; i++) {
        var linkedData = linkedDataList[i];
        var elemName = linkedData.linked_elem_name;

        var filterField = linkedData.linked_elem_field;
        var filterData = {
            'fields': [filterField],
            'operations': ["in"],
            'values': [currentSelectedGID.split(' ')]
        };

        if (!linkedData.linked_elem_parent_name && !linkedData.linked_elem_parent_option) {
            applyFilters("", elemName + 'data_select', elemName, treeSessionKey, filterData);
        }

        var linkedInputName = linkedData.linked_elem_parent_name + "_parameters_option_";
        var parentDivID = 'data_' + linkedData.linked_elem_parent_name;

        if (linkedData.linked_elem_parent_option) {
            linkedInputName = linkedInputName + linkedData.linked_elem_parent_option + "_" + elemName;
            parentDivID += linkedData.linked_elem_parent_option;
            applyFilters(parentDivID, linkedInputName + 'data_select', linkedInputName, treeSessionKey, filterData);
        } else {
            $("select[id^='" + linkedInputName + "']").each(function () {
                if ($(this)[0].id.indexOf("_" + elemName) < 0) {
                    return;
                }
                var option_name = $(this)[0].id.replace("_" + elemName, '').replace(linkedInputName, '');
                linkedInputName = $(this)[0].id;
                parentDivID += option_name; // todo : possible bug. option names will be concatenated many times if this each runs more than once
                applyFilters(parentDivID, linkedInputName + 'data_select', linkedInputName, treeSessionKey, filterData);
            });
        }
    }
}
